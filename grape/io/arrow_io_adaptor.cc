/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "grape/io/arrow_io_adaptor.h"

#include <glog/logging.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "arrow/api.h"
#include "arrow/csv/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/io/api.h"
#include "arrow/util/uri.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

namespace grape {

ArrowIOAdaptor::ArrowIOAdaptor(std::string location)
    : location_(std::move(location)),
      enable_partial_read_(false),
      total_parts_(0),
      index_(0) {
    is_s3_=false;
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
}

ArrowIOAdaptor::ArrowIOAdaptor(std::string location, std::string fspath)
    : location_(std::move(location)),
      fspath_(std::move(fspath)),
      enable_partial_read_(false),
      total_parts_(0),
      index_(0) {
  auto result = arrow::fs::FileSystemFromUriOrPath(fspath_);
  if (result.ok()) {
    is_s3_=true;
    fs_ = result.ValueOrDie();
  } else {
    is_s3_=false;
    fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
  }
}

ArrowIOAdaptor::~ArrowIOAdaptor() {
  Close();
  fs_.reset();
}

int64_t ArrowIOAdaptor::tell() {
  if (ifp_) {
    return ifp_->Tell().ValueOr(-1);
  }
  if (ofp_) {
    return ofp_->Tell().ValueOr(-1);
  }
  return -1;
}

bool ArrowIOAdaptor::seek(const int64_t offset, const FileLocation seek_from) {
  if (!ifp_) {
    return false;
  } else {
    switch (seek_from) {
    case kFileLocationBegin: {
      auto status = ifp_->Seek(offset);
    } break;
    case kFileLocationCurrent: {
      auto p = ifp_->Tell();
      if (p.ok()) {
        RETURN_ON_ARROW_ERROR(ifp_->Seek(p.ValueUnsafe() + offset));
      } else {
        return false;
      }
    } break;
    case kFileLocationEnd: {
      auto sz = ifp_->GetSize();
      if (sz.ok()) {
        RETURN_ON_ARROW_ERROR(ifp_->Seek(sz.ValueUnsafe() - offset));
      } else {
        return false;
      }
    } break;
    default: {
      return false;
    }
    }
    return true;
  }
}

void ArrowIOAdaptor::Open() { return this->Open("r"); }

void ArrowIOAdaptor::Open(const char* mode) {
  std::string tmp_location = (is_s3_? location_ : realPath(location_));
  if (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL) {
    int t = location_.find_last_of('/');

    if (t != -1) {
      std::string folder_path = location_.substr(0, t);
      if (access(folder_path.c_str(), 0) != 0) {
        MakeDirectory(folder_path);
      }
    }

    if (strchr(mode, 'w') != NULL) {
      DISCARD_ARROW_ERROR_AND_ASSIGN(
          ofp_, fs_->OpenOutputStream(tmp_location));
    } else {
      DISCARD_ARROW_ERROR_AND_ASSIGN(
          ofp_, fs_->OpenAppendStream(tmp_location));
    }
    return;
  } else {
    DISCARD_ARROW_ERROR_AND_ASSIGN(ifp_,
                                   fs_->OpenInputFile(tmp_location));

    if (!strchr(mode, 'b')) {
      // check the partial read flag
      if (enable_partial_read_) {
        setPartialReadImpl();
      }
      preReadPartialTable(enable_partial_read_);
    }
  }
}

bool ArrowIOAdaptor::Configure(const std::string& key,
                               const std::string& value) {
  return true;
}

bool ArrowIOAdaptor::SetPartialRead(const int index, const int total_parts) {
  // make sure that the bytes of each line of the file
  // is smaller than macro FINELINE
  if (index < 0 || total_parts <= 0 || index >= total_parts) {
    VLOG(1) << "error during set_partial_read with [" << index << ", "
            << total_parts << "]";
    return false;
  }
  if (ifp_ != nullptr) {
    VLOG(2) << "WARNING!! std::set partial read after open have no effect,"
               "You probably want to set partial before open!";
    return false;
  }
  enable_partial_read_ = true;
  index_ = index;
  total_parts_ = total_parts;

  return true;
}

bool ArrowIOAdaptor::preReadPartialTable(bool partial) {
  int index = index_;
  int64_t offset, nbytes;
  if (partial) {
    offset = partial_read_offset_[index];
    nbytes = partial_read_offset_[index + 1] - partial_read_offset_[index];
  } else {
    RETURN_ON_ERROR(seek(0, kFileLocationEnd));
    offset = 0;
    nbytes = tell();
  }
  batch_idx_ = 0;
  row_idx_ = 0;

#if defined(ARROW_VERSION) && ARROW_VERSION <= 9000000
  std::shared_ptr<arrow::io::InputStream> input =
      arrow::io::RandomAccessFile::GetStream(ifp_, offset, nbytes);
#else
  std::shared_ptr<arrow::io::InputStream> input;
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      input, arrow::io::RandomAccessFile::GetStream(ifp_, offset, nbytes));
#endif

  arrow::MemoryPool* pool = arrow::default_memory_pool();

  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  auto convert_options = arrow::csv::ConvertOptions::Defaults();
  read_options.autogenerate_column_names = true;
  parse_options.delimiter = delimiter_;

  std::shared_ptr<arrow::csv::TableReader> reader;
#if defined(ARROW_VERSION) && ARROW_VERSION >= 4000000
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      reader, arrow::csv::TableReader::Make(arrow::io::IOContext(pool), input,
                                            read_options, parse_options,
                                            convert_options));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      reader, arrow::csv::TableReader::Make(pool, input, read_options,
                                            parse_options, convert_options));
#endif

  auto result = reader->Read();
  if (!result.status().ok()) {
    if (result.status().message() == "Empty CSV file") {
      preread_table_ = nullptr;
      return true;
    } else {
      return false;
    }
  }
  preread_table_ = result.ValueOrDie();

  std::shared_ptr<arrow::Table> table = preread_table_;
  arrow::TableBatchReader batch_reader(*table);
  auto s = batch_reader.ReadAll(&batches_);

  return true;
}

std::string ArrowIOAdaptor::realPath(std::string const& path) {
  char absolute_path_c[LINESIZE];
  if (realpath(path.c_str(), absolute_path_c)) {}
  return std::string(absolute_path_c);
}

bool ArrowIOAdaptor::setPartialReadImpl() {
  RETURN_ON_ERROR(seek(0, kFileLocationEnd));
  int64_t total_file_size = tell();
  int64_t part_size = total_file_size / total_parts_;

  partial_read_offset_.resize(total_parts_ + 1, 0);
  partial_read_offset_[total_parts_] = total_file_size;

  // move breakpoint to the next of nearest character '\n'
  for (int i = 1; i < total_parts_; ++i) {
    partial_read_offset_[i] = i * part_size;

    if (partial_read_offset_[i] < partial_read_offset_[i - 1]) {
      partial_read_offset_[i] = partial_read_offset_[i - 1];
    } else {
      // traversing backwards to find the nearest character '\n',
      RETURN_ON_ERROR(seek(partial_read_offset_[i], kFileLocationBegin));
      int dis = 0;
      while (true) {
        char buffer[1];
        std::memset(buff, 0, sizeof(buffer));
        bool status = Read(buffer, 1);
        if (!status || buffer[0] == '\n') {
          break;
        } else {
          dis++;
        }
      }
      // move to next character of '\n'
      partial_read_offset_[i] += (dis + 1);
      if (partial_read_offset_[i] > total_file_size) {
        partial_read_offset_[i] = total_file_size;
      }
    }
  }

  int64_t file_stream_pos = partial_read_offset_[index_];
  RETURN_ON_ERROR(seek(file_stream_pos, kFileLocationBegin));
  return true;
}

bool ArrowIOAdaptor::ReadLine(std::string& line) {
  if (preread_table_ == nullptr) {
    return false;
  }
  std::shared_ptr<arrow::Table> table = preread_table_;

  if ((size_t) batch_idx_ >= batches_.size()) {
    return false;
  }
  std::shared_ptr<arrow::RecordBatch> batch = batches_[batch_idx_];
  size_t rows = batch->num_rows();

  if ((size_t) row_idx_ >= rows) {
    return false;
  }

  std::stringstream ss;
  for (int i = 0; i < table->num_columns(); i++) {
    auto array = table->column(i)->chunk(batch_idx_);
    ss << array->GetScalar(row_idx_).ValueOrDie()->ToString()
       << (i == (table->num_columns() - 1) ? "" : " ");
  }

  line = ss.str();
  if ((size_t) row_idx_ + 1 >= rows) {
    row_idx_ = 0;
    batch_idx_++;
  } else {
    row_idx_++;
  }
  return true;
}

bool ArrowIOAdaptor::ReadArchive(OutArchive& archive) {
  if (ifp_) {
    size_t length;
    RETURN_ON_ARROW_ERROR(
        ifp_->Read(sizeof(size_t), reinterpret_cast<void*>(&length)));
    archive.Allocate(length);
    RETURN_ON_ARROW_ERROR(
        ifp_->Read(length, reinterpret_cast<void*>(archive.GetBuffer())));
    return true;
  } else {
    VLOG(1) << "invalid operation.";
    return false;
  }
}

bool ArrowIOAdaptor::WriteArchive(InArchive& archive) {
  if (ofp_) {
    size_t length = archive.GetSize();
    RETURN_ON_ARROW_ERROR(ofp_->Write((void*) &length, sizeof(size_t)));
    RETURN_ON_ARROW_ERROR(ofp_->Write(archive.GetBuffer(), length));
    RETURN_ON_ARROW_ERROR(ofp_->Flush());
    return true;
  } else {
    VLOG(1) << "invalid operation.";
    return false;
  }
}

bool ArrowIOAdaptor::Read(void* buffer, size_t size) {
  if (ifp_ == nullptr) {
    return false;
  }
  auto r = ifp_->Read(size, buffer);
  if (r.ok()) {
    if (r.ValueUnsafe() < static_cast<int64_t>(size)) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

bool ArrowIOAdaptor::Write(void* buffer, size_t size) {
  if (ofp_) {
    auto status = ofp_->Write(buffer, size);
    return status.ok();
  } else {
    return false;
  }
}

void ArrowIOAdaptor::Close() {
  if (ifp_) {
    auto s1 = ifp_->Close();
  }
  if (ofp_) {
    auto status = ofp_->Flush();
    if (status.ok()) {
      auto s2 = ofp_->Close();
    }
  }
}

void ArrowIOAdaptor::MakeDirectory(const std::string& path) {
  std::string dir = path;
  int len = dir.size();
  if (dir[len - 1] != '/') {
    dir[len] = '/';
    len++;
  }
  std::string temp;
  for (int i = 1; i < len; i++) {
    if (dir[i] == '/') {
      temp = dir.substr(0, i);
      if (access(temp.c_str(), 0) != 0) {
        if (mkdir(temp.c_str(), 0777) != 0) {
          VLOG(1) << "failed operaiton.";
        }
      }
    }
  }
}

bool ArrowIOAdaptor::IsExist() {
  std::string path = location_;
  auto mfinfo = fs_->GetFileInfo(path);
  return mfinfo.ok() &&
         mfinfo.ValueUnsafe().type() != arrow::fs::FileType::NotFound;
}

}  // namespace grape
