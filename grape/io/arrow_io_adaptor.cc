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
    : file_(nullptr),
      location_(std::move(location)),
      using_std_getline_(false),
      enable_partial_read_(false),
      total_parts_(0),
      index_(0) {
  batch_idx_ = 0;
  row_idx_ = 0;
  fs_ = arrow::fs::FileSystemFromUriOrPath(location_, &location_).ValueOrDie();
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

void ArrowIOAdaptor::seek(const int64_t offset, const FileLocation seek_from) {
  if (!ifp_) {
    return;
  }
  switch (seek_from) {
  case kFileLocationBegin: {
    ifp_->Seek(offset);
  } break;
  case kFileLocationCurrent: {
    auto p = ifp_->Tell();
    if (p.ok()) {
      ifp_->Seek(p.ValueUnsafe() + offset);
    } else {
      return;
    }
  } break;
  case kFileLocationEnd: {
    auto sz = ifp_->GetSize();
    if (sz.ok()) {
      ifp_->Seek(sz.ValueUnsafe() - offset);
    } else {
      return;
    }
  } break;
  default: {
    return;
  }
  }
}

void ArrowIOAdaptor::Open() { return this->Open("r"); }

void ArrowIOAdaptor::Open(const char* mode) {
  if (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL) {
    int t = location_.find_last_of('/');
    if (t != -1) {
      std::string folder_path = location_.substr(0, t);
      if (access(folder_path.c_str(), 0) != 0) {
        MakeDirectory(folder_path);
      }
    }

    if (strchr(mode, 'w') != NULL) {
      RETURN_ON_ARROW_ERROR_AND_ASSIGN(ofp_, fs_->OpenOutputStream(location_));
    } else {
      RETURN_ON_ARROW_ERROR_AND_ASSIGN(ofp_, fs_->OpenAppendStream(location_));
    }
    return;
  } else {
    RETURN_ON_ARROW_ERROR_AND_ASSIGN(ifp_, fs_->OpenInputFile(location_));

    // check the partial read flag
    if (enable_partial_read_) {
      setPartialReadImpl();
      preReadPartialTable();
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

bool ArrowIOAdaptor::preReadPartialTable() {
  std::cout << "Reading CSV with Arrow." << std::endl;
  int index = index_;
  int64_t offset = partial_read_offset_[index];
  int64_t nbytes =
      partial_read_offset_[index + 1] - partial_read_offset_[index];
  std::cout << "offset: " << offset << " nbytes: " << nbytes << std::endl;

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
  // parse_options.delimiter = delimiter_;

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
  std::cout << "get partial table" << std::endl;
  preread_table_ = result.ValueOrDie();

  return true;
}

bool ArrowIOAdaptor::setPartialReadImpl() {
  seek(0, kFileLocationEnd);
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
      seek(partial_read_offset_[i], kFileLocationBegin);
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
  seek(file_stream_pos, kFileLocationBegin);
  return true;
}

bool ArrowIOAdaptor::ReadLine(std::string& line) {
  std::cout << "Arrow Read Line " << std::endl;
  if (preread_table_ == nullptr) {
    return false;
  }
  std::shared_ptr<arrow::Table> table = preread_table_;

  auto batch_reader = arrow::TableBatchReader(*table);
  auto s = batch_reader.ReadAll(batches);

  std::shared_ptr<arrow::RecordBatch> batch;
  batch = batches[batch_idx_];
  for (int j = 0; j < batch->num_rows(); j++) {
      std::stringstream ss;
      for (int i = 0; i < table->num_columns(); i++) {
        auto column = table->column(i);
        auto array = column->chunk(0);

        ss << array->GetScalar(j).ValueOrDie()->ToString()
           << (i == (table->num_columns() - 1) ? "\n" : " ");
      }
      line = ss.str();
    }


  while (batch_reader.ReadNext(&batch).ok() && batch != nullptr) {
    for (int j = 0; j < batch->num_rows(); j++) {
      std::stringstream ss;
      for (int i = 0; i < table->num_columns(); i++) {
        auto column = table->column(i);
        auto array = column->chunk(0);

        ss << array->GetScalar(j).ValueOrDie()->ToString()
           << (i == (table->num_columns() - 1) ? "\n" : " ");
      }
      line = ss.str();
    }
  }
  return true;
}

bool ArrowIOAdaptor::ReadArchive(OutArchive& archive) {
  if (!using_std_getline_ && file_) {
    size_t length;
    bool status = fread(&length, sizeof(size_t), 1, file_);
    if (!status) {
      return false;
    }
    archive.Allocate(length);
    status = fread(archive.GetBuffer(), 1, length, file_);
    return status;
  } else {
    VLOG(1) << "invalid operation.";
    return false;
  }
}

bool ArrowIOAdaptor::WriteArchive(InArchive& archive) {
  if (!using_std_getline_ && file_) {
    size_t length = archive.GetSize();
    bool status = fwrite(&length, sizeof(size_t), 1, file_);
    if (!status) {
      return false;
    }
    status = fwrite(archive.GetBuffer(), 1, length, file_);
    if (!status) {
      return false;
    }
    fflush(file_);
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

bool ArrowIOAdaptor::Write(void* buffer, size_t size) { return false; }

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
  auto s = fs_->CreateDir(path, true);
}

bool ArrowIOAdaptor::IsExist() {
  std::string path = location_;
  auto mfinfo = fs_->GetFileInfo(path);
  return mfinfo.ok() &&
         mfinfo.ValueUnsafe().type() != arrow::fs::FileType::NotFound;
}

}  // namespace grape
