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

#ifndef GRAPE_ARROW_IO_ADAPTOR_H_
#define GRAPE_ARROW_IO_ADAPTOR_H_

#include <stdio.h>

#include <fstream>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "arrow/csv/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/io/api.h"
#include "arrow/util/uri.h"
#include "grape/io/io_adaptor_base.h"

#ifndef RETURN_ON_ARROW_ERROR_AND_ASSIGN
#define RETURN_ON_ARROW_ERROR_AND_ASSIGN(lhs, expr) \
  do {                                              \
    auto result = (expr);                           \
    if (!result.status().ok()) {                    \
      return false;                                 \
    }                                               \
    lhs = std::move(result).ValueOrDie();           \
  } while (0)
#endif  // RETURN_ON_ARROW_ERROR_AND_ASSIGN

#ifndef DISCARD_ARROW_ERROR_AND_ASSIGN
#define DISCARD_ARROW_ERROR_AND_ASSIGN(lhs, expr) \
  do {                                            \
    auto result = (expr);                         \
    lhs = std::move(result).ValueOrDie();         \
  } while (0)
#endif  // DISCARD_ARROW_ERROR_AND_ASSIGN

#ifndef RETURN_ON_ARROW_ERROR
#define RETURN_ON_ARROW_ERROR(expr) \
  do {                              \
    auto status = (expr);           \
    if (!status.ok()) {             \
      return false;                 \
    }                               \
  } while (0)
#endif  // RETURN_ON_ARROW_ERROR_AND

#ifndef RETURN_ON_ERROR
#define RETURN_ON_ERROR(expr) \
  do {                        \
    auto status = (expr);     \
    if (!status) {            \
      return false;           \
    }                         \
  } while (0)
#endif  // RETURN_ON_ARROW

namespace grape {
class InArchive;
class OutArchive;

enum FileLocation {
  kFileLocationBegin = 0,
  kFileLocationCurrent = 1,
  kFileLocationEnd = 2,
};
#define LINESIZE 65536

/**
 * @brief A default adaptor to read/write files from local locations.
 *
 */
class ArrowIOAdaptor : public IOAdaptorBase {
 public:
  explicit ArrowIOAdaptor(std::string location);
  
  explicit ArrowIOAdaptor(std::string location, std::string fspath);

  ~ArrowIOAdaptor() override;

  void Open() override;

  void Open(const char* mode) override;

  void Close() override;

  bool Configure(const std::string& key, const std::string& value) override;

  bool SetPartialRead(int index, int total_parts) override;

  bool ReadLine(std::string& line) override;

  bool ReadArchive(OutArchive& archive) override;

  bool WriteArchive(InArchive& archive) override;

  bool Read(void* buffer, size_t size) override;

  bool Write(void* buffer, size_t size) override;

  void MakeDirectory(const std::string& path) override;

  bool IsExist() override;

 private:
  int64_t tell();
  bool seek(const int64_t offset, const FileLocation seek_from);
  bool setPartialReadImpl();
  bool preReadPartialTable(bool partial);
  std::string realPath(std::string const& path);

  std::string location_;
  std::string fspath_;
  bool enable_partial_read_;
  int total_parts_;
  int index_;
  bool is_s3_;

  char buff[LINESIZE];
  std::vector<int64_t> partial_read_offset_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::io::RandomAccessFile> ifp_;  // for input
  std::shared_ptr<arrow::io::OutputStream> ofp_;      // for output

  // for arrow
  char delimiter_ = ' ';
  std::shared_ptr<arrow::Table> preread_table_;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
  int batch_idx_;
  int row_idx_;
};
}  // namespace grape

#endif  // GRAPE_IO_LOCAL_IO_ADAPTOR_H_
