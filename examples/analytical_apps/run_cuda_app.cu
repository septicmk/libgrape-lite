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

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include "grape/cuda/utils/cuda_utils.h"
#include "run_cuda_app.h"

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  grape::gflags::SetUsageMessage(
      "Usage: mpiexec [mpi_opts] ./run_app [grape_opts]");
  if (argc == 1) {
    grape::gflags::ShowUsageWithFlagsRestrict(argv[0], "analytical_apps");
    exit(1);
  }
  grape::gflags::ParseCommandLineFlags(&argc, &argv, true);
  grape::gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging("analytical_apps");
  google::InstallFailureSignalHandler();

  grape::cuda::Init();

  std::string name = FLAGS_application;

  if (FLAGS_segmented_partition && FLAGS_directed) {
    grape::cuda::Run<int64_t, uint32_t, grape::EmptyType, true, true>();
  } else if (FLAGS_segmented_partition && !FLAGS_directed) {
    grape::cuda::Run<int64_t, uint32_t, grape::EmptyType, true, false>();
  } else if (!FLAGS_segmented_partition && FLAGS_directed) {
    grape::cuda::Run<int64_t, uint32_t, grape::EmptyType, false, true>();
  } else {
    grape::cuda::Run<int64_t, uint32_t, grape::EmptyType, false, false>();
  }
  grape::cuda::Finalize();

  google::ShutdownGoogleLogging();
}
