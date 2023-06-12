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

#ifndef EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_
#define EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <grape/grape.h>
#include <grape/util.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuda/bfs/bfs.h"
#include "cuda/cdlp/cdlp.h"
#include "cuda/lcc/lcc.h"
#include "cuda/lcc/lcc_directed.h"
#include "cuda/lcc/lcc_opt.h"
#include "cuda/lcc/lcc_preprocess.h"
#include "cuda/pagerank/pagerank.h"
#include "cuda/sssp/sssp.h"
#include "cuda/wcc/wcc.h"
#include "cuda/wcc/wcc_opt.h"
#include "flags.h"
#include "grape/cuda/fragment/host_fragment.h"
#include "grape/cuda/worker/gpu_batch_shuffle_worker.h"
#include "grape/cuda/worker/gpu_worker.h"
#include "grape/fragment/loader.h"
#include "grape/worker/comm_spec.h"
#include "timer.h"

#ifndef __AFFINITY__
#define __AFFINITY__ false
#endif

namespace grape {

namespace cuda {

void Init() {
  if (FLAGS_efile.empty()) {
    LOG(FATAL) << "Please assign input edge files.";
  }

  if (access(FLAGS_out_prefix.c_str(), 0) != 0) {
    mkdir(FLAGS_out_prefix.c_str(), 0777);
  }

  grape::InitMPIComm();
}

void Finalize() {
  grape::FinalizeMPIComm();
  VLOG(1) << "Workers finalized.";
}

template <typename FRAG_T, typename APP_T, typename... Args>
void DoPreprocess(std::shared_ptr<FRAG_T> fragment, std::shared_ptr<APP_T> app,
                  const CommSpec& comm_spec, int dev_id,
                  const std::string& out_prefix, Args... args) {
  auto spec = MultiProcessSpec(comm_spec, __AFFINITY__);
  if (FLAGS_app_concurrency != -1) {
    spec.thread_num = FLAGS_app_concurrency;
    if (__AFFINITY__) {
      if (spec.cpu_list.size() >= spec.thread_num) {
        spec.cpu_list.resize(spec.thread_num);
      } else {
        uint32_t num_to_append = spec.thread_num - spec.cpu_list.size();
        for (uint32_t i = 0; i < num_to_append; ++i) {
          spec.cpu_list.push_back(spec.cpu_list[i]);
        }
      }
    }
  }
  auto worker = APP_T::CreateWorker(app, fragment);
  worker->Init(comm_spec, spec);
  MPI_Barrier(comm_spec.comm());
  worker->Query(std::forward<Args>(args)...);
  MPI_Barrier(comm_spec.comm());
  std::ofstream ostream;
  worker->Output(ostream);
  worker->Finalize();
}

template <typename FRAG_T, typename APP_T, typename... Args>
double DoQuery(std::shared_ptr<FRAG_T> fragment, std::shared_ptr<APP_T> app,
               const CommSpec& comm_spec, int dev_id,
               const std::string& out_prefix, Args... args) {
  // timer_next("load application");
  auto worker = APP_T::CreateWorker(app, fragment);
  worker->Init(comm_spec, std::forward<Args>(args)...);
  MPI_Barrier(comm_spec.comm());

  // timer_next("run algorithm");
  double time_start = timer();
  CHECK_CUDA(cudaSetDevice(dev_id));
  worker->Query();
  MPI_Barrier(comm_spec.comm());
  double time_end = timer();

  // timer_next("print output");

  if (!out_prefix.empty()) {
    std::ofstream ostream;
    std::string output_path =
        grape::GetResultFilename(out_prefix, fragment->fid());
    ostream.open(output_path);
    worker->Output(ostream);
    ostream.close();
    VLOG(1) << "Worker-" << comm_spec.worker_id()
            << " finished: " << output_path;
  }
  worker->Finalize();
  // timer_end();

  return time_end - time_start;
}

template <typename OID_T, typename VID_T, typename VDATA_T, bool isSeg,
          bool isDirected>
struct FragmentTraits {};

template <typename OID_T, typename VID_T, typename VDATA_T>
struct FragmentTraits<OID_T, VID_T, VDATA_T, true, true> {
  using VERTEX_MAP_T =
      GlobalVertexMap<OID_T, VID_T, SegmentedPartitioner<OID_T>>;
  using WFRAG_T =
      grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, float,
                                grape::LoadStrategy::kBothOutIn, VERTEX_MAP_T>;
  using SFRAG_T =
      grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, grape::EmptyType,
                                grape::LoadStrategy::kBothOutIn, VERTEX_MAP_T>;
};

template <typename OID_T, typename VID_T, typename VDATA_T>
struct FragmentTraits<OID_T, VID_T, VDATA_T, true, false> {
  using VERTEX_MAP_T =
      GlobalVertexMap<OID_T, VID_T, SegmentedPartitioner<OID_T>>;
  using WFRAG_T =
      grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, float,
                                grape::LoadStrategy::kOnlyOut, VERTEX_MAP_T>;
  using SFRAG_T =
      grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, grape::EmptyType,
                                grape::LoadStrategy::kOnlyOut, VERTEX_MAP_T>;
};

template <typename OID_T, typename VID_T, typename VDATA_T>
struct FragmentTraits<OID_T, VID_T, VDATA_T, false, true> {
  using VERTEX_MAP_T = GlobalVertexMap<OID_T, VID_T, HashPartitioner<OID_T>>;
  using WFRAG_T =
      grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, float,
                                grape::LoadStrategy::kBothOutIn, VERTEX_MAP_T>;
  using SFRAG_T =
      grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, grape::EmptyType,
                                grape::LoadStrategy::kBothOutIn, VERTEX_MAP_T>;
};

template <typename OID_T, typename VID_T, typename VDATA_T>
struct FragmentTraits<OID_T, VID_T, VDATA_T, false, false> {
  using VERTEX_MAP_T = GlobalVertexMap<OID_T, VID_T, HashPartitioner<OID_T>>;
  using WFRAG_T =
      grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, float,
                                grape::LoadStrategy::kOnlyOut, VERTEX_MAP_T>;
  using SFRAG_T =
      grape::cuda::HostFragment<OID_T, VID_T, VDATA_T, grape::EmptyType,
                                grape::LoadStrategy::kOnlyOut, VERTEX_MAP_T>;
};

template <template <class> class APP_T, typename FRAG_T, typename... Args>
double CreateAndQuery(const grape::CommSpec& comm_spec,
                      std::shared_ptr<FRAG_T> fragment,
                      const std::string& out_prefix, Args... args) {
  // using fragment_t = FRAG_T;
  // using oid_t = typename FRAG_T::oid_t;
  // timer_next("load graph");
  auto app = std::make_shared<APP_T<FRAG_T>>();
  int dev_id = comm_spec.local_id();
  fragment->AllocateDeviceCSR();
  double time = DoQuery<FRAG_T, APP_T<FRAG_T>, Args...>(
      fragment, app, comm_spec, dev_id, out_prefix, args...);
  fragment->ReleaseDeviceCSR();
  return time;
}

template <template <class> class APP_T, template <class> class PRE_T,
          typename FRAG_T, typename... Args>
double CreateAndQueryWithPreprocess(const grape::CommSpec& comm_spec,
                                    std::shared_ptr<FRAG_T> fragment,
                                    const std::string& out_prefix,
                                    Args... args) {
  // using fragment_t = FRAG_T;
  // using oid_t = typename FRAG_T::oid_t;
  // timer_next("load graph");
  auto app = std::make_shared<APP_T<FRAG_T>>();
  auto pre = std::make_shared<PRE_T<FRAG_T>>();
  DoPreprocess<FRAG_T, PRE_T<FRAG_T>, Args...>(fragment, pre, comm_spec, dev_id,
                                               out_prefix, args...);
  fragment->AllocateDeviceCSR();
  double time = DoQuery<FRAG_T, APP_T<FRAG_T>, Args...>(
      fragment, app, comm_spec, dev_id, out_prefix, args...);
  fragment->ReleaseDeviceCSR();
  return time;
}

template <typename OID_T, typename VID_T, typename VDATA_T, bool isSeg,
          bool isDirected>
void Run() {
  constexpr char SOCKET_FILE[] = "/tmp/test.sock";
  CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  bool is_coordinator = comm_spec.worker_id() == grape::kCoordinatorRank;
  // timer_start(is_coordinator);

  std::string efile = FLAGS_efile;
  std::string vfile = FLAGS_vfile;
  std::string out_prefix = FLAGS_out_prefix;

  std::string application;
  AppConfig app_config;

  app_config.lb = ParseLoadBalancing(FLAGS_lb);
  app_config.wl_alloc_factor_in = 0.4;
  app_config.wl_alloc_factor_out_local = 0.2;
  app_config.wl_alloc_factor_out_remote = 0.2;

  LoadGraphSpec graph_spec = DefaultLoadGraphSpec();

  graph_spec.set_directed(FLAGS_directed);
  graph_spec.set_rebalance(FLAGS_rebalance, FLAGS_rebalance_vertex_factor);
  if (FLAGS_deserialize) {
    std::string deserialization_prefix(FLAGS_serialization_prefix);
    if (FLAGS_directed) {
      deserialization_prefix = deserialization_prefix + "-sssp";
    }
    graph_spec.set_deserialize(true, deserialization_prefix);
  } else if (FLAGS_serialize) {
    graph_spec.set_serialize(true, FLAGS_serialization_prefix);
  }
  if (FLAGS_segmented_partition) {
    graph_spec.set_rebalance(false, 0);
  }
  int dev_id = comm_spec.local_id();
  CHECK_CUDA(cudaSetDevice(dev_id));
  using WFRAG_T = typename FragmentTraits<OID_T, VID_T, VDATA_T, isSeg,
                                          isDirected>::WFRAG_T;
  using SFRAG_T = typename FragmentTraits<OID_T, VID_T, VDATA_T, isSeg,
                                          isDirected>::SFRAG_T;

  std::shared_ptr<WFRAG_T> fragmentW =
      LoadGraph<WFRAG_T, ArrowIOAdaptor>(efile, vfile, comm_spec, graph_spec);
  std::shared_ptr<SFRAG_T> fragmentS =
      LoadGraph<SFRAG_T, ArrowIOAdaptor>(efile, vfile, comm_spec, graph_spec);

  // start server
  int server_fd, client_fd;
  char buffer[2048];

  MPI_Barrier(comm_spec.comm());
  if (comm_spec.local_id() == 0) {
    server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd == -1) {
      std::cerr << "Failed to create socket." << std::endl;
      return -1;
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    strncpy(address.sun_path, SOCKET_FILE, sizeof(address.sun_path) - 1);
    unlink(SOCKET_FILE);
    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) == -1) {
      std::cerr << "Failed to bind socket to file." << std::endl;
      close(server_fd);
      return -1;
    }
    if (listen(server_fd, 5) == -1) {
      std::cerr << "Failed to start listening for connections." << std::endl;
      close(server_fd);
      return -1;
    }
  }
  MPI_Barrier(comm_spec.comm());

  if (comm_spec.local_id() == 0) {
    std::cout << "Graph Loaded." << std::endl;
  }

  while (true) {
    MPI_Barrier(comm_spec.comm());
    ssize_t num_bytes;
    if (comm_spec.local_id() == 0) {
      num_bytes = recv(client_fd, buffer, sizeof(buffer), 0);
      if (num_bytes == -1 || num_bytes == 0) {
        std::cerr << "Connection closed by client." << std::endl;
        break;
      }
    }
    MPI_Barrier(comm_spec.comm());
    MPI_Bcast(buffer, static_cast<size_t>(num_bytes), MPI_CHAR, 0,
              comm_spec.comm());

    std::stringstream ss{buffer, static_cast<size_t>(num_bytes)};
    std::string application, out_prefix;
    ss >> out_prefix >> bfs;

    double ptime = 0.0;
    if (application == "bfs") {
      OID_T bfs_source;
      ss >> bfs_source;
      ptime = CreateAndQuery<BFS>(comm_spec, fragmentS, out_prefix, app_config,
                                  bfs_source);
    } else if (application == "sssp") {
      OID_T sssp_source;
      ss >> sssp_source;
      ptime = CreateAndQuery<SSSP>(comm_spec, fragmentW, out_prefix, app_config,
                                   sssp_source, 0);
      fragmentW = nullptr;
    } else if (application == "wcc") {
      ptime = CreateAndQuery<WCC>(comm_spec, fragmentS, out_prefix, app_config);
    } else if (application == "pagerank") {
      double pr_d, pr_mr;
      ss >> pr_d >> pr_mr;
      ptime = CreateAndQuery<Pagerank>(comm_spec, fragmentS, out_prefix,
                                       app_config, pr_d, pr_mr);
    } else if (application == "lcc") {
      if (FLAGS_directed) {
        ptime =
            CreateAndQuery<LCCD>(comm_spec, fragmentS, out_prefix, app_config);
      } else {
        VID_T** col = (VID_T**) malloc(sizeof(VID_T*));
        size_t** row_offset = (size_t**) malloc(sizeof(size_t*));
        ptime = CreateAndQueryWithPreprocess<LCCOPT, LCCP>(
            comm_spec, fragmentS, out_prefix, app_config, col, row_offset);
        free(*col);
        free(*row_offset);
        free(col);
        free(row_offset);
      }
    } else if (application == "cdlp") {
      double cdlp_mr;
      ss >> cdlp_mr;
      ptime = CreateAndQuery<CDLP>(comm_spec, fragmentS, out_prefix, app_config,
                                   cdlp_mr);
    } else if (application == "exit") {
      break;
    } else {
      LOG(FATAL) << "Invalid app name: " << application;
    }

    MPI_Barrier(comm_spec.comm());
    if (comm_spec.local_id() == 0) {
      std::stringstream sout;
      sout << " - run algorithm: " << ptime << " sec";
      std::string msg = sout.str();
      const ssize_t num_bytes_2 = send(client_fd, msg.c_str(), msg.length(), 0);
    }
    MPI_Barrier(comm_spec.comm());
  }

  if (comm_spec.local_id() == 0) {
    close(client_fd);
  }
}
}  // namespace cuda
}  // namespace grape
#endif  // EXAMPLES_ANALYTICAL_APPS_RUN_CUDA_APP_H_
