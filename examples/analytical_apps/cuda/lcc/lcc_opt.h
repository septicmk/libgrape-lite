/** Copyright 2023 Alibaba Group Holding Limited.

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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_OPT_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_OPT_H_
#ifdef __CUDACC__
#include <iomanip>
#include <iostream>

#include "cuda/app_config.h"
#include "grape/grape.h"

namespace grape {
namespace cuda {
template <typename FRAG_T>
class LCCOPTContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using msg_t = vid_t;

  explicit LCCOPTContext(const FRAG_T& frag)
      : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~LCCOPTContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "Pagerank kernel time: " << traversal_kernel_time * 1000;
    VLOG(1) << "Send msg time: " << send_msg_time * 1000;
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config,
            msg_t** sorted_col, size_t** offset) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();

    this->lb = app_config.lb;
    this->stage = 0;

    global_degree.Init(vertices, 0);
    tricnt.Init(vertices, 0);
    hi_q.Init(iv.size());
    mid_q.Init(iv.size());
    lo_q.Init(iv.size());
    tricnt.H2D();

    row_offset.resize(vertices.size() + 1, 0);
    size_t n_vertices = vertices.size();

    using nbr_t = typename FRAG_T::nbr_t;

    messages.InitBuffer(
        ov.size() * (sizeof(thrust::pair<vid_t, size_t>)),
        1 * (sizeof(thrust::pair<vid_t, msg_t>)));  // rely on syncLengths()

    size_t n_edges = (*offset)[n_vertices];
    col_indices.resize(n_edges, 0);
    auto* d_col_indices = thrust::raw_pointer_cast(col_indices.data());
    CHECK_CUDA(cudaMemcpy(d_col_indices, *sorted_col, sizeof(msg_t) * n_edges,
                          cudaMemcpyHostToDevice));

    auto* d_row_offset = thrust::raw_pointer_cast(row_offset.data());

    CHECK_CUDA(cudaMemcpy(d_row_offset, *offset,
                          sizeof(size_t) * (n_vertices + 1),
                          cudaMemcpyHostToDevice));
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    tricnt.D2H();
    global_degree.D2H();

    for (auto v : iv) {
      double score = 0;
      if (global_degree[v] >= 2) {
        score = 2.0 * (tricnt[v]) /
                (static_cast<int64_t>(global_degree[v]) *
                 (static_cast<int64_t>(global_degree[v]) - 1));
      }
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << score << std::endl;
    }
  }

  LoadBalancing lb{};
  VertexArray<msg_t, vid_t> global_degree;
  VertexArray<size_t, vid_t> tricnt;
  thrust::device_vector<size_t> row_offset;
  thrust::device_vector<msg_t> col_indices;
  Queue<vertex_t, vid_t> hi_q, mid_q, lo_q;
  int stage{};
#ifdef PROFILING
  double get_msg_time{};
  double traversal_kernel_time{};
  double send_msg_time{};
#endif
};

template <typename FRAG_T>
class LCCOPT : public GPUAppBase<FRAG_T, LCCOPTContext<FRAG_T>>,
               public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(LCCOPT<FRAG_T>, LCCOPTContext<FRAG_T>, FRAG_T)
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;
  using msg_t = vid_t;

  static constexpr grape::MessageStrategy message_strategy =
      grape::MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;
  static constexpr bool need_build_device_vm = true;  // for debug

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    auto d_frag = frag.DeviceObject();
    auto d_global_degree = ctx.global_degree.DeviceObject();
    auto d_mm = messages.DeviceObject();
    WorkSourceRange<vertex_t> ws_in(*inner_vertices.begin(),
                                    inner_vertices.size());

    ForEach(messages.stream(), ws_in, [=] __device__(vertex_t v) mutable {
      msg_t degree = d_frag.GetLocalOutDegree(v);
      d_global_degree[v] = degree;
    });
    messages.ForceContinue();
  }

  void TriangleCounting(const fragment_t& frag, context_t& ctx,
                        message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    Stream stream_hi, stream_mi, stream_lo;
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto d_tricnt = ctx.tricnt.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& lo_q = ctx.lo_q;
    auto& hi_q = ctx.hi_q;
    auto& mid_q = ctx.mid_q;
    auto d_lo_q = ctx.lo_q.DeviceObject();
    auto d_hi_q = ctx.hi_q.DeviceObject();
    auto d_mid_q = ctx.mid_q.DeviceObject();

    {
      WorkSourceRange<vertex_t> ws_in(*iv.begin(), iv.size());
      auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      ForEach(stream, ws_in, [=] __device__(vertex_t u) mutable {
        size_t idx = u.GetValue();
        d_mid_q.Append(vertex_t(idx));
        // size_t degree = d_row_offset[idx + 1] - d_row_offset[idx];
        // if (degree > 5000) {
        //  d_hi_q.Append(vertex_t(idx));
        //} else if (degree < 0) {
        //  d_lo_q.Append(vertex_t(idx));
        //} else {
        //  d_mid_q.Append(vertex_t(idx));
        //}
      });
    }
    stream.Sync();

    double lo_time_start = grape::GetCurrentTime();
    {
      WorkSourceArray<vertex_t> ws_in(lo_q.data(), lo_q.size(stream));
      auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_filling_offset = d_row_offset + 1;
      auto* d_col_indices = thrust::raw_pointer_cast(ctx.col_indices.data());
      thrust::device_vector<msg_t> bins;
      size_t nblocks = 256;
      size_t bucket_stride = 256;
      size_t bucket_num = 32;
      size_t bucket_size = 31;
      size_t cached_size = 31;
      bins.resize(bucket_stride * (bucket_size - cached_size) * nblocks, 0);
      auto* global_data = thrust::raw_pointer_cast(bins.data());
      ForEachWithIndexWarpShared(
          stream_lo, ws_in,
          [=] __device__(uint32_t * shm, size_t lane, size_t cid, size_t csize,
                         size_t cnum, size_t idx, vertex_t u) mutable {
            dev::ShmHashTable<uint32_t> hash_table;
            hash_table.init(shm, global_data, threadIdx.x / csize * bucket_num,
                            bucket_size, cached_size, bucket_num,
                            bucket_stride);

            hash_table.clear(lane, csize);
            __syncwarp();

            idx = u.GetValue();
            size_t triangle_count = 0;
            for (auto eid = d_row_offset[idx] + lane;
                 eid < d_filling_offset[idx]; eid += csize) {
              if (!hash_table.insert(d_col_indices[eid])) {
                assert(false);
              }
            }
            __syncwarp();

            for (auto eid = d_row_offset[idx]; eid < d_filling_offset[idx];
                 eid++) {
              size_t tmp = 0;
              vertex_t v(d_col_indices[eid]);
              auto edge_begin_v = d_row_offset[v.GetValue()],
                   edge_end_v = d_filling_offset[v.GetValue()];
              for (auto i = lane + edge_begin_v; i < edge_end_v; i += csize) {
                auto w = vertex_t(d_col_indices[i]);
                if (hash_table.lookup(w.GetValue())) {
                  tmp++;
                  dev::atomicAdd64(&d_tricnt[w], 1);
                }
              }
              size_t v_cnt = dev::warp_reduce(tmp);
              if (lane == 0) {
                dev::atomicAdd64(&d_tricnt[v], v_cnt);
                triangle_count += v_cnt;
              }
            }
            __syncwarp();
            if (lane == 0) {
              dev::atomicAdd64(&d_tricnt[u], triangle_count);
            }
          });
    }
    // stream.Sync();
    double lo_time_end = grape::GetCurrentTime();

    double mi_time_start = grape::GetCurrentTime();
    {
      WorkSourceArray<vertex_t> ws_in(mid_q.data(), mid_q.size(stream));
      auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_filling_offset = d_row_offset + 1;
      auto* d_col_indices = thrust::raw_pointer_cast(ctx.col_indices.data());
      ForEachWithIndexWarpDynamic(
          stream_mi, ws_in,
          [=] __device__(size_t lane, size_t idx, vertex_t u) mutable {
            idx = u.GetValue();
            size_t triangle_count = 0;
            for (auto eid = d_row_offset[idx]; eid < d_filling_offset[idx];
                 eid++) {
              vertex_t v(d_col_indices[eid]);
              auto edge_begin_u = d_row_offset[u.GetValue()],
                   edge_end_u = d_filling_offset[u.GetValue()];
              auto edge_begin_v = d_row_offset[v.GetValue()],
                   edge_end_v = d_filling_offset[v.GetValue()];
              size_t degree_u = edge_end_u - edge_begin_u;
              size_t degree_v = edge_end_v - edge_begin_v;
              size_t tmp = dev::intersect_num(
                  &d_col_indices[edge_begin_u], degree_u,
                  &d_col_indices[edge_begin_v], degree_v,
                  [=] __device__(msg_t key) mutable {
                    dev::atomicAdd64(&d_tricnt[vertex_t(key)], 1);
                  });
              if (lane == 0) {
                dev::atomicAdd64(&d_tricnt[v], tmp);
                triangle_count += tmp;
              }
            }

            /*if (breakpoint < d_filling_offset[idx]) {
              int thread_lane = threadIdx.x & 31;
              int warp_lane = threadIdx.x / 32;
              __shared__ msg_t cache[256];
              __shared__ msg_t s_v[256];
              __shared__ size_t s_start_pos[256];
              __shared__ uint32_t s_row_offset[256];
              typedef cub::WarpScan<uint32_t, 32> WarpScan;
              __shared__ typename WarpScan::TempStorage temp_storage[8];
              int shm_per_warp = 32;
              int shm_per_thd = 1;
              int offset = warp_lane * shm_per_warp;
              msg_t* my_cache = cache + offset;
              auto edge_begin_u = d_row_offset[idx],
                   edge_end_u = d_filling_offset[idx];
              size_t degree_u = edge_end_u - edge_begin_u;
              size_t left_size = edge_end_u - breakpoint;
              size_t aligned_left_size =
                  (left_size & 31) ? (((left_size >> 5) + 1) << 5) : left_size;
              msg_t* search = &d_col_indices[edge_begin_u];
              size_t search_size = degree_u;
              __syncwarp();
              // sample u'neighbor into cache;
              for (size_t i = 0; i < shm_per_thd; ++i) {
                my_cache[i * 32 + thread_lane] =
                    search[(thread_lane + i * 32) * search_size / shm_per_warp];
              }
              __syncwarp();
              // make all lane can enter this loop;
              assert(aligned_left_size >= left_size);
              size_t chunk_size = 1;
              for (auto eid = breakpoint + thread_lane;
                   eid < breakpoint + aligned_left_size; eid += chunk_size) {
                uint32_t active_mask =
                    __ballot_sync(0xffffffff, eid < breakpoint + left_size &&
                                                  thread_lane < chunk_size);
                uint32_t active_num = __popc(active_mask);
                size_t total_edges = 0;
                uint32_t degree_v = 0;
                if (eid < breakpoint + left_size && thread_lane < chunk_size) {
                  vertex_t v(d_col_indices[eid]);
                  auto edge_begin_v = d_row_offset[v.GetValue()],
                       edge_end_v = d_filling_offset[v.GetValue()];
                  degree_v = edge_end_v - edge_begin_v;
                  s_v[offset + thread_lane] = v.GetValue();
                  s_start_pos[offset + thread_lane] = edge_begin_v;
                  total_edges = degree_v;
                }
                __syncwarp();
                total_edges = dev::warp_reduce(total_edges);
                WarpScan(temp_storage[warp_lane])
                    .ExclusiveSum(degree_v, s_row_offset[threadIdx.x]);
                __syncwarp();
                // process the agg_edges of this 32 vertices with full
                // warp
                size_t aligned_total_edges =
                    (total_edges & 31) ? (((total_edges >> 5) + 1) << 5)
                                       : total_edges;
                for (size_t i = thread_lane; i < aligned_total_edges; i += 32) {
                  int64_t search_result = -1;
                  vertex_t v;
                  if (i < total_edges) {
                    int loc = thrust::upper_bound(
                                  thrust::seq, s_row_offset + offset,
                                  s_row_offset + offset + active_num, i) -
                              s_row_offset - offset - 1;
                    assert(loc < active_num);
                    v = vertex_t(s_v[offset + loc]);
                    auto edge_begin_v = s_start_pos[offset + loc];
                    msg_t* lookup = &d_col_indices[edge_begin_v];
                    size_t lookup_ei = i - s_row_offset[offset + loc];
                    auto key = lookup[lookup_ei];
                    search_result = dev::binary_search_2phase(
                        search, my_cache, key, search_size, shm_per_warp);
                    if (search_result >= 0) {
                      dev::atomicAdd64(&d_tricnt[vertex_t(key)], 1);
                    }
                  }
                  uint32_t amask = __ballot_sync(0xffffffff, i < total_edges);
                  uint32_t smask = __ballot_sync(amask, search_result >= 0);
                  uint32_t vmask =
                      __match_any_sync(smask & amask, v.GetValue());
                  uint32_t v_count = __popc(vmask & smask);
                  uint32_t leader = __ffs(vmask & smask) - 1;
                  if (v_count > 0 && thread_lane == leader) {
                    dev::atomicAdd64(&d_tricnt[v], v_count);
                    triangle_count += v_count;
                  }
                }
              }
              triangle_count = dev::warp_reduce(triangle_count);
            }
            // if (thread_lane == 0) {
            //  dev::atomicAdd64(&d_tricnt[u], triangle_count);
            //}
            */
            if (lane == 0) {
              dev::atomicAdd64(&d_tricnt[u], triangle_count);
            }
          });
    }
    // stream.Sync();
    double mi_time_end = grape::GetCurrentTime();

    double hi_time_start = grape::GetCurrentTime();
    {
      WorkSourceArray<vertex_t> ws_in(hi_q.data(), hi_q.size(stream));
      auto* d_row_offset = thrust::raw_pointer_cast(ctx.row_offset.data());
      auto* d_filling_offset = d_row_offset + 1;
      auto* d_col_indices = thrust::raw_pointer_cast(ctx.col_indices.data());
      ForEachWithIndexBlockDynamic(
          stream_hi, ws_in,
          [=] __device__(size_t lane, size_t idx, vertex_t u) mutable {
            idx = u.GetValue();
            size_t triangle_count = 0;
            for (auto eid = d_row_offset[idx]; eid < d_filling_offset[idx];
                 eid++) {
              vertex_t v(d_col_indices[eid]);
              auto edge_begin_u = d_row_offset[u.GetValue()],
                   edge_end_u = d_filling_offset[u.GetValue()];
              auto edge_begin_v = d_row_offset[v.GetValue()],
                   edge_end_v = d_filling_offset[v.GetValue()];
              size_t degree_u = edge_end_u - edge_begin_u;
              size_t degree_v = edge_end_v - edge_begin_v;
              size_t tmp = dev::intersect_num_blk(
                  &d_col_indices[edge_begin_u], degree_u,
                  &d_col_indices[edge_begin_v], degree_v,
                  [=] __device__(msg_t key) mutable {
                    dev::atomicAdd64(&d_tricnt[vertex_t(key)], 1);
                  });
              if (lane == 0) {
                dev::atomicAdd64(&d_tricnt[v], tmp);
                triangle_count += tmp;
              }
              __syncthreads();
            }
            __syncthreads();
            if (lane == 0) {
              dev::atomicAdd64(&d_tricnt[u], triangle_count);
            }
          });
    }
    // stream.Sync();
    double hi_time_end = grape::GetCurrentTime();
    stream_lo.Sync();
    stream_mi.Sync();
    stream_hi.Sync();

    // std::cout << frag.fid() << " lo:" << (lo_time_end - lo_time_start) * 1000
    //          << " mi:" << (mi_time_end - mi_time_start) * 1000
    //          << " hi:" << (hi_time_end - hi_time_start) * 1000 << std::endl;

    {  // send d_tricnt
      WorkSourceRange<vertex_t> ws_in(*ov.begin(), ov.size());
      ForEach(stream, ws_in, [=] __device__(vertex_t v) mutable {
        if (d_tricnt[v] != 0) {
          d_mm.template SyncStateOnOuterVertex(d_frag, v, d_tricnt[v]);
        }
      });
    }

    messages.ForceContinue();
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto& global_degree = ctx.global_degree;
    auto d_global_degree = global_degree.DeviceObject();
    auto d_tricnt = ctx.tricnt.DeviceObject();
    auto d_mm = messages.DeviceObject();

    if (ctx.stage == 0) {
      TriangleCounting(frag, ctx, messages);
    }

    if (ctx.stage == 1) {
      messages.template ParallelProcess<dev_fragment_t, size_t>(
          d_frag, [=] __device__(vertex_t v, size_t tri_cnt) mutable {
            dev::atomicAdd64(&d_tricnt[v], tri_cnt);
          });
    }

    ctx.stage = ctx.stage + 1;
  }
};

}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_LCC_LCC_OPT_H_
