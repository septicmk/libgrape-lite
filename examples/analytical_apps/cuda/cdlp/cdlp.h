/** Copyright 2022 Alibaba Group Holding Limited.

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

#ifndef EXAMPLES_ANALYTICAL_APPS_CUDA_CDLP_CDLP_H_
#define EXAMPLES_ANALYTICAL_APPS_CUDA_CDLP_CDLP_H_

#ifdef __CUDACC__
#include <algorithm>

#include "cuda/app_config.h"
#include "grape/grape.h"

namespace grape {
namespace cuda {
template <typename FRAG_T>
class CDLPContext : public grape::VoidContext<FRAG_T> {
 public:
  using vid_t = typename FRAG_T::vid_t;
  using oid_t = typename FRAG_T::oid_t;
  using vertex_t = typename FRAG_T::vertex_t;
  using label_t = uint32_t;

  explicit CDLPContext(const FRAG_T& frag) : grape::VoidContext<FRAG_T>(frag) {}

#ifdef PROFILING
  ~CDLPContext() {
    VLOG(1) << "Get msg time: " << get_msg_time * 1000;
    VLOG(1) << "CDLP kernel time: " << traversal_kernel_time * 1000;
  }
#endif

  void Init(GPUMessageManager& messages, AppConfig app_config, int max_round) {
    auto& frag = this->fragment();
    auto vertices = frag.Vertices();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    this->step = 0;
    this->max_round = max_round;
    this->lb = app_config.lb;

    labels.Init(vertices);
    new_label.Init(iv, thrust::make_pair(0, false));

    hi_q.Init(iv.size());
    lo_q.Init(iv.size());

    d_lo_row_offset.resize(iv.size() + 1);
    d_hi_row_offset.resize(iv.size() + 1);

    for (auto v : iv) {
      labels[v] = frag.GetInnerVertexId(v);
    }

    for (auto v : ov) {
      labels[v] = frag.GetOuterVertexId(v);
    }
    labels.H2D();

#ifdef PROFILING
    get_msg_time = 0;
    traversal_kernel_time = 0;
#endif

    // messages.InitBuffer(100 * 1024 * 1024, 100 * 1024 * 1024);
    messages.InitBuffer(  // N.B. pair padding
        (sizeof(thrust::pair<vid_t, label_t>)) * iv.size(),
        (sizeof(thrust::pair<vid_t, label_t>)) * 1);
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto iv = frag.InnerVertices();

    labels.D2H();

    for (auto v : iv) {
      os << frag.GetId(v) << " " << labels[v] << std::endl;
    }
  }

  int step;
  int max_round;
  LoadBalancing lb;
  VertexArray<label_t, vid_t> labels;
  VertexArray<thrust::pair<label_t, bool>, vid_t> new_label;

  Queue<vertex_t, vid_t> hi_q, lo_q;
  // low-degree: segment sort + scan
  thrust::device_vector<size_t> d_lo_row_offset;
  thrust::device_vector<label_t> d_col_indices;
  thrust::device_vector<label_t> d_sorted_col_indices;
  // high-degree: shm_ht + CMS + gm_ht
  thrust::device_vector<size_t> d_hi_row_offset;
  thrust::device_vector<label_t> d_hi_col_indices;
  thrust::device_vector<label_t> d_hi_label_hash;
  thrust::device_vector<uint32_t> d_hi_label_cnt;

#ifdef PROFILING
  double get_msg_time;
  double traversal_kernel_time;
#endif
};

template <typename FRAG_T>
class CDLP : public GPUAppBase<FRAG_T, CDLPContext<FRAG_T>>,
             public ParallelEngine {
 public:
  INSTALL_GPU_WORKER(CDLP<FRAG_T>, CDLPContext<FRAG_T>, FRAG_T)
  using label_t = typename context_t::label_t;
  using dev_fragment_t = typename fragment_t::device_t;
  using vid_t = typename fragment_t::vid_t;
  using edata_t = typename fragment_t::edata_t;
  using vertex_t = typename dev_fragment_t::vertex_t;
  using nbr_t = typename dev_fragment_t::nbr_t;
  using size_type = size_t;

  static constexpr grape::MessageStrategy message_strategy =
      MessageStrategyTrait<FRAG_T::load_strategy>::message_strategy;
  static constexpr grape::LoadStrategy load_strategy =
      grape::LoadStrategy::kOnlyOut;
  static constexpr bool need_build_device_vm = true;  // for debug

  void PropagateLabel_hi(const fragment_t& frag, context_t& ctx,
                         message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto d_labels = ctx.labels.DeviceObject();
    bool isDirected = frag.load_strategy == grape::LoadStrategy::kBothOutIn;
    auto* d_offsets = thrust::raw_pointer_cast(ctx.d_hi_row_offset.data());
    auto* p_d_col_indices =
        thrust::raw_pointer_cast(ctx.d_hi_col_indices.data());
    auto& hi_q = ctx.hi_q;
    auto d_new_label = ctx.new_label.DeviceObject();

    WorkSourceArray<vertex_t> ws_iv(hi_q.data(), hi_q.size(stream));
    double T_update = grape::GetCurrentTime();
    if (isDirected) {
      ForEachIncomingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid =
                d_offsets[u.GetValue()] + d_frag.GetIncomingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
      ForEachOutgoingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid = d_offsets[u.GetValue()] + d_frag.GetLocalInDegree(u) +
                         d_frag.GetOutgoingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
    } else {
      ForEachOutgoingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid =
                d_offsets[u.GetValue()] + d_frag.GetOutgoingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
    }
    stream.Sync();
    T_update = grape::GetCurrentTime() - T_update;

    auto* local_labels = thrust::raw_pointer_cast(ctx.d_hi_col_indices.data());
    auto* global_data = thrust::raw_pointer_cast(ctx.d_hi_label_hash.data());
    auto* label_cnt = thrust::raw_pointer_cast(ctx.d_hi_label_cnt.data());

    thrust::fill(ctx.d_hi_label_hash.begin(), ctx.d_hi_label_hash.end(),
                 0xffffffff);
    thrust::fill(ctx.d_hi_label_cnt.begin(), ctx.d_hi_label_cnt.end(), 0);

    size_t bucket_size = 1024;
    size_t cms_size = 256;
    size_t cms_k = 16;

    ForEachWithIndexBlockShared(
        stream, ws_iv,
        [=] __device__(uint32_t * shm, size_t lane, size_t cid, size_t csize,
                       size_t cnum, size_t idx, vertex_t u) mutable {
          for (int i = threadIdx.x; i < 8192; i += blockDim.x) {
            if (i < bucket_size) {
              shm[i] = 0xffffffff;
            } else {
              shm[i] = 0;
            }
          }
          __syncthreads();

          idx = u.GetValue();
          size_t begin = d_offsets[idx], end = d_offsets[idx + 1];
          if (end == begin) {
            return;
          }

          dev::MFLCounter<uint32_t> counter;
          counter.init(shm, global_data, label_cnt, bucket_size, cms_size,
                       cms_k);
          __shared__ label_t new_label;

          label_t max_label = 0;
          int ht_score = 0;
          int cms_score = 0;
          int gt_score = 0;

          // step 1: try CMS speculatively
          label_t local_label = 0xffffffff;
          int sh_ht_freq = -1;
          int sh_cms_freq = -1;
          __syncthreads();
          for (auto eid = begin + threadIdx.x; eid < end; eid += blockDim.x) {
            auto l = local_labels[eid];
            int current_sh_ht_freq = counter.insert_shm_ht(l);
            int current_sh_cms_freq = -1;

            if (current_sh_ht_freq < 0) {
              current_sh_cms_freq = counter.insert_shm_cms(l);
            }

            // update locally
            int current = MAX(current_sh_ht_freq, current_sh_cms_freq);
            int old = MAX(sh_ht_freq, sh_cms_freq);
            if (current > old || current == old && local_label > l) {
              local_label = l;
              sh_ht_freq = current_sh_ht_freq;
              sh_cms_freq = current_sh_cms_freq;
            }
          }
          __syncthreads();

          // step 2: check whether our CMS works
          ht_score = sh_ht_freq > 0 ? sh_ht_freq : 0;
          cms_score = sh_cms_freq > 0 ? sh_cms_freq : 0;
          max_label = local_label;
          __syncthreads();
          max_label = dev::blockAllReduceMax(max_label);
          ht_score = dev::blockAllReduceMax(ht_score);
          cms_score = dev::blockAllReduceMax(cms_score);
          __syncthreads();
          if (ht_score > cms_score) {  // shared_memory is enough
            if (threadIdx.x == 0) {
              new_label = max_label;
            }
            __syncthreads();
            if (sh_ht_freq == ht_score) {
              atomicMin(&new_label, local_label);
            }
            __syncthreads();
          } else {
            // step 3: the bad case, we have to do it again
            label_t local_label = 0xffffffff;
            int ht_freq = -1;
            for (auto eid = begin + threadIdx.x; eid < end; eid += blockDim.x) {
              auto l = local_labels[eid];
              int current_ht_freq = counter.query_shm_ht(l);

              if (current_ht_freq < 0) {
                assert(current_ht_freq != 0);
                current_ht_freq = counter.insert_global_ht(l, begin, end);
              }

              // update locally
              if (current_ht_freq > ht_freq ||
                  current_ht_freq == ht_freq && local_label > l) {
                local_label = l;
                ht_freq = current_ht_freq;
              }
            }
            __syncthreads();

            // step 4: now we have the true count.
            max_label = local_label;
            gt_score = ht_freq > 0 ? ht_freq : 0;
            max_label = dev::blockAllReduceMax(max_label);
            gt_score = dev::blockAllReduceMax(gt_score);
            __syncthreads();

            if (threadIdx.x == 0) {
              new_label = max_label;
            }
            __syncthreads();

            if (gt_score == ht_freq) {
              atomicMin(&new_label, local_label);
            }
            __syncthreads();
          }
          __syncthreads();

          // step 5: process the new label
          if (threadIdx.x == 0) {
            if (new_label != d_labels[u]) {
              d_new_label[u].first = new_label;
              d_new_label[u].second = true;
              if (isDirected) {
                d_mm.template SendMsgThroughEdges(d_frag, u, new_label);
              } else {
                d_mm.template SendMsgThroughOEdges(d_frag, u, new_label);
              }
            } else {
              d_new_label[u].second = false;
            }
          }
        });
  }

  void PropagateLabel_lo(const fragment_t& frag, context_t& ctx,
                         message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_mm = messages.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto* p_d_col_indices = thrust::raw_pointer_cast(ctx.d_col_indices.data());
    auto d_labels = ctx.labels.DeviceObject();
    auto* d_offsets = thrust::raw_pointer_cast(ctx.d_lo_row_offset.data());
    bool isDirected = frag.load_strategy == grape::LoadStrategy::kBothOutIn;
    auto& lo_q = ctx.lo_q;

    WorkSourceArray<vertex_t> ws_iv(lo_q.data(), lo_q.size(stream));
    double T_update = grape::GetCurrentTime();
    if (isDirected) {
      ForEachIncomingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid =
                d_offsets[u.GetValue()] + d_frag.GetIncomingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
      ForEachOutgoingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid = d_offsets[u.GetValue()] + d_frag.GetLocalInDegree(u) +
                         d_frag.GetOutgoingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
    } else {
      ForEachOutgoingEdge(
          stream, d_frag, ws_iv,
          [=] __device__(vertex_t u, const nbr_t& nbr) mutable {
            vertex_t v = nbr.get_neighbor();
            size_t eid =
                d_offsets[u.GetValue()] + d_frag.GetOutgoingEdgeIndex(u, nbr);
            p_d_col_indices[eid] = d_labels[v];
          },
          ctx.lb);
    }
    stream.Sync();
    T_update = grape::GetCurrentTime() - T_update;

    auto* local_labels = thrust::raw_pointer_cast(ctx.d_col_indices.data());

    double T_segmentsort = grape::GetCurrentTime();
    {
      size_t num_segments = iv.size();
      size_t num_items = ctx.d_lo_row_offset[num_segments];
      auto* p_d_col_indices =
          thrust::raw_pointer_cast(ctx.d_col_indices.data());
      auto* p_d_sorted_col_indices =
          thrust::raw_pointer_cast(ctx.d_sorted_col_indices.data());

      stream.Sync();
      local_labels =
          SegmentSort(p_d_col_indices, p_d_sorted_col_indices, d_offsets,
                      d_offsets + 1, num_items, num_segments);
    }
    T_segmentsort = grape::GetCurrentTime() - T_segmentsort;

    uint32_t n_vertices = iv.size();

    auto d_new_label = ctx.new_label.DeviceObject();

    double T_counting = grape::GetCurrentTime();
    ForEachWithIndex(
        stream, ws_iv, [=] __device__(size_t idx, vertex_t v) mutable {
          idx = v.GetValue();
          size_t begin = d_offsets[idx], end = d_offsets[idx + 1];
          size_t size = end - begin;

          if (size > 0) {
            label_t new_label;
            label_t curr_label = local_labels[begin];
            int64_t curr_count = 1;
            label_t best_label = 0;
            int64_t best_count = 0;

            // Enumerate its neighbor to find MFL
            // TODO(mengke.mk) Single thread with severe load-imbalance.
            for (auto eid = begin + 1; eid < end; eid++) {
              if (local_labels[eid] != local_labels[eid - 1]) {
                if (curr_count > best_count ||
                    (curr_count == best_count && curr_label < best_label)) {
                  best_label = curr_label;
                  best_count = curr_count;
                }
                curr_label = local_labels[eid];
                curr_count = 1;
              } else {
                ++curr_count;
              }
            }

            if (curr_count > best_count ||
                (curr_count == best_count && curr_label < best_label)) {
              new_label = curr_label;
            } else {
              new_label = best_label;
            }

            if (new_label != d_labels[v]) {
              d_new_label[v].first = new_label;
              d_new_label[v].second = true;
              if (isDirected) {
                d_mm.template SendMsgThroughEdges(d_frag, v, new_label);
              } else {
                d_mm.template SendMsgThroughOEdges(d_frag, v, new_label);
              }
            } else {
              d_new_label[v].second = false;
            }
          }
        });
    stream.Sync();
    T_counting = grape::GetCurrentTime() - T_counting;

    // std::cout << "Frag " << frag.fid() << " update time: " << T_update * 1000
    //          << " segmentsort time: " << T_segmentsort * 1000
    //          << " counting time: " << T_counting * 1000 << std::endl;
  }

  void Update(const fragment_t& frag, context_t& ctx,
              message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto d_labels = ctx.labels.DeviceObject();
    auto d_new_label = ctx.new_label.DeviceObject();

    WorkSourceRange<vertex_t> ws_iv(*iv.begin(), iv.size());

    ForEach(stream, ws_iv, [=] __device__(vertex_t v) mutable {
      if (d_new_label[v].second) {
        d_labels[v] = d_new_label[v].first;
      }
    });
  }

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto& stream = messages.stream();
    auto iv = frag.InnerVertices();
    auto ov = frag.OuterVertices();
    auto d_labels = ctx.labels.DeviceObject();
    auto d_lo_q = ctx.lo_q.DeviceObject();
    auto d_hi_q = ctx.hi_q.DeviceObject();
    WorkSourceRange<vertex_t> ws_iv(*iv.begin(), iv.size());
    WorkSourceRange<vertex_t> ws_ov(*ov.begin(), ov.size());

    thrust::device_vector<size_t> lo_out_degree(iv.size());
    thrust::device_vector<size_t> hi_out_degree(iv.size());
    auto* d_lo_out_degree = thrust::raw_pointer_cast(lo_out_degree.data());
    auto* d_hi_out_degree = thrust::raw_pointer_cast(hi_out_degree.data());

    ++ctx.step;
    if (ctx.step > ctx.max_round) {
      return;
    } else {
      messages.ForceContinue();
    }

    bool isDirected = (frag.load_strategy == grape::LoadStrategy::kBothOutIn);
    ForEachWithIndex(stream, ws_iv,
                     [=] __device__(size_t idx, vertex_t v) mutable {
                       size_t degree = 0;
                       degree = d_frag.GetLocalOutDegree(v);
                       if (isDirected) {
                         degree += d_frag.GetLocalInDegree(v);
                       }
                       if (degree >= 256) {
                         d_hi_q.Append(v);
                         d_hi_out_degree[idx] = degree;
                         d_lo_out_degree[idx] = 0;
                       } else {
                         d_lo_q.Append(v);
                         d_lo_out_degree[idx] = degree;
                         d_hi_out_degree[idx] = 0;
                       }
                     });

    auto* pd_lo_row_offset =
        thrust::raw_pointer_cast(ctx.d_lo_row_offset.data());
    auto* pd_hi_row_offset =
        thrust::raw_pointer_cast(ctx.d_hi_row_offset.data());
    auto size = iv.size();

    PrefixSum(d_lo_out_degree, pd_lo_row_offset + 1, size,
              stream.cuda_stream());
    PrefixSum(d_hi_out_degree, pd_hi_row_offset + 1, size,
              stream.cuda_stream());
    stream.Sync();
    // ReportMemoryUsage("After inclusive sum.");

    size_t lo_size = ctx.d_lo_row_offset[size];
    size_t hi_size = ctx.d_hi_row_offset[size];

    // std::cout << "lo_size: " << lo_size << ", hi_size: " << hi_size
    //          << std::endl;

    ctx.d_col_indices.resize(lo_size, 0);
    ctx.d_sorted_col_indices.resize(lo_size, 0);
    PropagateLabel_lo(frag, ctx, messages);

    ctx.d_hi_label_hash.resize(hi_size, 0);
    ctx.d_hi_col_indices.resize(hi_size, 0);
    ctx.d_hi_label_cnt.resize(hi_size, 0);
    PropagateLabel_hi(frag, ctx, messages);

    Update(frag, ctx, messages);
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto d_frag = frag.DeviceObject();
    auto d_labels = ctx.labels.DeviceObject();

    ctx.step++;

    // receive messages and set labels
    {
      messages.template ParallelProcess<dev_fragment_t, label_t>(
          d_frag, [=] __device__(vertex_t u, label_t msg) mutable {
            d_labels[u] = msg;
          });
    }

    if (ctx.step > ctx.max_round) {
      return;
    } else {
      messages.ForceContinue();
    }

    PropagateLabel_lo(frag, ctx, messages);
    PropagateLabel_hi(frag, ctx, messages);

    Update(frag, ctx, messages);
  }
};  // namespace cuda
}  // namespace cuda
}  // namespace grape
#endif  // __CUDACC__
#endif  // EXAMPLES_ANALYTICAL_APPS_CUDA_CDLP_CDLP_H_
