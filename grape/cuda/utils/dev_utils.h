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

#ifndef GRAPE_CUDA_UTILS_DEV_UTILS_H_
#define GRAPE_CUDA_UTILS_DEV_UTILS_H_
#include "grape/cuda/utils/array_view.h"
#include "grape/cuda/utils/launcher.h"

__device__ static const char logtable[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1,    0,     1,     1,     2,     2,     2,     2,     3,     3,     3,
    3,     3,     3,     3,     3,     LT(4), LT(5), LT(5), LT(6), LT(6), LT(6),
    LT(6), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)};

namespace grape {
namespace cuda {

namespace dev {

#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 256

DEV_HOST_INLINE size_t round_up(size_t numerator, size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

// Refer:https://github.com/gunrock/gunrock/blob/a7fc6948f397912ca0c8f1a8ccf27d1e9677f98f/gunrock/oprtr/intersection/cta.cuh#L84
DEV_INLINE unsigned ilog2(unsigned int v) {
  register unsigned int t, tt;
  if (tt = v >> 16)
    return ((t = tt >> 8) ? 24 + logtable[t] : 16 + logtable[tt]);
  else
    return ((t = v >> 8) ? 8 + logtable[t] : logtable[v]);
}
// Refer:
// https://forums.developer.nvidia.com/t/why-doesnt-runtime-library-provide-atomicmax-nor-atomicmin-for-float/164171/7
DEV_INLINE float atomicMinFloat(float* addr, float value) {
  if (isnan(value)) {
    return *addr;
  }
  value += 0.0f;
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMin(reinterpret_cast<int*>(addr),
                                       __float_as_int(value)))
            : __uint_as_float(atomicMax(reinterpret_cast<unsigned int*>(addr),
                                        __float_as_uint(value)));

  return old;
}

DEV_INLINE size_t atomicAdd64(size_t* address, size_t val) {
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long int>(val + assumed));
  } while (assumed != old);

  return old;
}

template <typename T>
DEV_INLINE bool BinarySearch(const ArrayView<T>& array, const T& target) {
  size_t l = 0;
  size_t r = array.size();
  while (l < r) {
    size_t m = (l + r) >> 1;
    const T& elem = array[m];

    if (elem == target) {
      return true;
    } else if (elem > target) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  return false;
}

template <typename T>
DEV_INLINE bool BinarySearchWarp(const ArrayView<T>& array, const T& target) {
  size_t cosize = 32;
  size_t lane = threadIdx.x & (cosize - 1);
  size_t worknum = array.size();
  size_t x = worknum / cosize;
  size_t y = worknum % cosize;
  // size_t per_work = worknum / cosize + (lane < worknum % cosize);
  size_t l = lane * x + (lane < y ? lane : y);
  size_t r = l + x + (lane < y ? 1 : 0);
  bool found = false;
  __syncwarp();
  while (l < r) {
    size_t m = (l + r) >> 1;
    const T& elem = array[m];

    if (elem == target) {
      found = true;
      break;
    } else if (elem > target) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  __syncwarp();
  return __any_sync(__activemask(), found);
}

template <int NT, typename T>
DEV_INLINE int BinarySearch(const T* arr, const T& key) {
  int mid = ((NT >> 1) - 1);

  if (NT > 512)
    mid = arr[mid] > key ? mid - 256 : mid + 256;
  if (NT > 256)
    mid = arr[mid] > key ? mid - 128 : mid + 128;
  if (NT > 128)
    mid = arr[mid] > key ? mid - 64 : mid + 64;
  if (NT > 64)
    mid = arr[mid] > key ? mid - 32 : mid + 32;
  if (NT > 32)
    mid = arr[mid] > key ? mid - 16 : mid + 16;
  mid = arr[mid] > key ? mid - 8 : mid + 8;
  mid = arr[mid] > key ? mid - 4 : mid + 4;
  mid = arr[mid] > key ? mid - 2 : mid + 2;
  mid = arr[mid] > key ? mid - 1 : mid + 1;
  mid = arr[mid] > key ? mid : mid + 1;

  return mid;
}

#if __CUDACC_VER_MAJOR__ >= 9
#define SHFL_DOWN(a, b) __shfl_down_sync(0xFFFFFFFF, a, b)
#define SHFL(a, b) __shfl_sync(0xFFFFFFFF, a, b)
#else
#define SHFL_DOWN(a, b) __shfl_down(a, b)
#define SHFL(a, b) __shfl(a, b)
#endif

template <typename T>
DEV_INLINE T warp_reduce(T val) {
  T sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8);
  sum += SHFL_DOWN(sum, 4);
  sum += SHFL_DOWN(sum, 2);
  sum += SHFL_DOWN(sum, 1);
  sum = SHFL(sum, 0);
  return sum;
}

template <typename T>
DEV_INLINE size_t CountCommonNeighbor(T* u_start, size_t u_len, T* v_start,
                                      size_t v_len) {
  int cosize = 32;
  int lane = threadIdx.x & (cosize - 1);

  // exchange u v to make v_len is samller than u_len
  if (v_len < u_len) {
    T* t = v_start;
    v_start = u_start;
    u_start = t;
    size_t tt = v_len;
    v_len = u_len;
    u_len = tt;
  }

  // calculate work per thred
  size_t total = u_len + v_len;
  size_t remain = total % cosize;
  size_t per_work = total / cosize;

  size_t diag_id = per_work * lane + (lane < remain ? lane : remain);
  size_t diag_work = per_work + (lane < remain ? 1 : 0);

  T u_min, u_max, v_min, v_max;
  int found;
  size_t global_cnt = 0;
  if (diag_work > 0) {
    found = (diag_id == 0 ? 1 : 0);
    T u_cur = 0, v_cur = 0;
    u_min = diag_id < v_len ? 0 : (diag_id - v_len);
    v_min = diag_id < u_len ? 0 : (diag_id - u_len);

    u_max = diag_id < u_len ? diag_id : u_len;
    v_max = diag_id < v_len ? diag_id : v_len;
    while (!found) {
      u_cur = (u_min + u_max) >> 1;
      v_cur = diag_id - u_cur;

      T u_test = u_start[u_cur];
      T v_test = v_start[v_cur];
      if (u_test == v_test) {
        found = 1;
        break;
      }

      // only two element
      if (u_max - u_min == 1 && v_max - v_min == 1) {
        found = 1;
        if (u_test < v_test) {
          u_cur = u_max;
          v_cur = diag_id - u_cur;
        }
        break;
      }

      if (u_test < v_test) {
        u_min = u_cur;
        v_max = v_cur;
      } else {
        u_max = u_cur;
        v_min = v_cur;
      }
    }
    size_t local = 0;
    if ((u_cur < u_len) && (v_cur < v_len)) {
      size_t idx = 0;
      while (idx < diag_work) {
        T u_test = u_start[u_cur];
        T v_test = v_start[v_cur];
        int64_t comp = (int64_t) u_test - (int64_t) v_test;
        // local = __any_sync(0xffffffff, comp==0); // __any -> __any_sync
        local = __any(comp == 0);
        if (local == 1) {
          global_cnt = 1;
          break;
        }

        u_cur += (comp <= 0);
        v_cur += (comp >= 0);
        idx += (comp == 0) + 1;

        if ((v_cur == v_len) || (u_cur == u_len))
          break;
      }
    }
  }
  return global_cnt;
}

template <typename T>
DEV_INLINE bool binary_search_2phase(T* list, T* cache, T key, size_t size) {
  int p = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  int mid = 0;
  // phase 1: search in the cache
  int bottom = 0;
  int top = WARP_SIZE;
  while (top > bottom + 1) {
    mid = (top + bottom) / 2;
    T y = cache[p + mid];
    if (key == y)
      return true;
    if (key < y)
      top = mid;
    if (key > y)
      bottom = mid;
  }

  // phase 2: search in global memory
  bottom = bottom * size / WARP_SIZE;
  top = top * size / WARP_SIZE - 1;
  while (top >= bottom) {
    mid = (top + bottom) / 2;
    T y = list[mid];
    if (key == y)
      return true;
    if (key < y)
      top = mid - 1;
    else
      bottom = mid + 1;
  }
  return false;
}

template <typename T, typename Y>
DEV_INLINE size_t intersect_num(T* a, size_t size_a, T* b, size_t size_b,
                                Y* d_tricnt) {
  size_t t_cnt = intersect_num_bs_cache(a, size_a, b, size_b, d_tricnt);
  size_t warp_cnt = warp_reduce(t_cnt);
  __syncwarp();
  return warp_cnt;
}

template <typename T, typename Y>
DEV_INLINE size_t intersect_num_bs_cache(T* a, size_t size_a, T* b, size_t size_b,
                                         Y* d_tricnt) {
  if (size_a == 0 || size_b == 0)
    return 0;
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T cache[MAX_BLOCK_SIZE];
  size_t num = 0;
  T* lookup = a;
  T* search = b;
  size_t lookup_size = size_a;
  size_t search_size = size_b;
  if (size_a > size_b) {
    lookup = b;
    search = a;
    lookup_size = size_b;
    search_size = size_a;
  }
  cache[warp_lane * WARP_SIZE + thread_lane] =
      search[thread_lane * search_size / WARP_SIZE];
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i];  // each thread picks a vertex as the key
    if (binary_search_2phase(search, cache, key, search_size)) {
      num += 1;
      dev::atomicAdd64(&d_tricnt[vertex_t(key)], 1);
    }
  }
  return num;
}

void WarmupNccl(const grape::CommSpec& comm_spec, const Stream& stream,
                std::shared_ptr<ncclComm_t>& nccl_comm) {
  auto fnum = comm_spec.fnum();
  auto fid = comm_spec.fid();
  std::vector<thrust::device_vector<char>> buf_in(fnum), buf_out(fnum);

  for (int _ = 0; _ < 10; _++) {
    size_t length = 4 * 1024 * 1024;

    CHECK_NCCL(ncclGroupStart());

    for (fid_t i = 1; i < fnum; ++i) {
      fid_t src_fid = (fid + i) % fnum;
      int peer = comm_spec.FragToWorker(src_fid);

      buf_in[src_fid].resize(length);
      CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(buf_in[src_fid].data()),
                          length, ncclChar, peer, *nccl_comm,
                          stream.cuda_stream()));
    }

    for (fid_t i = 1; i < fnum; ++i) {
      fid_t dst_fid = (fid + fnum - i) % fnum;
      int peer = comm_spec.FragToWorker(dst_fid);

      buf_out[dst_fid].resize(length);
      CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(buf_out[dst_fid].data()),
                          length, ncclChar, peer, *nccl_comm,
                          stream.cuda_stream()));
    }

    CHECK_NCCL(ncclGroupEnd());
    stream.Sync();
  }
}

template <typename T>
void ncclSendRecv(const grape::CommSpec& comm_spec, const Stream& stream,
                  std::shared_ptr<ncclComm_t>& nccl_comm,
                  const std::vector<int>& migrate_to,
                  const ArrayView<T>& send_buf, ArrayView<T> recv_buf) {
  int to_rank = migrate_to[comm_spec.worker_id()];

  if (to_rank != -1) {
    size_t send_size = send_buf.size();
    MPI_Send(&send_size, 1, MPI_UINT64_T, to_rank, 1, comm_spec.comm());
    CHECK_NCCL(ncclSend(send_buf.data(), sizeof(T) * send_buf.size(), ncclChar,
                        to_rank, *nccl_comm, stream.cuda_stream()));
  } else {
    for (int src_worker_id = 0; src_worker_id < comm_spec.worker_num();
         src_worker_id++) {
      if (migrate_to[src_worker_id] == comm_spec.worker_id()) {
        size_t recv_size;
        MPI_Status stat;
        MPI_Recv(&recv_size, 1, MPI_UINT64_T, src_worker_id, 1,
                 comm_spec.comm(), &stat);
        CHECK_NCCL(ncclRecv(recv_buf.data(), sizeof(T) * recv_size, ncclChar,
                            src_worker_id, *nccl_comm, stream.cuda_stream()));
      }
    }
  }
}

}  // namespace dev
}  // namespace cuda
}  // namespace grape

#endif  // GRAPE_CUDA_UTILS_DEV_UTILS_H_
