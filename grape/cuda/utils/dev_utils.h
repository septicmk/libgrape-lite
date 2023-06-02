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
#define BLK_SIZE 256
#define WARP_SHM_SIZE 256
#define BLK_SHM_SIZE 8192

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
  unsigned long long int val_as_ull = (unsigned long long) val;
  return (size_t) atomicAdd(address_as_ull, val_as_ull);
  // unsigned long long int old = *address_as_ull, assumed;

  // do {
  //  assumed = old;
  //  old = atomicCAS(address_as_ull, assumed,
  //                  static_cast<unsigned long long int>(val + assumed));
  //} while (assumed != old);

  // return old;
}

//__inline__ __device__ double atomicAddD(double* address, double val) {
//  unsigned long long int* address_as_ull = (unsigned long long int*) address;
//  unsigned long long int old = *address_as_ull, assumed;
//  if (val == 0.0)
//    return __longlong_as_double(old);
//  do {
//    assumed = old;
//    old = atomicCAS(address_as_ull, assumed,
//                    __double_as_longlong(val +
//                    __longlong_as_double(assumed)));
//  } while (assumed != old);
//  return __longlong_as_double(old);
//}

DEV_INLINE double atomicAddD(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val == 0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
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
DEV_INLINE T block_reduce(T val) {
  static __shared__ int shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warp_reduce(val);
  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
  if (wid == 0)
    val = warp_reduce(val);
  return val;
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

// |--------- bucket stride ----------|
// |----------- bin count ------------|
// |-- bucket num --||-- bucket num --|
template <typename T>
class ShmHashTable {
 public:
  __device__ __forceinline__ void init(T* shm_data, T* global_data, int offset,
                                       int bucket_size, int cached_size,
                                       int bucket_num, int bucket_stride) {
    bin_count = shm_data;
    cache = shm_data + bucket_stride;
    data =
        global_data + blockIdx.x * bucket_stride * (bucket_size - cached_size);
    this->offset = offset;
    this->bucket_size = bucket_size;
    this->cached_size = cached_size;
    this->bucket_num = bucket_num;
    this->bucket_stride = bucket_stride;
  }

  __device__ __forceinline__ void clear(int thread_lane, int csize) {
    for (int i = offset + thread_lane; i < offset + bucket_num; i += csize) {
      bin_count[i] = 0;
    }
  }

  __device__ __forceinline__ bool insert(T element) {
    int key = element & (bucket_num - 1);
    int index = atomicAdd(&bin_count[key + offset], 1);
    if (index < cached_size) {
      cache[index * bucket_stride + offset + key] = element;
    } else if (index < bucket_size) {
      index -= cached_size;
      data[index * bucket_stride + offset + key] = element;
    } else {
      // printf("%d %d %d %d %d\n", threadIdx.x, key, offset, index, element);
      // assert(false);
      return false;
    }
    return true;
  }

  __device__ __forceinline__ bool lookup(T element) {
    int key = element & (bucket_num - 1);
    int index = bin_count[key + offset];
    assert(index < bucket_size);
    int check_cache_depth = index < cached_size ? index : cached_size;
    // check in cache.
    for (int step = 0; step < check_cache_depth; ++step) {
      if (cache[step * bucket_stride + offset + key] == element) {
        return true;
      }
    }
    // check in global memory.
    for (int step = 0; step < index - check_cache_depth; ++step) {
      if (data[step * bucket_stride + offset + key] == element) {
        return true;
      }
    }
    return false;
  }

  T* bin_count;
  T* cache;
  T* data;
  int offset;
  int bucket_size;
  int cached_size;
  int bucket_num;
  int bucket_stride;
};

template <typename T>
DEV_INLINE int64_t binary_search_2phase(T* list, T* cache, T key, size_t size,
                                        size_t shm_per_warp) {
  int mid = 0;
  // phase 1: search in the cache
  int bottom = 0;
  int top = shm_per_warp;
  while (top > bottom + 1) {
    mid = (top + bottom) / 2;
    T y = cache[mid];
    if (key == y)
      return mid * size / shm_per_warp;
    if (key < y)
      top = mid;
    if (key > y)
      bottom = mid;
  }

  // phase 2: search in global memory
  bottom = bottom * size / shm_per_warp;
  top = top * size / shm_per_warp - 1;
  while (top >= bottom) {
    mid = (top + bottom) / 2;
    T y = list[mid];
    if (key == y)
      return mid;
    if (key < y)
      top = mid - 1;
    else
      bottom = mid + 1;
  }
  return -1;
}

template <typename T, typename Y>
DEV_INLINE size_t intersect_num_bs_cache(T* a, size_t size_a, T* b,
                                         size_t size_b, Y callback) {
  if (size_a == 0 || size_b == 0)
    return 0;
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  int nwarp = BLK_SIZE / WARP_SIZE;
  __shared__ T cache[WARP_SHM_SIZE];
  int shm_size = WARP_SHM_SIZE;
  int shm_per_warp = shm_size / nwarp;
  int shm_per_thd = shm_size / BLK_SIZE;
  T* my_cache = cache + warp_lane * shm_per_warp;
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
  for (int i = 0; i < shm_per_thd; ++i) {
    my_cache[i * WARP_SIZE + thread_lane] =
        search[(thread_lane + i * WARP_SIZE) * search_size / shm_per_warp];
  }
  __syncwarp();

  for (auto i = thread_lane; i < lookup_size; i += WARP_SIZE) {
    auto key = lookup[i];  // each thread picks a vertex as the key
    if (binary_search_2phase(search, my_cache, key, search_size,
                             shm_per_warp) >= 0) {
      num += 1;
      callback(key);
    }
  }
  return num;
}

template <typename T, typename Y>
DEV_INLINE size_t intersect_num(T* a, size_t size_a, T* b, size_t size_b,
                                Y callback) {
  size_t t_cnt = intersect_num_bs_cache(a, size_a, b, size_b, callback);
  size_t warp_cnt = warp_reduce(t_cnt);
  __syncwarp();
  return warp_cnt;
}

template <typename T, typename Y>
DEV_INLINE size_t intersect_num_bs_cache_blk(T* a, size_t size_a, T* b,
                                             size_t size_b, Y callback) {
  if (size_a == 0 || size_b == 0)
    return 0;
  int thread_lane =
      threadIdx.x & (BLK_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / BLK_SIZE;  // warp index within the CTA
  int nwarp = BLK_SIZE / BLK_SIZE;
  __shared__ T cache[BLK_SHM_SIZE];
  int shm_size = BLK_SHM_SIZE;
  int shm_per_warp = shm_size / nwarp;
  int shm_per_thd = shm_size / BLK_SIZE;
  T* my_cache = cache + warp_lane * shm_per_warp;
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
  for (int i = 0; i < shm_per_thd; ++i) {
    my_cache[i * BLK_SIZE + thread_lane] =
        search[(thread_lane + i * BLK_SIZE) * search_size / shm_per_warp];
  }
  __syncthreads();

  for (auto i = thread_lane; i < lookup_size; i += BLK_SIZE) {
    auto key = lookup[i];  // each thread picks a vertex as the key
    if (binary_search_2phase(search, my_cache, key, search_size,
                                 shm_per_warp) >= 0) {
      num += 1;
      callback(key);
    }
  }
  return num;
}

template <typename T, typename Y>
DEV_INLINE size_t intersect_num_blk(T* a, size_t size_a, T* b, size_t size_b,
                                    Y callback) {
  size_t t_cnt = intersect_num_bs_cache_blk(a, size_a, b, size_b, callback);
  __syncthreads();
  size_t blk_cnt = block_reduce(t_cnt);
  __syncthreads();
  return blk_cnt;
}

template <typename T>
DEV_INLINE size_t intersect_num_bs_cache_directed(T* a, size_t size_a, T* b,
                                                  size_t size_b, char* wb) {
  if (size_a == 0 || size_b == 0)
    return 0;
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);        // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ T cache[WARP_SHM_SIZE];
  T* lookup = a;
  T* search = b;
  size_t lookup_size = size_a;
  size_t search_size = size_b;
  bool is_reverse = false;
  size_t num = 0;
  if (size_a > size_b) {
    is_reverse = true;
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
    int64_t search_loc =
        binary_search_2phase(search, cache, key, search_size, WARP_SIZE);
    int64_t lookup_loc = i;
    if (search_loc >= 0) {
      if (is_reverse) {
        num += wb[lookup_loc];
      } else {
        num += wb[search_loc];
      }
    }
  }
  return num;
}

template <typename T>
DEV_INLINE size_t intersect_num_directed(T* a, size_t size_a, T* b,
                                         size_t size_b, char* wb) {
  size_t t_cnt = intersect_num_bs_cache_directed(a, size_a, b, size_b, wb);
  size_t warp_cnt = warp_reduce(t_cnt);
  __syncwarp();
  return warp_cnt;
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
