





#include "utils.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <arrow/array.h>
#include <arrow/io/api.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>

#include <iomanip>
#include <iostream>

#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>

#include <cooperative_groups.h>
#include <cuda.h>
#include <set>

#define ITEMS_PER_THREAD 4
#define TB_SIZE          256
#define TILE_SIZE        ITEMS_PER_THREAD* TB_SIZE

CUCO_DECLARE_BITWISE_COMPARABLE(double);
namespace cg = cooperative_groups;

#define HYPER 0
#define CRYSTAL 1

template <typename Map, int ExecStyle>
__global__ void build_hash_primary_key(Map map_ref, int32_t* nationkey, size_t nationsize)
{
  int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= nationsize) return;
#pragma unroll
  for (int j = 0; j < ITEMS_PER_THREAD && (tid + j) < nationsize; j++) {
    auto this_thread = cg::tiled_partition<1>(cg::this_thread_block());
    map_ref.insert(this_thread, cuco::pair{nationkey[tid + j], tid + j});
  }
}

template <typename Map, int ExecStyle>
__global__ void probe_size(Map map_ref, int32_t* probe_col, int64_t size, int* oid_table_size)
{
  int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= size) return;

  if (ExecStyle == CRYSTAL) {
    // probe in a tiled fashion
    int probe_count = 0;
    for (int i = 0; i < ITEMS_PER_THREAD && (tid + i) < size; i++) {
      auto t1_idx = map_ref.find(probe_col[tid + i]);
      if (t1_idx == map_ref.end()) continue;
      probe_count++;
    }
    // __syncthreads();
    __shared__ int tb_probe_count;
    tb_probe_count = 0;
    __syncthreads();
    atomicAdd(&tb_probe_count, probe_count);
    __syncthreads();

    // store in global memory
    if (threadIdx.x == 0) atomicAdd(oid_table_size, tb_probe_count);
  } else if (ExecStyle == HYPER) {
    for (int i = 0; i < ITEMS_PER_THREAD && (tid + i) < size; i++) {
      auto t1_idx = map_ref.find(probe_col[tid + i]);
      if (t1_idx == map_ref.end()) continue;
      // probe_count++;
      atomicAdd(oid_table_size, 1);
    }
  }

}

template <typename Map, int ExecStyle>
__global__ void probe(
  Map map_ref, int32_t* probe_col, int64_t size, int64_t* oid_table, int* oid_table_size)
{
  int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);

  if (ExecStyle == CRYSTAL) {
    int32_t items[ITEMS_PER_THREAD];
    for (int i = 0; i < ITEMS_PER_THREAD && (tid + i) < size; i++) {
      items[i] = probe_col[tid + i];
    }

    // probe in a tiled fashion
    int probe_count = 0;
    __shared__ int buffer[TILE_SIZE * 2];
    __shared__ int buffer_ptr;
    buffer_ptr = 0;
    __syncthreads();
    for (int i = 0; i < ITEMS_PER_THREAD && (tid + i) < size; i++) {
      auto t1_idx = map_ref.find(items[i]);
      if (t1_idx == map_ref.end()) continue;
      int buffer_idx             = atomicAdd(&buffer_ptr, 2);
      buffer[buffer_idx]     = t1_idx->second;
      buffer[buffer_idx + 1] = tid + i;
      if (t1_idx->second == 0 && (tid + i) == 0) printf("DEBUG: both zero found\n");
    }
    __syncthreads();  // always use this when you expect shared memory to be updated
    // printf("%d\n", buffer_ptr);
    __shared__ int oid_table_pos;
    if (threadIdx.x == 0) { oid_table_pos = atomicAdd(oid_table_size, buffer_ptr); }
    __syncthreads();
    
    // block store
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      if (threadIdx.x + (2*i) * blockDim.x < buffer_ptr) {
        oid_table[oid_table_pos + threadIdx.x + (2*i) * blockDim.x] =
          buffer[threadIdx.x + (2*i) * blockDim.x];
      }
      if (threadIdx.x + (2*i + 1) * blockDim.x < buffer_ptr) {
        oid_table[oid_table_pos + threadIdx.x + (2*i + 1) * blockDim.x] =
          buffer[threadIdx.x + (2*i + 1) * blockDim.x];
      }
    }
  } else if (ExecStyle == HYPER) {
    int probe_count = 0;
    for (int i = 0; i < ITEMS_PER_THREAD && (tid + i) < size; i++) {
      auto t1_idx = map_ref.find(probe_col[tid + i]);
      if (t1_idx == map_ref.end()) continue;
      int oid_idx                = atomicAdd(oid_table_size, 1);
      oid_table[2 * oid_idx]     = t1_idx->second;
      oid_table[2 * oid_idx + 1] = tid + i;
    }
  }
}

/*
Test join of oid tables
*/
bool test_join_result(int64_t* expected_oid_table, int64_t expected_size,
  int64_t* actual_oid_table, int64_t actual_size) {
    if (expected_size != actual_size) return false;
    std::set<std::pair<int64_t, int64_t>> expected, actual;
    for (int i=0; i < expected_size; i++) {
      expected.insert(std::make_pair(expected_oid_table[2*i], expected_oid_table[2*i + 1]));
      actual.insert(std::make_pair(actual_oid_table[2*i], actual_oid_table[2*i + 1]));
    }
    return actual == expected;
  }

class StdOperations {
public:
  template<int ExecStyle>
  static int64_t* binary_join(std::shared_ptr<arrow::Table> t1,
                               std::shared_ptr<arrow::Table> t2,
                               std::string t1_key,
                               std::string t2_key)
  {
    auto t1_key_col = read_column_typecasted<int32_t>(t1, t1_key);
    auto t2_key_col = read_column_typecasted<int32_t>(t2, t2_key);

    int32_t *d_t1_key, *d_t2_key;
    cudaMalloc(&d_t1_key, t1->num_rows() * sizeof(int32_t));
    cudaMalloc(&d_t2_key, t2->num_rows() * sizeof(int32_t));
    cudaMemcpy(
      d_t1_key, t1_key_col.data(), t1->num_rows() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(
      d_t2_key, t2_key_col.data(), t2->num_rows() * sizeof(int32_t), cudaMemcpyHostToDevice);

    int TB = TILE_SIZE / ITEMS_PER_THREAD;

    auto hash_table =
      cuco::static_map{t1->num_rows() * 2,
                       cuco::empty_key{(int32_t)-1},
                       cuco::empty_value{(int32_t)-1},
                       thrust::equal_to<int32_t>{},
                       cuco::linear_probing<1, cuco::default_hash_function<int32_t>>()};
    build_hash_primary_key<decltype(hash_table.ref(cuco::insert)), ExecStyle><<<getGridSize(t1->num_rows(), TB * ITEMS_PER_THREAD), TB>>>(
      hash_table.ref(cuco::insert), d_t1_key, t1->num_rows());
    int64_t oid_table_size;
    int* d_oid_table_size;
    cudaMalloc(&d_oid_table_size, sizeof(int64_t));
    cudaMemset(d_oid_table_size, 0, sizeof(int64_t));

    probe_size<decltype(hash_table.ref(cuco::find)), ExecStyle><<<getGridSize(t2->num_rows(), TB * ITEMS_PER_THREAD), TB>>>(
      hash_table.ref(cuco::find), d_t2_key, t2->num_rows(), d_oid_table_size);
    cudaMemcpy(&oid_table_size, d_oid_table_size, sizeof(int64_t), cudaMemcpyDeviceToHost);
    std::cout << "oid table size: " << oid_table_size << std::endl;
    std::cout << "t2 table size: " << t2->num_rows() << std::endl;
    int64_t* d_oid_table;
    cudaMalloc(&d_oid_table, sizeof(int64_t) * oid_table_size * 2);
    cudaMemset(d_oid_table_size, 0, sizeof(int64_t));
    probe<decltype(hash_table.ref(cuco::find)), ExecStyle><<<getGridSize(t2->num_rows(), TB * ITEMS_PER_THREAD), TB>>>(
      hash_table.ref(cuco::find), d_t2_key, t2->num_rows(), d_oid_table, d_oid_table_size);

    int64_t* oid_table;
    oid_table = (int64_t*)malloc(sizeof(int64_t) * oid_table_size * 2);
    cudaMemcpy(
      oid_table, d_oid_table, sizeof(int64_t) * 2 * oid_table_size, cudaMemcpyDeviceToHost);

    // verify the results
    for (int i = 0; i < oid_table_size; i++) {
      assert(t1_key_col.data()[oid_table[2 * i]] == t2_key_col.data()[oid_table[2 * i + 1]]);
    }
    // free all the temporary structures
    cudaFree(d_t1_key);
    cudaFree(d_t2_key);
    cudaFree(d_oid_table);
    cudaFree(d_oid_table_size);

    return oid_table;
  }
};

int main(int argc, const char** argv)
{
  std::cout << std::setprecision(15);

  std::string dbDir         = getDataDir(argv, argc);
  // std::string lineitem_file = dbDir + "lineitem.parquet";
  // std::string part_file     = dbDir + "part.parquet";
  // std::string orders_file   = dbDir + "orders.parquet";
  std::string nation_file   = dbDir + "nation.parquet";
  std::string supplier_file = dbDir + "supplier.parquet";
  std::string partsupp_file = dbDir + "partsupp.parquet";

  // auto lineitem_table = getArrowTable(lineitem_file);
  // auto part_table     = getArrowTable(part_file);
  // auto orders_table   = getArrowTable(orders_file);
  auto nation_table   = getArrowTable(nation_file);
  auto supplier_table = getArrowTable(supplier_file);
  auto partsupp_table = getArrowTable(partsupp_file);

  StdOperations stdOps;
  std::cout << "Join in crystal style\n";
  auto t1 = stdOps.binary_join<CRYSTAL>(nation_table, supplier_table, "n_nationkey", "s_nationkey");
  std::cout << "Join in hyper style\n";
  auto t2 = stdOps.binary_join<HYPER>(nation_table, supplier_table, "n_nationkey", "s_nationkey");
  std::cout << "join result: " << test_join_result(t1, 10000, t2, 10000) << std::endl; 
  return 0;
}