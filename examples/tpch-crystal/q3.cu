#include "utils.h"

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <iostream>
#include <vector>

#define ITEMS_PER_THREAD 2
#define TB_SIZE          512
#define TILE_SIZE        ITEMS_PER_THREAD* TB_SIZE
CUCO_DECLARE_BITWISE_COMPARABLE(double);
namespace cg = cooperative_groups;

template <typename Map>
__global__ void k1_build_customer(int32_t* c_custkey,
                                  int8_t* c_mktsegment,
                                  int8_t mktsegment_predicate,
                                  Map custkey_map,
                                  size_t customer_size)
{
  int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= customer_size) return;
  auto this_thread = cg::tiled_partition<1>(cg::this_thread_block());

#pragma unroll
  for (int j = 0; j < ITEMS_PER_THREAD && (tid + j) < customer_size; j++) {
    if (c_mktsegment[tid + j] != mktsegment_predicate) continue;
    custkey_map.insert(this_thread, cuco::pair{c_custkey[tid + j], tid + j});
  }
}

// template <typename Map>
// __global__ void k2_probe_and_build_count(
//   int32_t *o_orderkey,
//   int32_t *o_custkey,
//   int32_t *o_orderdate,
//   int32_t orderdate_predicate,
//   Map custkey_map,
//   int32_t *count,
//   size_t orders_size
// ) {
//   int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
//   if (tid >= orders_size) return;
//   __shared__ int local_count, winner;
//   local_count = 0;
//   __syncthreads();
//   #pragma unroll
//   for (int j=0; j<ITEMS_PER_THREAD && (tid + j) < orders_size; j++) {
//     if (o_orderdate[tid + j] >= 9204) continue;
//     auto cust_idx_pair = custkey_map.find(o_custkey[tid + j]);
//     if (cust_idx_pair == custkey_map.end()) continue;
//     atomicAdd(&local_count, 1);
//   }
//   winner = threadIdx.x;
//   __syncthreads();
//   if (threadIdx.x == winner) atomicAdd(count, local_count);
// }
template <typename CustMap, typename OrderMap>
__global__ void k2_probe_and_build(int32_t* o_orderkey,
                                   int32_t* o_custkey,
                                   int32_t* o_orderdate,
                                   int32_t orderdate_predicate,
                                   CustMap custkey_map,
                                   OrderMap orderkey_map,
                                   size_t orders_size)
{
  int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= orders_size) return;
  auto this_thread = cg::tiled_partition<1>(cg::this_thread_block());
#pragma unroll
  for (int j = 0; j < ITEMS_PER_THREAD && (tid + j) < orders_size; j++) {
    if (o_orderdate[tid + j] >= 9204) continue;
    auto cust_idx_pair = custkey_map.find(o_custkey[tid + j]);
    if (cust_idx_pair == custkey_map.end()) continue;
    orderkey_map.insert(this_thread, cuco::pair{o_orderkey[tid + j], tid + j});
  }
}

template <typename OrderMap, typename AggMap>
__global__ void k3_groupjoin(int32_t* l_shipdate,
                             int32_t* l_orderkey,
                             double* l_extendedprice,
                             double* l_discount,
                             OrderMap order_map,
                             AggMap agg_map,
                             size_t lineitem_size)
{
  int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= lineitem_size) return;
  auto this_thread = cg::tiled_partition<1>(cg::this_thread_block());
#pragma unroll
  for (int j = 0; j < ITEMS_PER_THREAD && (tid + j) < lineitem_size; j++) {
    if (l_shipdate[tid + j] <= 9204) continue;
    auto order_idx_pair = order_map.find(l_orderkey[tid + j]);
    if (order_idx_pair == order_map.end()) continue;
    double res              = l_extendedprice[tid + j] * (1 - l_discount[tid + j]);
    int64_t aggkey          = l_orderkey[tid + j];
    auto [slot, is_new_key] = agg_map.insert_and_find(cuco::pair{aggkey, res});
    if (!is_new_key) {
      auto ref =
        cuda::atomic_ref<typename AggMap::mapped_type, cuda::thread_scope_device>{slot->second};
      ref.fetch_add(res, cuda::memory_order_relaxed);
    }
  }
}
#include <iomanip>

int main(int argc, const char** argv)
{
  std::cout << std::setprecision(10);
  std::string dir           = getDataDir(argv, argc);
  std::string lineitem_file = dir + "lineitem.parquet";
  std::string customer_file = dir + "customer.parquet";
  std::string orders_file   = dir + "orders.parquet";

  auto lineitem_table  = getArrowTable(lineitem_file);
  size_t lineitem_size = lineitem_table->num_rows();

  auto customer_table  = getArrowTable(customer_file);
  size_t customer_size = customer_table->num_rows();

  auto orders_table  = getArrowTable(orders_file);
  size_t orders_size = orders_table->num_rows();

  /**
   * K1: build customer hash table
   */
  auto c_custkey_map =
    cuco::static_map{customer_size * 2,
                     cuco::empty_key{(int32_t)-1},
                     cuco::empty_value{(int32_t)-1},
                     thrust::equal_to<int32_t>{},
                     cuco::linear_probing<1, cuco::default_hash_function<int32_t>>()};

  auto c_custkey = read_column_typecasted<int32_t>(customer_table, "c_custkey");
  StringDictEncodedColumn* c_mktsegment =
    read_string_dict_encoded_column(customer_table, "c_mktsegment");
  int8_t building_predicate = c_mktsegment->dict["BUILDING"];

  int32_t* d_c_custkey;
  int8_t* d_c_mktsegment;  // Allocate and copy data for d_c_custkey
  cudaMalloc(&d_c_custkey, customer_size * sizeof(int32_t));  // assuming int32_t type for custkey
  cudaMemcpy(
    d_c_custkey, c_custkey.data(), customer_size * sizeof(int32_t), cudaMemcpyHostToDevice);

  // Allocate and copy data for d_c_mktsegment
  cudaMalloc(&d_c_mktsegment,
             customer_size * sizeof(int8_t));  // assuming int8_t type for mktsegment
  cudaMemcpy(
    d_c_mktsegment, c_mktsegment->column, customer_size * sizeof(int8_t), cudaMemcpyHostToDevice);

  int thread_blocks = getGridSize(customer_size, TILE_SIZE);
  k1_build_customer<<<thread_blocks, TB_SIZE>>>(d_c_custkey,
                                                d_c_mktsegment,
                                                building_predicate,
                                                c_custkey_map.ref(cuco::insert),
                                                customer_size);
  CUDACHKERR();
  //  std::cout << "Customers filtered: " << c_custkey_map.size() << std::endl;

  /**
   * K2:
   *   - join customer with orders
   *   - build hash table for orders
   */
  auto o_orderkey                  = read_column_typecasted<int32_t>(orders_table, "o_orderkey");
  auto o_orderdate                 = read_column<int32_t>(orders_table, "o_orderdate");
  auto o_custkey                   = read_column_typecasted<int32_t>(orders_table, "o_custkey");
  int32_t orderdate_predicate = 9204;

  int32_t *d_o_orderkey, *d_o_orderdate, *d_o_custkey;
  // Allocate and copy data for d_o_orderkey
  cudaMalloc(&d_o_orderkey, orders_size * sizeof(int32_t));  // assuming int32_t type for orderkey
  cudaMemcpy(d_o_orderkey, o_orderkey.data(), orders_size * sizeof(int32_t), cudaMemcpyHostToDevice);

  // Allocate and copy data for d_o_custkey
  cudaMalloc(&d_o_custkey, orders_size * sizeof(int32_t));  // assuming int32_t type for custkey
  cudaMemcpy(d_o_custkey, o_custkey.data(), orders_size * sizeof(int32_t), cudaMemcpyHostToDevice);

  // Allocate and copy data for d_o_orderdate
  cudaMalloc(&d_o_orderdate, orders_size * sizeof(int32_t));  // assuming int32_t type for orderdate
  cudaMemcpy(d_o_orderdate, o_orderdate.data(), orders_size * sizeof(int32_t), cudaMemcpyHostToDevice);

  thread_blocks = getGridSize(orders_size, TILE_SIZE);

  // int32_t *d_orders_filtered_count;
  // cudaMalloc(&d_orders_filtered_count, sizeof(int32_t));
  // cudaMemset(d_orders_filtered_count, 0, sizeof(int32_t));
  // k2_probe_and_build_count<<<thread_blocks, TB_SIZE>>>(
  //   d_o_orderkey,
  //   d_o_custkey,
  //   d_o_orderdate,
  //   orderdate_predicate,
  //   c_custkey_map.ref(cuco::find),
  //   d_orders_filtered_count,
  //   orders_size
  // );
  // CUDACHKERR();
  // int32_t orders_filtered_count;
  // cudaMemcpy(&orders_filtered_count, d_orders_filtered_count, sizeof(int32_t),
  // cudaMemcpyDeviceToHost); std::cout << "Total orders filtered: " << orders_filtered_count <<
  // std::endl;
  auto o_orderkey_map =
    cuco::static_map{orders_size * 2,
                     cuco::empty_key{(int32_t)-1},
                     cuco::empty_value{(int32_t)-1},
                     thrust::equal_to<int32_t>{},
                     cuco::linear_probing<1, cuco::default_hash_function<int32_t>>()};
  k2_probe_and_build<<<thread_blocks, TB_SIZE>>>(d_o_orderkey,
                                                 d_o_custkey,
                                                 d_o_orderdate,
                                                 orderdate_predicate,
                                                 c_custkey_map.ref(cuco::find),
                                                 o_orderkey_map.ref(cuco::insert),
                                                 orders_size);
  CUDACHKERR();
  std::cout << "Orders filterd: " << o_orderkey_map.size() << std::endl;
  /**
   * K3
   *  - final join lineitem with order based on order_map
   *  - do aggregation atomically
   */
  auto l_shipdate      = read_column<int32_t>(lineitem_table, "l_shipdate");
  auto l_orderkey      = read_column_typecasted<int32_t>(lineitem_table, "l_orderkey");
  auto o_shippriority  = read_column_typecasted<int32_t>(orders_table, "o_shippriority");
  auto l_extendedprice = read_column<double>(lineitem_table, "l_extendedprice");
  auto l_discount      = read_column<double>(lineitem_table, "l_discount");

  int32_t *d_l_shipdate, *d_l_orderkey, *d_o_shippriority;
  double *d_l_extendedprice, *d_l_discount;
  cudaMalloc(&d_l_shipdate, lineitem_size * sizeof(int32_t));  // assuming int32_t type for shipdate
  cudaMemcpy(
    d_l_shipdate, l_shipdate.data(), lineitem_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_orderkey, lineitem_size * sizeof(int32_t));  // assuming int32_t type for orderkey
  cudaMemcpy(
    d_l_orderkey, l_orderkey.data(), lineitem_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_o_shippriority,
             orders_size * sizeof(int8_t));  // assuming int8_t type for shippriority
  cudaMemcpy(
    d_o_shippriority, o_shippriority.data(), orders_size * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_extendedprice,
             lineitem_size * sizeof(double));  // assuming double type for extendedprice
  cudaMemcpy(d_l_extendedprice,
             l_extendedprice.data(),
             lineitem_size * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_discount, lineitem_size * sizeof(double));  // assuming double type for discount
  cudaMemcpy(
    d_l_discount, l_discount.data(), lineitem_size * sizeof(double), cudaMemcpyHostToDevice);

  auto agg_map  = cuco::static_map{o_orderkey_map.size(),
                                  cuco::empty_key{(int64_t)-1},
                                  cuco::empty_value{0.},
                                  thrust::equal_to<int64_t>{},
                                  cuco::linear_probing<1, cuco::default_hash_function<int64_t>>()};
  thread_blocks = getGridSize(lineitem_size, TILE_SIZE);
  k3_groupjoin<<<thread_blocks, TB_SIZE>>>(d_l_shipdate,
                                           d_l_orderkey,
                                           d_l_extendedprice,
                                           d_l_discount,
                                           o_orderkey_map.ref(cuco::find),
                                           agg_map.ref(cuco::insert_and_find),
                                           lineitem_size);
  CUDACHKERR();
  std::cout << "Size of final result: " << agg_map.size() << std::endl;

  int agg_map_size = agg_map.size();
  std::cout << agg_map.capacity() << "| map size: " << agg_map_size << std::endl;
  thrust::device_vector<double> result_rev(agg_map_size);
  thrust::device_vector<int64_t> result_keys(agg_map_size);
  agg_map.retrieve_all(result_keys.begin(), result_rev.begin());

  std::vector<std::pair<double, int64_t>> rev;
  for (int i = 0; i < agg_map_size; i++) {
    //   std::cout << result_keys[i] << " : " << result_rev[i] << std::endl;
    rev.push_back(std::make_pair(result_rev[i], result_keys[i]));
  }
  std::sort(rev.rbegin(), rev.rend());
  std::cout << "Total results: " << rev.size() << std::endl;
  std::cout << "Printing the first 10 sorted revenues:\n";
  for (int i = 0; i < 10; i++) {
    std::cout << rev[i].second << " : " << rev[i].first << std::endl;
  }
}
