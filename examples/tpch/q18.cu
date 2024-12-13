/**
 * -- TPC-H Query 18

select
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice,
        sum(l_quantity)
from
        customer,
        orders,
        lineitem
where
        o_orderkey in (
                select
                        l_orderkey
                from
                        lineitem
                group by
                        l_orderkey having
                                sum(l_quantity) > 300
        )
        and c_custkey = o_custkey
        and o_orderkey = l_orderkey
group by
        c_name,
        c_custkey,
        o_orderkey,
        o_orderdate,
        o_totalprice
order by
        o_totalprice desc,
        o_orderdate
limit 100

 */

#include "utils.h"

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <cooperative_groups.h>
#include <cuda.h>

#include <arrow/array.h>
#include <arrow/io/api.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>

#include <iomanip>
#include <iostream>
#define TILE_SIZE 1
namespace cg = cooperative_groups;

struct multijoin_lco_t {
  int32_t lagg_idx;
  int32_t c_idx;
  int32_t o_idx;
};

template <typename AggMap>
__global__ void hash_groupby(AggMap agg_map,
                             int32_t* l_orderkey,
                             int32_t* l_quantity,
                             size_t l_size)
{
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= l_size) return;

  auto value          = l_quantity[tid];
  auto key            = l_orderkey[tid];
  auto [slot, is_new] = agg_map.insert_and_find(cuco::pair{key, value});
  if (!is_new) {
    auto ref =
      cuda::atomic_ref<typename AggMap::mapped_type, cuda::thread_scope_device>{slot->second};
    ref.fetch_add(value, cuda::memory_order_relaxed);
  }
}

template <typename JoinMap>
__global__ void build_joinmap(JoinMap join_map, int32_t* keycol, size_t size)
{
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) return;

  auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  join_map.insert(this_thread, cuco::pair{keycol[tid], tid});
}

template <typename JoinMap>
__global__ void probe_l_c_o_join_size(int32_t* lagg_orderkey,
                                      int32_t* lagg_quantity,
                                      JoinMap o_joinmap,
                                      JoinMap c_joinmap,
                                      int32_t* o_custkey,
                                      int32_t* result_idx,
                                      int32_t agg_size)
{
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= agg_size) return;
  if (lagg_quantity[tid] <= 300) return;  // filter by quantity
  auto o_idx = o_joinmap.find(lagg_orderkey[tid]);
  if (o_idx == o_joinmap.end()) return;
  auto c_idx = c_joinmap.find(o_custkey[o_idx->second]);
  if (c_idx == c_joinmap.end()) return;

  int32_t ridx = atomicAdd(result_idx, 1);
}

template <typename JoinMap>
__global__ void probe_l_c_o_join(int32_t* lagg_orderkey,
                                 int32_t* lagg_quantity,
                                 JoinMap o_joinmap,
                                 JoinMap c_joinmap,
                                 int32_t* o_custkey,
                                 multijoin_lco_t* result,
                                 int32_t* result_idx,
                                 int32_t agg_size)
{
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= agg_size) return;
  if (lagg_quantity[tid] <= 300) return;  // filter by quantity
  auto o_idx = o_joinmap.find(lagg_orderkey[tid]);
  if (o_idx == o_joinmap.end()) return;
  auto c_idx = c_joinmap.find(o_custkey[o_idx->second]);
  if (c_idx == c_joinmap.end()) return;

  int32_t ridx          = atomicAdd(result_idx, 1);
  result[ridx].c_idx    = c_idx->second;
  result[ridx].o_idx    = o_idx->second;
  result[ridx].lagg_idx = tid;
}

int main(int argc, const char** argv)
{
  std::cout << std::setprecision(10);

  std::string dbDir         = getDataDir(argv, argc);
  std::string lineitem_file = dbDir + "lineitem.parquet";
  std::string orders_file   = dbDir + "orders.parquet";
  std::string customer_file = dbDir + "customer.parquet";

  auto lineitem_table  = getArrowTable(lineitem_file);
  auto orders_table    = getArrowTable(orders_file);
  auto customer_table  = getArrowTable(customer_file);
  size_t lineitem_size = lineitem_table->num_rows();
  size_t orders_size   = orders_table->num_rows();
  size_t customer_size = customer_table->num_rows();

  auto l_orderkey = read_column_typecasted<int32_t>(lineitem_table, "l_orderkey");
  auto l_quantity = read_column_typecasted<int32_t>(lineitem_table, "l_quantity");
  auto o_orderkey = read_column_typecasted<int32_t>(orders_table, "o_orderkey");
  auto o_custkey  = read_column_typecasted<int32_t>(orders_table, "o_custkey");
  auto c_custkey  = read_column_typecasted<int32_t>(customer_table, "c_custkey");

  int TB = 1024;

  int32_t *d_l_orderkey, *d_l_quantity, *d_o_orderkey, *d_o_custkey, *d_c_custkey;
  int32_t* d_res_idx;
  auto l_qty_agg_map =
    cuco::static_map{lineitem_size,
                     cuco::empty_key{(int32_t)-1},
                     cuco::empty_value{(int32_t)-1},
                     thrust::equal_to<int32_t>{},
                     cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int32_t>>()};

  auto l_o_joinmap =
    cuco::static_map{orders_size * 2,
                     cuco::empty_key{(int32_t)-1},
                     cuco::empty_value{(int32_t)-1},
                     thrust::equal_to<int32_t>{},
                     cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int32_t>>()};
  auto c_o_joinmap =
    cuco::static_map{customer_size * 2,
                     cuco::empty_key{(int32_t)-1},
                     cuco::empty_value{(int32_t)-1},
                     thrust::equal_to<int32_t>{},
                     cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int32_t>>()};

  cudaMalloc(&d_l_orderkey, sizeof(int32_t) * lineitem_size);
  cudaMalloc(&d_l_quantity, sizeof(int32_t) * lineitem_size);
  cudaMalloc(&d_o_orderkey, sizeof(int32_t) * orders_size);
  cudaMalloc(&d_o_custkey, sizeof(int32_t) * orders_size);
  cudaMalloc(&d_c_custkey, sizeof(int32_t) * customer_size);
  cudaMalloc(&d_res_idx, sizeof(int32_t));
  cudaMemcpy(
    d_l_orderkey, l_orderkey.data(), sizeof(int32_t) * lineitem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_l_quantity, l_quantity.data(), sizeof(int32_t) * lineitem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_o_orderkey, o_orderkey.data(), sizeof(int32_t) * orders_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_o_custkey, o_custkey.data(), sizeof(int32_t) * orders_size, cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_c_custkey, c_custkey.data(), sizeof(int32_t) * customer_size, cudaMemcpyHostToDevice);

  hash_groupby<<<getGridSize(lineitem_size, TB), TB>>>(
    l_qty_agg_map.ref(cuco::insert_and_find), d_l_orderkey, d_l_quantity, lineitem_size);
  // the result of the groupby needs to be materialized in order for the join probing
  int32_t lagg_size = l_qty_agg_map.size();
  thrust::device_vector<int32_t> agg_keys(lagg_size);
  thrust::device_vector<int32_t> agg_values(lagg_size);
  l_qty_agg_map.retrieve_all(agg_keys.begin(), agg_values.begin());
  int32_t* d_lagg_orderkey = thrust::raw_pointer_cast(agg_keys.data());
  int32_t* d_lagg_quantity = thrust::raw_pointer_cast(agg_values.data());

  build_joinmap<<<getGridSize(orders_size, TB), TB>>>(
    l_o_joinmap.ref(cuco::insert), d_o_orderkey, orders_size);
  build_joinmap<<<getGridSize(customer_size, TB), TB>>>(
    c_o_joinmap.ref(cuco::insert), d_c_custkey, customer_size);
  cudaMemset(d_res_idx, 0, sizeof(int32_t));
  probe_l_c_o_join_size<<<getGridSize(lagg_size, TB), TB>>>(d_lagg_orderkey,
                                                            d_lagg_quantity,
                                                            l_o_joinmap.ref(cuco::find),
                                                            c_o_joinmap.ref(cuco::find),
                                                            d_o_custkey,
                                                            d_res_idx,
                                                            lagg_size);
  int32_t lco_joinsize;
  cudaMemcpy(&lco_joinsize, d_res_idx, sizeof(int32_t), cudaMemcpyDeviceToHost);
  multijoin_lco_t* d_loc_join;
  cudaMalloc(&d_loc_join, sizeof(multijoin_lco_t) * lco_joinsize);
  cudaMemset(d_res_idx, 0, sizeof(int32_t));
  probe_l_c_o_join<<<getGridSize(lagg_size, TB), TB>>>(d_lagg_orderkey,
                                                       d_lagg_quantity,
                                                       l_o_joinmap.ref(cuco::find),
                                                       c_o_joinmap.ref(cuco::find),
                                                       d_o_custkey,
                                                       d_loc_join,
                                                       d_res_idx,
                                                       lagg_size);

  int32_t* lagg_qty = (int32_t*)malloc(sizeof(int32_t) * agg_keys.size());
  cudaMemcpy(lagg_qty, d_lagg_quantity, sizeof(int32_t) * agg_keys.size(), cudaMemcpyDeviceToHost);
  multijoin_lco_t* loc_join = (multijoin_lco_t*)malloc(sizeof(multijoin_lco_t) * lco_joinsize);
  cudaMemcpy(loc_join, d_loc_join, sizeof(multijoin_lco_t) * lco_joinsize, cudaMemcpyDeviceToHost);

  StringColumn* c_name = read_string_column(customer_table, "c_name");
  auto o_orderdate     = read_column<int32_t>(orders_table, "o_orderdate");
  auto o_totalprice    = read_column<double>(orders_table, "o_totalprice");

  for (int i = 0; i < lco_joinsize; i++) {
    int c = loc_join[i].c_idx;
    for (int j = 0; j < c_name->sizes[c]; j++) {
      std::cout << c_name->data[c_name->offsets[c] + j];
    }
    std::cout << " " << c_custkey[loc_join[i].c_idx] << " " << o_orderkey[loc_join[i].o_idx];
    std::cout << " " << o_orderdate[loc_join[i].o_idx] << " " << o_totalprice[loc_join[i].o_idx]
              << " " << lagg_qty[loc_join[i].lagg_idx] << std::endl;
  }
}