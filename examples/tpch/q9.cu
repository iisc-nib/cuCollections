/*
-- TPC-H Query 9

select
        nation,
        o_year,
        sum(amount) as sum_profit
from
        (
                select
                        n_name as nation,
                        extract(year from o_orderdate) as o_year,
                        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
                from
                        part,
                        supplier,
                        lineitem,
                        partsupp,
                        orders,
                        nation
                where
                        s_suppkey = l_suppkey
                        and ps_suppkey = l_suppkey
                        and ps_partkey = l_partkey
                        and p_partkey = l_partkey
                        and o_orderkey = l_orderkey
                        and s_nationkey = n_nationkey
                        and p_name like '%green%'
        ) as profit
group by
        nation,
        o_year
order by
        nation,
        o_year desc

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

CUCO_DECLARE_BITWISE_COMPARABLE(double);

#define TILE_SIZE 1
namespace cg = cooperative_groups;
template <typename Map>
__global__ void build_hash_primary_key(Map map_ref, int32_t* nationkey, size_t nationsize)
{
  int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nationsize) return;
  auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  map_ref.insert(this_thread, cuco::pair{nationkey[tid], tid});
}

struct multijoin_t {
  int32_t n_idx;
  int32_t s_idx;
  int32_t ps_idx;
};

template <typename ProbeMap>
__global__ void probe_partsupp_size(ProbeMap s_map_ref,
                                    ProbeMap n_map_ref,
                                    int32_t* ps_suppkey,
                                    int32_t* s_suppkey,
                                    int32_t* s_nationkey,
                                    int32_t* n_nationkey,
                                    int32_t* res_size,
                                    size_t ps_size)
{
  int32_t tid = (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= ps_size) return;
  // now join the part supp with supplier
  auto s_idx = s_map_ref.find(ps_suppkey[tid]);
  if (s_idx == s_map_ref.end()) return;
  // get the nation idx, based on the supplier idx
  auto n_idx = n_map_ref.find(s_nationkey[s_idx->second]);
  if (n_idx == n_map_ref.end()) return;
  // append n,s,ps indices to the result
  atomicAdd(res_size, 1);
}

template <typename ProbeMap>
__global__ void probe_partsupp(ProbeMap s_map_ref,
                               ProbeMap n_map_ref,
                               int32_t* ps_suppkey,
                               int32_t* s_suppkey,
                               int32_t* s_nationkey,
                               int32_t* n_nationkey,
                               multijoin_t* result,
                               int32_t* res_idx,
                               size_t ps_size)
{
  int32_t tid = (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= ps_size) return;
  // now join the part supp with supplier
  auto s_idx = s_map_ref.find(ps_suppkey[tid]);
  if (s_idx == s_map_ref.end()) return;
  // get the nation idx, based on the supplier idx
  auto n_idx = n_map_ref.find(s_nationkey[s_idx->second]);
  if (n_idx == n_map_ref.end()) return;
  // append n,s,ps indices to the result
  auto idx           = atomicAdd(res_idx, 1);
  result[idx].n_idx  = n_idx->second;
  result[idx].s_idx  = s_idx->second;
  result[idx].ps_idx = tid;
}

int main(int argc, const char** argv)
{
  std::cout << std::setprecision(10);

  std::string dbDir         = getDataDir(argv, argc);
  std::string lineitem_file = dbDir + "lineitem.parquet";
  std::string part_file     = dbDir + "part.parquet";
  std::string orders_file   = dbDir + "orders.parquet";
  std::string nation_file   = dbDir + "nation.parquet";
  std::string supplier_file = dbDir + "supplier.parquet";
  std::string partsupp_file = dbDir + "partsupp.parquet";

  auto lineitem_table = getArrowTable(lineitem_file);
  auto part_table     = getArrowTable(part_file);
  auto orders_table   = getArrowTable(orders_file);
  auto nation_table   = getArrowTable(nation_file);
  auto supplier_table = getArrowTable(supplier_file);
  auto partsupp_table = getArrowTable(partsupp_file);

  size_t lineitem_size = lineitem_table->num_rows();
  size_t part_size     = part_table->num_rows();
  size_t orders_size   = orders_table->num_rows();
  size_t nation_size   = nation_table->num_rows();
  size_t supplier_size = supplier_table->num_rows();
  size_t partsupp_size = partsupp_table->num_rows();

  // now we need to semi-materialize the join
  // first join supplier and nation and semi materialize into a separate table.
  auto n_nationkey_map =
    cuco::static_map{nation_size * 2,
                     cuco::empty_key{(int32_t)-1},
                     cuco::empty_value{(int32_t)-1},
                     thrust::equal_to<int32_t>{},
                     cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int32_t>>()};

  auto s_supplierkey_map =
    cuco::static_map{supplier_size * 2,
                     cuco::empty_key{(int32_t)-1},
                     cuco::empty_value{(int32_t)-1},
                     thrust::equal_to<int32_t>{},
                     cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int32_t>>()};
  int TB = 1024;
  int32_t *n_nationkey, *d_n_nationkey;
  int32_t *s_supplierkey, *d_s_supplierkey;
  int32_t *s_nationkey, *d_s_nationkey;
  int32_t *ps_suppkey, *d_ps_suppkey;
  n_nationkey   = read_column_typecasted<int32_t>(nation_table, "n_nationkey");
  s_supplierkey = read_column_typecasted<int32_t>(supplier_table, "s_suppkey");
  s_nationkey   = read_column_typecasted<int32_t>(supplier_table, "s_nationkey");
  ps_suppkey    = read_column_typecasted<int32_t>(partsupp_table, "ps_suppkey");

  cudaMalloc(&d_n_nationkey, nation_size * sizeof(int32_t));
  cudaMalloc(&d_s_supplierkey, supplier_size * sizeof(int32_t));
  cudaMalloc(&d_s_nationkey, supplier_size * sizeof(int32_t));
  cudaMalloc(&d_ps_suppkey, partsupp_size * sizeof(int32_t));
  cudaMemcpy(d_n_nationkey, n_nationkey, nation_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_s_supplierkey, s_supplierkey, supplier_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s_nationkey, s_nationkey, supplier_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ps_suppkey, ps_suppkey, partsupp_size * sizeof(int32_t), cudaMemcpyHostToDevice);

  build_hash_primary_key<<<getGridSize(nation_size, TB), TB>>>(
    n_nationkey_map.ref(cuco::insert), d_n_nationkey, nation_size);

  build_hash_primary_key<<<getGridSize(supplier_size, TB), TB>>>(
    s_supplierkey_map.ref(cuco::insert), d_s_supplierkey, supplier_size);
  cudaDeviceSynchronize();
  int32_t join_size, *d_join_size;
  cudaMalloc(&d_join_size, sizeof(int32_t));
  cudaMemset(d_join_size, 0, sizeof(int32_t));
  probe_partsupp_size<<<getGridSize(partsupp_size, TB), TB>>>(s_supplierkey_map.ref(cuco::find),
                                                              n_nationkey_map.ref(cuco::find),
                                                              d_ps_suppkey,
                                                              d_s_supplierkey,
                                                              d_s_nationkey,
                                                              d_n_nationkey,
                                                              d_join_size,
                                                              partsupp_size);
  cudaMemcpy(&join_size, d_join_size, sizeof(int32_t), cudaMemcpyDeviceToHost);
  multijoin_t *n_s_ps_join, *d_n_s_ps_join;
  cudaMalloc(&d_n_s_ps_join, sizeof(multijoin_t)*join_size);
  cudaMemset(d_join_size, 0, sizeof(int32_t));
  probe_partsupp<<<getGridSize(partsupp_size, TB), TB>>>(s_supplierkey_map.ref(cuco::find),
                                                              n_nationkey_map.ref(cuco::find),
                                                              d_ps_suppkey,
                                                              d_s_supplierkey,
                                                              d_s_nationkey,
                                                              d_n_nationkey,
                                                              d_n_s_ps_join,
                                                              d_join_size,
                                                              partsupp_size);
  n_s_ps_join = (multijoin_t*)malloc(sizeof(multijoin_t)*join_size);
  cudaMemcpy(n_s_ps_join, d_n_s_ps_join, sizeof(multijoin_t)*join_size, cudaMemcpyDeviceToHost);

  for (size_t i=0; i<join_size; i++) {
    auto n_idx = n_s_ps_join[i].n_idx;
    auto s_idx = n_s_ps_join[i].s_idx;
    auto ps_idx = n_s_ps_join[i].ps_idx;
    std::cout << n_nationkey[n_idx] << " " << s_nationkey[s_idx] << " " << s_supplierkey[s_idx] << " " << 
      ps_suppkey[ps_idx] << "\n";
  }

  std::cout << "nationkey size: " << n_nationkey_map.size() << "\n";
  std::cout << "supplierkey size: " << s_supplierkey_map.size() << "\n";
  std::cout << "nation, supplier, partsupplier sizes: " << nation_size << " " << supplier_size << " " << partsupp_size << "\n";
  std::cout << "supplier, nation and partsupplier join size: " << join_size << "\n";
}