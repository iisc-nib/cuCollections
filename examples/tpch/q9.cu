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
#include <cuco/static_multimap.cuh>

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
void CUDACHKERR()
{
  auto err = cudaGetLastError();
  if (err != cudaSuccess) { std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << "\n"; }
}
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

template <typename Map>
__global__ void build_hash_primary_key_partname_filter(
  Map map_ref, int32_t* keycol, size_t size, char* data, int32_t* offsets, int32_t* sizes)
{
  int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;
  // filter predicate like '%green%'
  char* pattern  = "green";
  int32_t p_size = 5;
  bool filter    = false;
  for (int i = 0; i < sizes[tid] - p_size + 1; i++) {
    bool match = true;
    for (int j = 0; j < p_size; j++) {
      if (pattern[j] != data[offsets[tid] + i + j]) {
        match = false;
        break;
      }
    }
    if (match) {
      filter = true;
      break;
    }
  }
  if (!filter) return;
  auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  map_ref.insert(this_thread, cuco::pair{keycol[tid], tid});
}

struct multijoin_t {
  int32_t n_idx;
  int32_t s_idx;
  int32_t ps_idx;
};
struct multijoin_t_pol {
  int32_t p_idx;
  int32_t o_idx;
  int32_t l_idx;
};

struct nsps_pol_t {
  int32_t nsps_idx;
  int32_t pol_idx;
};

template <typename ProbeMap>
__global__ void probe_lineitem_size(ProbeMap p_map_ref,
                                    ProbeMap o_map_ref,
                                    int32_t* p_partkey,
                                    int32_t* l_partkey,
                                    int32_t* o_orderkey,
                                    int32_t* l_orderkey,
                                    int32_t* res,
                                    size_t l_size)
{
  int32_t tid = (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= l_size) return;
  // lineitem with part
  auto p_idx = p_map_ref.find(l_partkey[tid]);
  if (p_idx == p_map_ref.end()) return;
  // lineitem with order
  auto o_idx = o_map_ref.find(l_orderkey[tid]);
  if (o_idx == o_map_ref.end()) return;
  atomicAdd(res, 1);
}

template <typename ProbeMap>
__global__ void probe_lineitem(ProbeMap p_map_ref,
                               ProbeMap o_map_ref,
                               int32_t* p_partkey,
                               int32_t* l_partkey,
                               int32_t* o_orderkey,
                               int32_t* l_orderkey,
                               int32_t* res_idx,
                               size_t l_size,
                               multijoin_t_pol* result)
{
  int32_t tid = (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= l_size) return;
  // lineitem with part
  auto p_idx = p_map_ref.find(l_partkey[tid]);
  if (p_idx == p_map_ref.end()) return;
  // lineitem with order
  auto o_idx = o_map_ref.find(l_orderkey[tid]);
  if (o_idx == o_map_ref.end()) return;
  auto idx          = atomicAdd(res_idx, 1);
  result[idx].p_idx = p_idx->second;
  result[idx].o_idx = o_idx->second;
  result[idx].l_idx = tid;
}
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

template <class Ref>
__global__ void for_each_probe_size(Ref ref,
                                    multijoin_t* n_s_ps,
                                    int32_t* s_suppkey,
                                    int32_t* ps_partkey,
                                    int32_t* res_idx,
                                    size_t sz)
{
  int32_t tid = (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= sz) return;
  int64_t key = ((int64_t)s_suppkey[n_s_ps[tid].s_idx]) << 32;
  key |= ((int64_t)ps_partkey[n_s_ps[tid].ps_idx]);
  ref.for_each(key, [&] __device__(auto const slot) {
    auto const [slot_key, slot_value] = slot;
    atomicAdd(res_idx, 1);
  });
}

template <class Ref, typename AggMap>
__global__ void for_each_probe(Ref ref,
                               multijoin_t* n_s_ps,
                               int32_t* s_suppkey,
                               int32_t* ps_partkey,
                               multijoin_t_pol* pol_join,
                               multijoin_t* nsps_join,
                               int8_t* n_name,
                               int32_t* o_orderdate,
                               double* l_extendedprice,
                               double* l_discount,
                               double* ps_supplycost,
                               int64_t* l_quantity,
                               AggMap agg_map,
                               size_t sz)
{
  int32_t tid = (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= sz) return;
  int64_t key = ((int64_t)s_suppkey[n_s_ps[tid].s_idx]) << 32;
  key |= ((int64_t)ps_partkey[n_s_ps[tid].ps_idx]);
  ref.for_each(key, [&] __device__(auto const slot) {
    auto const [slot_key, slot_value] = slot;
    auto pol_idx                      = (int32_t)slot_value;
    auto nsps_idx                     = tid;

    auto l_idx  = pol_join[pol_idx].l_idx;
    auto o_idx  = pol_join[pol_idx].o_idx;
    auto ps_idx = nsps_join[nsps_idx].ps_idx;
    auto n_idx  = nsps_join[nsps_idx].n_idx;

    int8_t nation_name = n_name[n_idx];
    int32_t date       = o_orderdate[o_idx];
    int32_t year       = 1998;
    if (date < 8035)
      year = 1991;
    else if (date < 8401)
      year = 1992;
    else if (date < 8766)
      year = 1993;
    else if (date < 9131)
      year = 1994;
    else if (date < 9496)
      year = 1995;
    else if (date < 9862)
      year = 1996;
    else if (date < 10227)
      year = 1997;
    int64_t gb_k = nation_name;
    gb_k <<= 32;
    gb_k |= year;

    double amount = l_extendedprice[l_idx] * (1 - l_discount[l_idx]) -
                    (ps_supplycost[ps_idx] * l_quantity[l_idx]);
    // printf("%d \n",year);
    if (amount == 0) return;

    auto [gb_slot, is_new_key] = agg_map.insert_and_find(cuco::pair{gb_k, amount});
    if (!is_new_key) {
      auto ref =
        cuda::atomic_ref<typename AggMap::mapped_type, cuda::thread_scope_device>{gb_slot->second};
      ref.fetch_add(amount, cuda::memory_order_relaxed);
    }
  });
}

int main(int argc, const char** argv)
{
  std::cout << std::setprecision(15);

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
  int32_t* d_n_nationkey;
  int32_t* d_s_supplierkey;
  int32_t* d_s_nationkey;
  int32_t* d_ps_suppkey;
  int32_t* d_ps_partkey;
  StringColumn* p_name;
  auto n_nationkey   = read_column_typecasted<int32_t>(nation_table, "n_nationkey");
  auto s_supplierkey = read_column_typecasted<int32_t>(supplier_table, "s_suppkey");
  auto s_nationkey   = read_column_typecasted<int32_t>(supplier_table, "s_nationkey");
  auto ps_suppkey    = read_column_typecasted<int32_t>(partsupp_table, "ps_suppkey");
  auto ps_partkey    = read_column_typecasted<int32_t>(partsupp_table, "ps_partkey");
  p_name             = read_string_column(part_table, "p_name");
  cudaMalloc(&d_n_nationkey, nation_size * sizeof(int32_t));
  cudaMalloc(&d_s_supplierkey, supplier_size * sizeof(int32_t));
  cudaMalloc(&d_s_nationkey, supplier_size * sizeof(int32_t));
  cudaMalloc(&d_ps_suppkey, partsupp_size * sizeof(int32_t));
  cudaMalloc(&d_ps_partkey, partsupp_size * sizeof(int32_t));
  cudaMemcpy(
    d_n_nationkey, n_nationkey.data(), nation_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_s_supplierkey, s_supplierkey.data(), supplier_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_s_nationkey, s_nationkey.data(), supplier_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_ps_suppkey, ps_suppkey.data(), partsupp_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_ps_partkey, ps_partkey.data(), partsupp_size * sizeof(int32_t), cudaMemcpyHostToDevice);

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
  int32_t n_s_ps_joinsize = join_size;
  multijoin_t *n_s_ps_join, *d_n_s_ps_join;
  cudaMalloc(&d_n_s_ps_join, sizeof(multijoin_t) * join_size);
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

  n_s_ps_join = (multijoin_t*)malloc(sizeof(multijoin_t) * join_size);
  cudaMemcpy(n_s_ps_join, d_n_s_ps_join, sizeof(multijoin_t) * join_size, cudaMemcpyDeviceToHost);

  // for (size_t i=0; i<join_size; i++) {
  //   auto n_idx = n_s_ps_join[i].n_idx;
  //   auto s_idx = n_s_ps_join[i].s_idx;
  //   auto ps_idx = n_s_ps_join[i].ps_idx;
  //   std::cout << n_nationkey[n_idx] << " " << s_nationkey[s_idx] << " " << s_supplierkey[s_idx]
  //   << " " <<
  //     ps_suppkey[ps_idx] << "\n";
  // }

  auto p_partkey_map =
    cuco::static_map{part_size * 2,
                     cuco::empty_key{(int32_t)-1},
                     cuco::empty_value{(int32_t)-1},
                     thrust::equal_to<int32_t>{},
                     cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int32_t>>()};
  auto o_orderkey_map =
    cuco::static_map{orders_size * 2,
                     cuco::empty_key{(int32_t)-1},
                     cuco::empty_value{(int32_t)-1},
                     thrust::equal_to<int32_t>{},
                     cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int32_t>>()};
  int32_t* d_p_partkey;   // build hash on this
  int32_t* d_o_orderkey;  // build hash on this
  int32_t* d_l_partkey;
  int32_t* d_l_orderkey;

  auto p_partkey  = read_column_typecasted<int32_t>(part_table, "p_partkey");
  auto o_orderkey = read_column_typecasted<int32_t>(orders_table, "o_orderkey");
  auto l_partkey  = read_column_typecasted<int32_t>(lineitem_table, "l_partkey");
  auto l_orderkey = read_column_typecasted<int32_t>(lineitem_table, "l_orderkey");

  char* d_char_data;
  int32_t *d_offsets, *d_sizes;
  int tot_size = p_name->offsets[part_size - 1] + p_name->sizes[part_size - 1];
  cudaMalloc(&d_char_data, sizeof(char) * tot_size);
  cudaMalloc(&d_offsets, sizeof(int32_t) * part_size);
  cudaMalloc(&d_sizes, sizeof(int32_t) * part_size);
  cudaMemcpy(d_char_data, p_name->data, sizeof(char) * tot_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, p_name->offsets, sizeof(int32_t) * part_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sizes, p_name->sizes, sizeof(int32_t) * part_size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_p_partkey, sizeof(int32_t) * part_size);
  cudaMalloc(&d_o_orderkey, sizeof(int32_t) * orders_size);
  cudaMalloc(&d_l_partkey, sizeof(int32_t) * lineitem_size);
  cudaMalloc(&d_l_orderkey, sizeof(int32_t) * lineitem_size);

  cudaMemcpy(d_p_partkey, p_partkey.data(), sizeof(int32_t) * part_size, cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_o_orderkey, o_orderkey.data(), sizeof(int32_t) * orders_size, cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_l_partkey, l_partkey.data(), sizeof(int32_t) * lineitem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_l_orderkey, l_orderkey.data(), sizeof(int32_t) * lineitem_size, cudaMemcpyHostToDevice);

  build_hash_primary_key_partname_filter<<<getGridSize(part_size, TB), TB>>>(
    p_partkey_map.ref(cuco::insert), d_p_partkey, part_size, d_char_data, d_offsets, d_sizes);
  // std::cout << "p_partkey map size: " << p_partkey_map.size() << "\n";
  CUDACHKERR();
  build_hash_primary_key<<<getGridSize(orders_size, TB), TB>>>(
    o_orderkey_map.ref(cuco::insert), d_o_orderkey, orders_size);
  // std::cout << "o_orderkey map size: " << o_orderkey_map.size() << "\n";
  CUDACHKERR();
  cudaMemset(d_join_size, 0, sizeof(int32_t));
  CUDACHKERR();
  probe_lineitem_size<<<getGridSize(lineitem_size, TB), TB>>>(p_partkey_map.ref(cuco::find),
                                                              o_orderkey_map.ref(cuco::find),
                                                              d_p_partkey,
                                                              d_l_partkey,
                                                              d_o_orderkey,
                                                              d_l_orderkey,
                                                              d_join_size,
                                                              lineitem_size);
  multijoin_t_pol* d_p_o_l_join;
  cudaMemcpy(&join_size, d_join_size, sizeof(int32_t), cudaMemcpyDeviceToHost);
  int32_t p_o_l_joinsize = join_size;
  cudaMemset(d_join_size, 0, sizeof(int32_t));
  cudaMalloc(&d_p_o_l_join, sizeof(multijoin_t_pol) * join_size);
  probe_lineitem<<<getGridSize(lineitem_size, TB), TB>>>(p_partkey_map.ref(cuco::find),
                                                         o_orderkey_map.ref(cuco::find),
                                                         d_p_partkey,
                                                         d_l_partkey,
                                                         d_o_orderkey,
                                                         d_l_orderkey,
                                                         d_join_size,
                                                         lineitem_size,
                                                         d_p_o_l_join);

  multijoin_t_pol* pol_join = (multijoin_t_pol*)malloc(sizeof(multijoin_t_pol) * join_size);
  cudaMemcpy(pol_join, d_p_o_l_join, sizeof(multijoin_t_pol) * join_size, cudaMemcpyDeviceToHost);
  // std::map<std::pair<int32_t, int32_t>, int32_t> ps_map;
  auto l_suppkey = read_column_typecasted<int32_t>(lineitem_table, "l_suppkey");
  // for (size_t i=0; i<join_size; i++ ){
  //   ps_map[std::make_pair(l_suppkey[pol_join[i].l_idx], l_partkey[pol_join[i].l_idx])]++;
  // }
  // for (auto &p: ps_map) {
  //   if (p.second > 1) {
  //     std::cout << "suppkey: " << p.first.first << ", partkey: " << p.first.second << ", count: "
  //     << p.second << "\n";
  //   }
  // }
  // std::cout << "now working on l_pskey_map\n";
  // std::cout << "part order lineitem join size: " << p_o_l_joinsize << "\n";
  // std::cout << "nation supplier partsupp join size: " << n_s_ps_joinsize << "\n";
  auto l_pskey_map = cuco::experimental::static_multimap{
    p_o_l_joinsize * 2,
    cuco::empty_key{(int64_t)-1},
    cuco::empty_value{(int64_t)-1},
    {},
    cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int64_t>>{},
    {},
    cuco::storage<2>{}};

  thrust::host_vector<cuco::pair<int64_t, int64_t>> h_ls_lp_dict(p_o_l_joinsize);
  for (int i = 0; i < p_o_l_joinsize; i++) {
    int32_t l_idx   = pol_join[i].l_idx;
    int64_t ls      = ((int64_t)l_suppkey[l_idx]) << 32;
    int64_t lp      = ((int64_t)l_partkey[l_idx]);
    h_ls_lp_dict[i] = cuco::make_pair(ls | lp, (int64_t)i);
  }
  thrust::device_vector<cuco::pair<int64_t, int64_t>> d_ls_lp_dict = h_ls_lp_dict;
  l_pskey_map.insert(d_ls_lp_dict.begin(), d_ls_lp_dict.end());

  cudaMemset(d_join_size, 0, sizeof(int32_t));
  for_each_probe_size<<<getGridSize(n_s_ps_joinsize, TB), TB>>>(l_pskey_map.ref(cuco::for_each),
                                                                d_n_s_ps_join,
                                                                d_s_supplierkey,
                                                                d_ps_partkey,
                                                                d_join_size,
                                                                n_s_ps_joinsize);
  cudaMemcpy(&join_size, d_join_size, sizeof(int32_t), cudaMemcpyDeviceToHost);
  int32_t nsps_pol_joinsize = join_size;
  // std::cout << "all join size: " << nsps_pol_joinsize << "\n";
  nsps_pol_t* d_nsps_pol;
  cudaMalloc(&d_nsps_pol, sizeof(nsps_pol_t) * nsps_pol_joinsize);
  auto agg_map =
    cuco::static_map{nsps_pol_joinsize * 2,
                     cuco::empty_key{(int64_t)-1},
                     cuco::empty_value<double>{0.},
                     thrust::equal_to<int64_t>{},
                     cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int64_t>>()};
  StringDictEncodedColumn* n_name = read_string_dict_encoded_column(nation_table, "n_name");
  auto o_orderdate                = read_column<int32_t>(orders_table, "o_orderdate");
  auto ps_supplycost              = read_column<double>(partsupp_table, "ps_supplycost");
  auto l_ep                       = read_column<double>(lineitem_table, "l_extendedprice");
  auto l_disc                     = read_column<double>(lineitem_table, "l_discount");
  auto l_qty                      = read_column<int64_t>(lineitem_table, "l_quantity");

  int8_t* d_n_name;
  int32_t* d_o_orderdate;
  int64_t* d_l_quantity;
  double *d_l_extendedprice, *d_l_discount, *d_ps_supplycost;
  cudaMalloc(&d_n_name, sizeof(int8_t) * nation_size);
  cudaMalloc(&d_o_orderdate, sizeof(int32_t) * orders_size);
  cudaMalloc(&d_l_quantity, sizeof(int64_t) * lineitem_size);
  cudaMalloc(&d_l_extendedprice, sizeof(double) * lineitem_size);
  cudaMalloc(&d_l_discount, sizeof(double) * lineitem_size);
  cudaMalloc(&d_ps_supplycost, sizeof(double) * partsupp_size);
  cudaMemcpy(d_n_name, n_name->column, sizeof(int8_t) * nation_size, cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_o_orderdate, o_orderdate.data(), sizeof(int32_t) * orders_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_l_quantity, l_qty.data(), sizeof(int64_t) * lineitem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_l_extendedprice, l_ep.data(), sizeof(double) * lineitem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_l_discount, l_disc.data(), sizeof(double) * lineitem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(
    d_ps_supplycost, ps_supplycost.data(), sizeof(double) * partsupp_size, cudaMemcpyHostToDevice);

  for_each_probe<<<getGridSize(n_s_ps_joinsize, TB), TB>>>(l_pskey_map.ref(cuco::for_each),
                                                           d_n_s_ps_join,
                                                           d_s_supplierkey,
                                                           d_ps_partkey,
                                                           d_p_o_l_join,
                                                           d_n_s_ps_join,
                                                           d_n_name,
                                                           d_o_orderdate,
                                                           d_l_extendedprice,
                                                           d_l_discount,
                                                           d_ps_supplycost,
                                                           d_l_quantity,
                                                           agg_map.ref(cuco::insert_and_find),
                                                           n_s_ps_joinsize);
  auto agg_map_size = agg_map.size();
  // std::cout << "end of join probes\n";
  // std::cout << "Final grouped by count: " << agg_map_size << "\n";
  // nsps_pol_t* nsps_pol = (nsps_pol_t*)malloc(sizeof(nsps_pol_t)*nsps_pol_joinsize);
  // cudaMemcpy(nsps_pol, d_nsps_pol, sizeof(nsps_pol_t)*nsps_pol_joinsize, cudaMemcpyDeviceToHost);

  thrust::device_vector<double> sum_profit(agg_map_size);
  thrust::device_vector<int64_t> gb_keys(agg_map_size);
  agg_map.retrieve_all(gb_keys.begin(), sum_profit.begin());

  for (int i = 0; i < agg_map_size; i++) {
    int32_t year = (0xFFFFFFFF & gb_keys[i]);
    int8_t name  = (gb_keys[i] >> 32);
    for (auto e : n_name->dict) {
      if (e.second == name) {
        std::cout << e.first << "\t\t\t";
        break;
      }
    }
    std::cout << year << " " << sum_profit[i] << "\n";
  }
}