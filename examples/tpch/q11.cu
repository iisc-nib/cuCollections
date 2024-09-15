#include<iostream>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <parquet/arrow/reader.h>
#include <cooperative_groups.h>
#include <cuco/static_map.cuh>
#include <cuda.h>
#include "utils.h"

/**
 * c_custkey: int64
c_name: large_string
c_address: large_string
c_nationkey: int64
c_phone: large_string
c_acctbal: double
c_mktsegment: large_string
c_comment: large_string

l_orderkey: int64
l_partkey: int64
l_suppkey: int64
l_linenumber: int64
l_quantity: int64
l_extendedprice: double
l_discount: double
l_tax: double
l_returnflag: large_string
l_linestatus: large_string
l_shipdate: date32[day]
l_commitdate: date32[day]
l_receiptdate: date32[day]
l_shipinstruct: large_string
l_shipmode: large_string


o_orderkey: int64
o_custkey: int64
o_orderstatus: large_string
o_totalprice: double
o_orderdate: date32[day]
o_orderpriority: large_string
o_clerk: large_string
o_shippriority: int64
o_comment: large_string
 */
CUCO_DECLARE_BITWISE_COMPARABLE(double);

struct agg_key_type {
  int64_t l_orderkey;
  int32_t o_orderdate;
  int64_t o_shippriority;
  __host__ __device__ agg_key_type() {}
  __host__ __device__ agg_key_type(int32_t x): l_orderkey(x), o_orderdate(x), o_shippriority(x) {}
};
#define TILE_SIZE 1
namespace cg = cooperative_groups;
template <typename Map>
__global__ void build_hash_order(Map map_ref, int64_t* column, int32_t* filter, size_t column_size) {
  int tid = (threadIdx.x + blockIdx.x*blockDim.x) ;
  auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  
  if (tid >= column_size) return;
  if (filter[tid] >= 9204) return;
  
  map_ref.insert(this_thread, cuco::pair{column[tid], tid});
}

template <typename Map>
__global__ void build_hash_customer(Map map_ref, int64_t* column, int32_t* filter, size_t column_size, int32_t building_code) {
  int tid = (threadIdx.x + blockIdx.x*blockDim.x) / TILE_SIZE;
  auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  
  if (tid >= column_size) return;
  if (filter[tid] != building_code) return;
  map_ref.insert(this_thread, cuco::pair{column[tid], tid});
}

template <typename Map, typename AggMap>
__global__ void probe(Map order_hash,
                      Map customer_hash,
                      int32_t* l_shipdate,
                      int64_t* l_orderkey,
                      int64_t* o_custkey,
                      int32_t* o_orderdate,
                      int64_t* o_shippriority,
                      double* l_extendedprice,
                      double* l_discount,
                      size_t l_size, size_t o_size,
                      double* result, AggMap agg_map) {
  int tid = (threadIdx.x + blockIdx.x*blockDim.x) / TILE_SIZE;
  if (tid >= l_size) return;

  // filter line item by shipdate
  if (l_shipdate[tid] <= 9204) return; 
  auto order_idx = order_hash.find(l_orderkey[tid]);
  if (order_idx == order_hash.end()) return;
  auto cust_idx = customer_hash.find(o_custkey[order_idx->second]);
  if (cust_idx == customer_hash.end()) return;
  //printf("%f\n", l_extendedprice[tid]); 
  // result[l_orderkey[tid]%l_size] += l_extendedprice[tid];
  //atomicAdd(result+(l_orderkey[tid]%l_size), l_extendedprice[tid]);
  //group by o_orderdate, l_orderkey, o_shippriority
  // prepare the packed key
  int64_t aggkey = 0;
  aggkey |= l_orderkey[tid];
  aggkey |= (o_orderdate[order_idx->second] << 32);
  double res = l_extendedprice[tid]*(1.0f - l_discount[tid]);
  auto [slot, is_new_key]= agg_map.insert_and_find(cuco::pair{aggkey, res});
  if (!is_new_key) {
    auto ref = cuda::atomic_ref<typename AggMap::mapped_type, cuda::thread_scope_device>{slot->second};
    ref.fetch_add(res, cuda::memory_order_relaxed);
  }
}

struct floatbitwise : cuco::is_bitwise_comparable<double> {
  double val;
  floatbitwise(double v) : val(v){}
};
int main() {
  std::string dbDir = "/media/ajayakar/space/src/tpch/data/tables/scale-10.0/";
  std::string lineitem_file = dbDir + "lineitem.parquet";
  std::string orders_file = dbDir + "orders.parquet";
  std::string customer_file = dbDir + "customer.parquet";

  auto const agg_empty_key_sentinel = agg_key_type{-1};


  auto lineitem_table = getArrowTable(lineitem_file);
  size_t lineitem_size = lineitem_table->num_rows();

  auto orders_table = getArrowTable(orders_file);
  size_t orders_size = orders_table->num_rows();

  auto customer_table = getArrowTable(customer_file);
  size_t customer_size = customer_table->num_rows();
  
  int64_t* l_orderkey = read_column<int64_t>(lineitem_table, "l_orderkey");
  int32_t* l_shipdate = read_column<int32_t>(lineitem_table, "l_shipdate");
  double* l_extendedprice = read_column<double>(lineitem_table, "l_extendedprice");
  double* l_discount = read_column<double>(lineitem_table, "l_discount");

  int64_t* o_orderkey = read_column<int64_t>(orders_table, "o_orderkey");
  int64_t* o_custkey = read_column<int64_t>(orders_table, "o_custkey");
  int32_t* o_orderdate = read_column<int32_t>(orders_table, "o_orderdate");
  int64_t* o_shippriority = read_column<int64_t>(orders_table, "o_shippriority");

  int32_t* c_custkey = read_column<int32_t>(customer_table, "c_custkey");
  StringDictEncodedColumn* c_mktsegment = read_string_dict_encoded_column(customer_table, "c_mktsegment");
  int32_t building_code = c_mktsegment->dict["BUILDING"];
  //for (auto e: c_mktsegment->dict) std::cout << e.first << " -> " << e.second << std::endl;

  auto o_orderkey_map = cuco::static_map{
    orders_size*2,
    cuco::empty_key{(int64_t)-1},
    cuco::empty_value{(int64_t)-1},
    thrust::equal_to<int64_t>{},
    cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int64_t>>()
  };  
  auto c_custkey_map = cuco::static_map{
    customer_size*2,
    cuco::empty_key{(int64_t)-1},
    cuco::empty_value{(int64_t)-1},
    thrust::equal_to<int64_t>{},
    cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int64_t>>()
  };  
  auto agg_map = cuco::static_map{customer_size*2,
    cuco::empty_key{(int64_t)-1},
    cuco::empty_value{(double)-1},
    thrust::equal_to<int64_t>{},
    cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int64_t>>()
  };
  int64_t *d_o_orderkey, *d_c_custkey, *d_l_orderkey, *d_o_custkey, *d_o_shippriority; 
  int32_t *d_o_orderdate, *d_c_mktsegment, *d_l_shipdate, *d_l_orderdate;
  double *d_l_extendedprice, *d_l_discount;
  cudaMalloc(&d_o_orderkey, orders_size*sizeof(int64_t));
  cudaMalloc(&d_c_custkey, customer_size*sizeof(int64_t));
  cudaMalloc(&d_l_orderkey, lineitem_size*sizeof(int64_t));
  cudaMalloc(&d_o_custkey, orders_size*sizeof(int64_t));
  cudaMalloc(&d_o_shippriority, orders_size*sizeof(int64_t));
  cudaMalloc(&d_o_orderdate, orders_size*sizeof(int32_t));
  cudaMalloc(&d_c_mktsegment, customer_size*sizeof(int32_t));
  cudaMalloc(&d_l_shipdate, lineitem_size*sizeof(int32_t));
  cudaMalloc(&d_l_extendedprice, lineitem_size*sizeof(double));
  cudaMalloc(&d_l_discount, lineitem_size*sizeof(double));

  cudaMemcpy(d_o_orderkey, o_orderkey, orders_size*sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c_custkey, c_custkey, customer_size*sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_l_orderkey, l_orderkey, lineitem_size*sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_o_custkey, o_custkey, orders_size*sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_o_shippriority, o_shippriority, orders_size*sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_o_orderdate, o_orderdate, orders_size*sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c_mktsegment, c_mktsegment->column, customer_size*sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_l_shipdate, l_shipdate, lineitem_size*sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_l_extendedprice, l_extendedprice, lineitem_size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_l_discount, l_discount, lineitem_size*sizeof(double), cudaMemcpyHostToDevice);

  int threadBlockSize = 1024;
  build_hash_order<<<getGridSize(orders_size, threadBlockSize), threadBlockSize>>> (
    o_orderkey_map.ref(cuco::insert),
    d_o_orderkey,
    d_o_orderdate,
    orders_size
  );
  build_hash_customer<<<getGridSize(customer_size, threadBlockSize), threadBlockSize>>> (
    c_custkey_map.ref(cuco::insert),
    d_c_custkey,
    d_c_mktsegment,
    customer_size,
    building_code
  );
  size_t res_size = lineitem_size;
  double* res = new double[res_size];
  for (int i=0; i<res_size; i++) res[i] = 0;
  double *d_res;
  cudaMalloc(&d_res, lineitem_size*sizeof(double));
  cudaMemcpy(d_res, res, lineitem_size*sizeof(double), cudaMemcpyHostToDevice);
  probe<<<getGridSize(lineitem_size, threadBlockSize), threadBlockSize>>> (
    o_orderkey_map.ref(cuco::find),
    c_custkey_map.ref(cuco::find),
    d_l_shipdate,
    d_l_orderkey,
    d_o_custkey,
    d_o_orderdate,
    d_o_shippriority,
    d_l_extendedprice,
    d_l_discount,
    lineitem_size, 
    orders_size,
    d_res, 
    agg_map.ref(cuco::insert_and_find));
  int agg_map_size = agg_map.size();
  std::cout << agg_map.capacity() << "| map size: " << agg_map_size << std::endl;
  thrust::device_vector<double> result_rev(agg_map_size);
  thrust::device_vector<int64_t> result_keys(agg_map_size);
  agg_map.retrieve_all(result_keys.begin(), result_rev.begin());

  std::vector<double> rev;
  for (int i=0; i<agg_map_size; i++) {
  //   std::cout << result_keys[i] << " : " << result_rev[i] << std::endl;
    rev.push_back(result_rev[i]);
  }
  std::sort(rev.rbegin(), rev.rend());
  std::cout << "Printing the first 10 sorted revenues:\n";
  for (int i=0; i<10; i++) {
    std::cout << rev[i] << std::endl;
  }
  // cudaMemcpy(res, d_res, lineitem_size*sizeof(double),cudaMemcpyDeviceToHost);
  // std::vector<double> r;
  // for (int i=0; i<lineitem_size; i++) {
  //   if (res[i]!=0) r.push_back(res[i]);
  // }
  // std::sort(r.rbegin(), r.rend());
  // for (int i=0; i<10; i++) {
  //   std::cout << r[i] ;
  //   std::cout << "\t" << l_extendedprice[i] << std::endl;
  // }
  // print out the max and min values of o_orderdate, o_shippriority and l_orderkey to see if it can be fit as a composite key of 32 bit
  // int32_t lok_min = INT_MAX, lok_max = INT_MIN;
  // for (int i=0; i<orders_size; i++) {
  //   lok_min = std::min(lok_min, o_orderdate[i]);
  //   lok_max = std::max(lok_max, o_orderdate[i]);
  // }
  // std::cout << "l_orderkey: min=" << lok_min << " max=" << lok_max << std::endl;
}
