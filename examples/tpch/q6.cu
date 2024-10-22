/*
-- TPC-H Query 6

select
        sum(l_extendedprice * l_discount) as revenue
from
        lineitem
where
        l_shipdate >= date '1994-01-01'
        and l_shipdate < date '1995-01-01'
        and l_discount between 0.06 - 0.01 and 0.06 + 0.01
        and l_quantity < 24
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

__global__ void aggregate_revenue(int32_t* l_shipdate,
                                  int64_t* l_quantity,
                                  double* l_discount,
                                  double* l_extendedprice,
                                  size_t lineitem_size,
                                  double* result)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= lineitem_size) return;
  // filter with date
  if (l_shipdate[tid] < 8766 || l_shipdate[tid] >= 9131) return;
  // filter with discount
  if (l_discount[tid] < (0.05) || l_discount[tid] > (0.07)) return;
  // filter with quantity
  if (l_quantity[tid] >= 24) return;

  double rev = l_extendedprice[tid] * l_discount[tid];
  atomicAdd(result, rev);
}

double cpu_aggregate_revenue(int32_t* l_shipdate,
                             int64_t* l_quantity,
                             double* l_discount,
                             double* l_extendedprice,
                             size_t lineitem_size)
{
  double res = 0;
  for (size_t i = 0; i < lineitem_size; i++) {
    // filter with date
    if (l_shipdate[i] < 8766 || l_shipdate[i] >= 9131) continue;
    // filter with discount
    if (l_discount[i] < (0.05) || l_discount[i] > (0.07)) continue;
    // filter with quantity
    if (l_quantity[i] >= 24) continue;
    res += (l_extendedprice[i] * l_discount[i]);
  }
  return res;
}
int main(int argc, const char** argv)
{
  std::cout << std::setprecision(10);

  std::string dbDir         = getDataDir(argv, argc);
  std::string lineitem_file = dbDir + "lineitem.parquet";

  auto lineitem_table  = getArrowTable(lineitem_file);
  size_t lineitem_size = lineitem_table->num_rows();

  int32_t* l_shipdate     = read_column<int32_t>(lineitem_table, "l_shipdate");
  int64_t* l_quantity     = read_column<int64_t>(lineitem_table, "l_quantity");
  double* l_discount      = read_column<double>(lineitem_table, "l_discount");
  double* l_extendedprice = read_column<double>(lineitem_table, "l_extendedprice");

  int32_t* d_l_shipdate;
  cudaMalloc(&d_l_shipdate, sizeof(int32_t) * lineitem_size);
  cudaMemcpy(d_l_shipdate, l_shipdate, sizeof(int32_t) * lineitem_size, cudaMemcpyHostToDevice);

  double *d_l_extendedprice, *d_l_discount;
  cudaMalloc(&d_l_extendedprice, sizeof(double) * lineitem_size);
  cudaMemcpy(
    d_l_extendedprice, l_extendedprice, sizeof(double) * lineitem_size, cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_discount, sizeof(double) * lineitem_size);
  cudaMemcpy(d_l_discount, l_discount, sizeof(double) * lineitem_size, cudaMemcpyHostToDevice);

  int64_t* d_l_quantity;
  cudaMalloc(&d_l_quantity, sizeof(int64_t) * lineitem_size);
  cudaMemcpy(d_l_quantity, l_quantity, sizeof(int64_t) * lineitem_size, cudaMemcpyHostToDevice);

  size_t TB = 32;
  double* d_res;
  cudaMalloc(&d_res, sizeof(double));
  cudaMemset(d_res, 0., sizeof(double));

  aggregate_revenue<<<std::ceil((float)lineitem_size / (float)TB), TB>>>(
    d_l_shipdate, d_l_quantity, d_l_discount, d_l_extendedprice, lineitem_size, d_res);

  double res;
  cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << "revenue\t\n";
  std::cout << res << "\n";
}