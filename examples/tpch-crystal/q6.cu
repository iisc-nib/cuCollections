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


#define ITEMS_PER_THREAD 2
#define TB_SIZE 512
#define TILE_SIZE ITEMS_PER_THREAD*TB_SIZE


__global__ void aggregate_revenue(int32_t* l_shipdate,
                                  int64_t* l_quantity,
                                  double* l_discount,
                                  double* l_extendedprice,
                                  size_t lineitem_size,
                                  double* result)
{
  int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= lineitem_size) return;

  // declare shared memory
  __shared__ double shout;
  __shared__ int winner;
  shout = 0;
  __syncthreads();
  for (int j=0; j<ITEMS_PER_THREAD && (tid + j) < lineitem_size; j++) {
    // filter with date
    if (l_shipdate[tid + j] < 8766 || l_shipdate[tid + j] >= 9131) continue;
    // filter with discount
    if (l_discount[tid + j] < (0.05) || l_discount[tid + j] > (0.07)) continue;
    // filter with quantity
    if (l_quantity[tid + j] >= 24) continue;

    double rev = l_extendedprice[tid + j] * l_discount[tid + j];
    atomicAdd(&shout, rev);
  }

  winner = threadIdx.x;
  __syncthreads();
  if (threadIdx.x == winner) {
    atomicAdd(result, shout);
  }
}

int main(int argc, const char** argv)
{
  std::cout << std::setprecision(10);

  std::string dbDir         = getDataDir(argv, argc);
  std::string lineitem_file = dbDir + "lineitem.parquet";

  auto lineitem_table  = getArrowTable(lineitem_file);
  size_t lineitem_size = lineitem_table->num_rows();

  auto l_shipdate      = read_column<int32_t>(lineitem_table, "l_shipdate");
  auto l_quantity      = read_column<int64_t>(lineitem_table, "l_quantity");
  auto l_discount      = read_column<double>(lineitem_table, "l_discount");
  auto l_extendedprice = read_column<double>(lineitem_table, "l_extendedprice");

  int32_t* d_l_shipdate;
  cudaMalloc(&d_l_shipdate, sizeof(int32_t) * lineitem_size);
  cudaMemcpy(
    d_l_shipdate, l_shipdate.data(), sizeof(int32_t) * lineitem_size, cudaMemcpyHostToDevice);

  double *d_l_extendedprice, *d_l_discount;
  cudaMalloc(&d_l_extendedprice, sizeof(double) * lineitem_size);
  cudaMemcpy(d_l_extendedprice,
             l_extendedprice.data(),
             sizeof(double) * lineitem_size,
             cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_discount, sizeof(double) * lineitem_size);
  cudaMemcpy(
    d_l_discount, l_discount.data(), sizeof(double) * lineitem_size, cudaMemcpyHostToDevice);

  int64_t* d_l_quantity;
  cudaMalloc(&d_l_quantity, sizeof(int64_t) * lineitem_size);
  cudaMemcpy(
    d_l_quantity, l_quantity.data(), sizeof(int64_t) * lineitem_size, cudaMemcpyHostToDevice);

  int TB = TILE_SIZE / ITEMS_PER_THREAD;
  int thread_blocks = getGridSize(lineitem_size, TB * ITEMS_PER_THREAD);
  double* d_res;
  cudaMalloc(&d_res, sizeof(double));
  cudaMemset(d_res, 0., sizeof(double));

  aggregate_revenue<<<thread_blocks, TB>>>(
    d_l_shipdate, d_l_quantity, d_l_discount, d_l_extendedprice, lineitem_size, d_res);

  double res;
  cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << "revenue\t\n";
  std::cout << res << "\n";
}