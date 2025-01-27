#include <iostream>
#include "utils.h"

/*
select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty
from
        lineitem
where
        l_shipdate <= date '1998-12-01' - interval '90' day
group by
        l_returnflag,
        l_linestatus

BlockLoad - l_shipdate
BlockPred
BlockLoad l_quantity
BlockLoad l_returnflag
BlockLoad l_linestatus
BlockAggregate hashfn (l_rf * 3 + l_ls), atomicAdd, buffer[6]
BlockStore
*/


#define ITEMS_PER_THREAD 8
#define TB_SIZE 32
#define TILE_SIZE ITEMS_PER_THREAD*TB_SIZE

__global__ void aggregate_crystal(
  int8_t* l_returnflag,
  int8_t* l_linestatus,
  int32_t* l_shipdate,
  double* l_extendedprice,
  double* res,
  int32_t predicate_date,
  int32_t ls_size,
  size_t lineitem_size
) {
  int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= lineitem_size) return;

  int32_t items[ITEMS_PER_THREAD]; // assuming this is loaded into the registers
  int8_t items2[ITEMS_PER_THREAD];
  int8_t items3[ITEMS_PER_THREAD];
  double items4[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  
  // do a block load
  for (int i=0; i < ITEMS_PER_THREAD && (tid + i) < lineitem_size; i++) {
    items[i] = l_shipdate[tid + i];
  }
  // blockpred
  for (int i=0; i < ITEMS_PER_THREAD && (tid + i) < lineitem_size; i++) {
    selection_flags[i] = items[i] <= predicate_date;
  }
  // blockload - l_qty, l_returnflag l_linestatus
  for (int i=0; i < ITEMS_PER_THREAD && (tid + i) < lineitem_size; i++) {
    items4[i] = l_extendedprice[tid + i];
    items2[i] = l_linestatus[tid + i];
    items3[i] = l_returnflag[tid + i];
  }
  // now run an aggregation
  __syncthreads();
  __shared__ double shout[6];
  for (int i=0; i<6; i++) shout[i] = 0;
  __syncthreads();
  // blockAggregation
  for (int i=0; i < ITEMS_PER_THREAD && (tid + i) < lineitem_size; i++) {
    if (selection_flags[i])
      atomicAdd(&(shout[items3[i] * ls_size + items2[i]]), items4[i]);
  }
  __syncthreads();
  // blockStore
  // printf("%d\n", threadIdx.x);
  if (threadIdx.x == 0) {
    for (int i=0; i<6; i++) {
      atomicAdd(&(res[i]), (double)shout[i]);
    }
  }
}

__global__ void aggregate(
  int8_t* l_returnflag,
  int8_t* l_linestatus,
  int32_t* l_shipdate,
  double* l_extendedprice,
  double* res,
  int32_t predicate_date,
  int32_t ls_size,
  size_t lineitem_size
) {
  int tid = ITEMS_PER_THREAD * (threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= lineitem_size) return;

  // declare shared memory
  __shared__ double shout[6];
  __shared__ int winner;
  for (int i=0; i<6; i++) shout[i] = 0;
  __syncthreads();

  #pragma unroll
  for (int j=0; j<ITEMS_PER_THREAD && (tid + j) < lineitem_size; j++) {
    if (l_shipdate[tid + j] > predicate_date) continue;
    int rf = l_returnflag[tid+j], ls = l_linestatus[tid+j];
    int agg_key = rf * ls_size + ls; // this is the hash function 
    atomicAdd(&(shout[agg_key]), l_extendedprice[tid+j]);
  }
  // printf("agg_key: %d\n", agg_key);
  winner = threadIdx.x;
  __syncthreads();
  if (threadIdx.x == winner) {
    for (int i=0; i<6; i++) {
      atomicAdd(&(res[i]), (double)shout[i]);
    }
  }
}

int main(int argc, const char** argv) {

  std::string dir = getDataDir(argv, argc);
  std::string lineitem_file = dir + "lineitem.parquet";

  auto lineitem_table = getArrowTable(lineitem_file);
  size_t lineitem_size = lineitem_table->num_rows();

  StringDictEncodedColumn* l_returnflag =
    read_string_dict_encoded_column(lineitem_table, "l_returnflag");
  StringDictEncodedColumn* l_linestatus =
    read_string_dict_encoded_column(lineitem_table, "l_linestatus");

  auto l_shipdate     = read_column<int32_t>(lineitem_table, "l_shipdate");
  auto l_extendedprice = read_column<double>(lineitem_table, "l_extendedprice");

  int8_t *d_l_returnflag, *d_l_linestatus;
  int32_t *d_l_shipdate;
  double *d_l_extendedprice;
  cudaMalloc(&d_l_returnflag, lineitem_size*sizeof(int8_t));
  cudaMemcpy(d_l_returnflag, l_returnflag->column, lineitem_size*sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_linestatus, lineitem_size * sizeof(int8_t));
  cudaMemcpy(d_l_linestatus, l_linestatus->column, lineitem_size * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_shipdate, lineitem_size * sizeof(int32_t));
  cudaMemcpy(d_l_shipdate, l_shipdate.data(), lineitem_size * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_extendedprice, lineitem_size * sizeof(double));
  cudaMemcpy(d_l_extendedprice, l_extendedprice.data(), lineitem_size * sizeof(double), cudaMemcpyHostToDevice);


  size_t groups_cardinality = l_returnflag->dict.size() * l_linestatus->dict.size();
  double *d_res;
  cudaMalloc(&d_res, sizeof(double)*groups_cardinality);
  cudaMemset(d_res, 0., sizeof(double)*groups_cardinality);

  int TB = TILE_SIZE / ITEMS_PER_THREAD;
  int32_t predicate_date = 10471; 
  int thread_blocks = getGridSize(lineitem_size, TB * ITEMS_PER_THREAD);
  aggregate_crystal<<<thread_blocks, TB>>>(
    d_l_returnflag, 
    d_l_linestatus,
    d_l_shipdate,
    d_l_extendedprice,
    d_res,
    predicate_date,
    l_linestatus->dict.size(),
    lineitem_size
  );
  cudaDeviceSynchronize();
  CUDACHKERR();
  double *res = (double*) malloc(sizeof(double)*groups_cardinality);
  cudaMemcpy(res, d_res, sizeof(double)*groups_cardinality, cudaMemcpyDeviceToHost);

  std::cout << "aggregate_crystal: \n";
  for (size_t i=0; i<groups_cardinality; i++) {
    if (res[i]!=0.) {
      std::cout << res[i] << std::endl;
    }
  }
  cudaMemset(d_res, 0., sizeof(double)*groups_cardinality);
  aggregate<<<thread_blocks, TB>>>(
    d_l_returnflag, 
    d_l_linestatus,
    d_l_shipdate,
    d_l_extendedprice,
    d_res,
    predicate_date,
    l_linestatus->dict.size(),
    lineitem_size
  );
  cudaDeviceSynchronize();
  CUDACHKERR();
  cudaMemcpy(res, d_res, sizeof(double)*groups_cardinality, cudaMemcpyDeviceToHost);

  std::cout << "aggregate : \n";
  for (size_t i=0; i<groups_cardinality; i++) {
    if (res[i]!=0.) {
      std::cout << res[i] << std::endl;
    }
  }
}