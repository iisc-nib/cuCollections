#include <iostream>
#include "utils.h"



#define ITEMS_PER_THREAD 8
#define TB_SIZE 32
#define TILE_SIZE ITEMS_PER_THREAD*TB_SIZE

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
    int agg_key = rf * ls_size + ls;
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

  int32_t* l_shipdate     = read_column<int32_t>(lineitem_table, "l_shipdate");
  double* l_extendedprice = read_column<double>(lineitem_table, "l_extendedprice");

  int8_t *d_l_returnflag, *d_l_linestatus;
  int32_t *d_l_shipdate;
  double *d_l_extendedprice;
  allocate_to_device<int8_t>(d_l_returnflag, l_returnflag->column, lineitem_size);
  allocate_to_device<int8_t>(d_l_linestatus, l_linestatus->column, lineitem_size);
  allocate_to_device<int32_t>(d_l_shipdate, l_shipdate, lineitem_size);
  allocate_to_device<double>(d_l_extendedprice, l_extendedprice, lineitem_size);

  size_t groups_cardinality = l_returnflag->dict.size() * l_linestatus->dict.size();
  double *d_res;
  cudaMalloc(&d_res, sizeof(double)*groups_cardinality);
  cudaMemset(d_res, 0., sizeof(double)*groups_cardinality);

  int TB = TILE_SIZE / ITEMS_PER_THREAD;
  int32_t predicate_date = 10471; 
  int thread_blocks = getGridSize(lineitem_size, TB * ITEMS_PER_THREAD);
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
  double *res = (double*) malloc(sizeof(double)*groups_cardinality);
  cudaMemcpy(res, d_res, sizeof(double)*groups_cardinality, cudaMemcpyDeviceToHost);

  std::cout << "RES: \n";
  for (size_t i=0; i<groups_cardinality; i++) {
    if (res[i]!=0.) {
      std::cout << res[i] << std::endl;
    }
  }
}