/*
-- TPC-H Query 1

select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
from
        lineitem
where
        l_shipdate <= date '1998-12-01' - interval '90' day
group by
        l_returnflag,
        l_linestatus
order by
        l_returnflag,
        l_linestatus
scan->map->groupby->map->sort

scan
    filter
    map (sum_disc_price)
    map (sum_charge)
    groupby (keys=[l_returnflag, l_linestatus], ...)
    map(for avg columns)
    sort(keys=[l_returnflag, l_linestatus])
*/

/*
Hyper style loop

sum_disc_price = malloc
sum_charge = malloc
for i in range(lineitem.size()):
    if (l_shipdate[i] <= date) // save some computations
        sum_disc_price[i] = l_extendedprice[i] * (1-l_discount[i])
        sum_charge[i] = sum_disc_price[i]*(1+l_tax[i])
create a static map for each groupbys? dictionary size for each domain known, create a static map of
that length

create one static map <key, index>
groupby result in the following struct (32 bytes 4 byte per entry)
// use a column format instead of struct, what if upper data streams want to use this?
struct gc {
    int sum_qty;
    float sum_base_price;
    float sum_disc_price;
    float sum_charge;
    float avg_qty;
    float avg_price;
    float avg_disc;
    int count_order;
}

grouped_col = malloc (sizeof(gc) * estimated_map_size);
group_dict = static_map(estimated_map_size)

for i in range(lineitem.size()):
    if (<l_returnflag[i], l_linestatus[i]> exists in group_dict):
        index_in_data = group_dict[<l_returnflag[i], l_linestatus[i]>]
        group(grouped_col, index_in_data, gc(l_quantity[i], l_ep[i], l_disc[i], l_tax[i]))
    else:
        index_in_data = fetchAndAdd(global_group_count, 1)
        group(grouped_col, index_in_data, gc(l_quantity[i], l_ep[i], l_disc[i], l_tax[i]))

gather the map data into an array and sort them and swap all entries in respective columns
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
__global__ void compute_disc_charge(double* disc_price,
                                    double* charge,
                                    double* l_extendedprice,
                                    double* l_discount,
                                    double* l_tax,
                                    size_t lineitem_size)
{
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid >= lineitem_size) return;

  disc_price[tid] = l_extendedprice[tid] - l_extendedprice[tid] * l_discount[tid];
  charge[tid]     = disc_price[tid] + l_tax[tid] * disc_price[tid];
}

// if the domain of the group by is known at compile time, then we can use that information to group
// by in this case since we have dictionary encoded certain columns (we can assume that this
// metadata is present in the database), and hence calculate the index of the group by accordingly.
__global__ void aggregate(size_t lineitem_size,
                          int8_t* l_returnflag,
                          int8_t* l_linestatus,
                          int32_t* l_shipdate,
                          int64_t* l_quantity,
                          double* l_extendedprice,
                          double* l_discount,
                          double* l_tax,
                          double* l_disc_price,
                          double* l_charge,
                          double* sum_charge,
                          double* sum_base_price,
                          double* sum_disc_price,
                          double* sum_discount,
                          int64_t* sum_qty,
                          int64_t* count_order,
                          int linestatus_size)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= lineitem_size) return;
  // filter by date as well.
  // printf("%d\n", l_shipdate[tid]);
  if (l_shipdate[tid] > 10471) return;  // l_shipdate <= date '1998-12-01' - interval '90' day
  // since the keys are dictionary encoded, we can use their integer values directly to address
  int agg_key = ((int32_t)l_returnflag[tid]) * linestatus_size + ((int32_t)l_linestatus[tid]);
  atomicAdd(&(sum_charge[agg_key]), l_charge[tid]);
  atomicAdd(&(sum_base_price[agg_key]), l_extendedprice[tid]);
  atomicAdd(&(sum_disc_price[agg_key]), l_disc_price[tid]);
  atomicAdd((int*)&(sum_qty[agg_key]), (int)l_quantity[tid]);
  atomicAdd((int*)&(count_order[agg_key]), 1);
  atomicAdd(&(sum_discount[agg_key]), l_discount[tid]);
}

__global__ void aggregate_avg_columns(size_t agg_table_size,
                                      double* sum_base_price,
                                      double* avg_price,
                                      int64_t* sum_qty,
                                      double* avg_qty,
                                      double* sum_discount,
                                      double* avg_discount,
                                      int64_t* count_order)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= agg_table_size) return;

  if (count_order[tid] == 0) return;
  avg_price[tid]    = sum_base_price[tid] / count_order[tid];
  avg_qty[tid]      = (float)sum_qty[tid] / (float)count_order[tid];
  avg_discount[tid] = sum_discount[tid] / count_order[tid];
}

int main(int argc, const char** argv)
{
  std::cout << std::setprecision(10);

  std::string dbDir         = getDataDir(argv, argc);
  std::string lineitem_file = dbDir + "lineitem.parquet";

  auto lineitem_table  = getArrowTable(lineitem_file);
  size_t lineitem_size = lineitem_table->num_rows();

  int64_t* l_quantity     = read_column<int64_t>(lineitem_table, "l_quantity");
  int32_t* l_shipdate     = read_column<int32_t>(lineitem_table, "l_shipdate");
  double* l_ep_test       = read_column<double>(lineitem_table, "l_extendedprice");
  double* l_extendedprice = read_column<double>(lineitem_table, "l_extendedprice");
  double* l_discount      = read_column<double>(lineitem_table, "l_discount");
  double* l_tax           = read_column<double>(lineitem_table, "l_tax");
  StringDictEncodedColumn* l_returnflag =
    read_string_dict_encoded_column(lineitem_table, "l_returnflag");
  StringDictEncodedColumn* l_linestatus =
    read_string_dict_encoded_column(lineitem_table, "l_linestatus");

  int8_t *d_l_returnflag, *d_l_linestatus;
  int32_t* d_l_shipdate;
  cudaMalloc(&d_l_linestatus, sizeof(int8_t) * lineitem_size);
  cudaMemcpy(
    d_l_linestatus, l_linestatus->column, sizeof(int8_t) * lineitem_size, cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_returnflag, sizeof(int8_t) * lineitem_size);
  cudaMemcpy(
    d_l_returnflag, l_returnflag->column, sizeof(int8_t) * lineitem_size, cudaMemcpyHostToDevice);
  cudaMalloc(&d_l_shipdate, sizeof(int32_t) * lineitem_size);
  cudaMemcpy(d_l_shipdate, l_shipdate, sizeof(int32_t) * lineitem_size, cudaMemcpyHostToDevice);

  // compute the sum_disc_price and sum_charge
  int64_t* d_l_quantity;
  cudaMalloc(&d_l_quantity, sizeof(int64_t) * lineitem_size);
  cudaMemcpy(d_l_quantity, l_quantity, sizeof(int64_t) * lineitem_size, cudaMemcpyHostToDevice);

  double *d_disc_price, *d_charge, *d_l_extendedprice, *d_l_discount, *d_l_tax;
  cudaMalloc(&d_disc_price, sizeof(double) * lineitem_size);

  cudaMalloc(&d_charge, sizeof(double) * lineitem_size);

  cudaMalloc(&d_l_extendedprice, sizeof(double) * lineitem_size);
  cudaMemcpy(
    d_l_extendedprice, l_extendedprice, sizeof(double) * lineitem_size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_l_discount, sizeof(double) * lineitem_size);
  cudaMemcpy(d_l_discount, l_discount, sizeof(double) * lineitem_size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_l_tax, sizeof(double) * lineitem_size);
  cudaMemcpy(d_l_tax, l_tax, sizeof(double) * lineitem_size, cudaMemcpyHostToDevice);

  // Launch kernel to compute the above 2 float arrays
  size_t TB = 32;
  compute_disc_charge<<<std::ceil((float)lineitem_size / (float)TB), TB>>>(
    d_disc_price, d_charge, d_l_extendedprice, d_l_discount, d_l_tax, lineitem_size);
  // create result columns
  size_t estimated_groups_size = l_returnflag->dict.size() * l_linestatus->dict.size();
  int64_t *d_sum_qty, *d_count_order;
  double *d_sum_base_price, *d_sum_disc_price, *d_sum_charge, *d_avg_qty, *d_avg_price, *d_avg_disc,
    *d_sum_discount;

  cudaMalloc(&d_sum_qty, sizeof(int64_t) * estimated_groups_size);
  cudaMemset(d_sum_qty, 0, sizeof(int64_t) * estimated_groups_size);

  cudaMalloc(&d_count_order, sizeof(int64_t) * estimated_groups_size);
  cudaMemset(d_count_order, 0, sizeof(int64_t) * estimated_groups_size);

  cudaMalloc(&d_sum_base_price, sizeof(double) * estimated_groups_size);
  cudaMemset(d_sum_base_price, 0., sizeof(double) * estimated_groups_size);

  cudaMalloc(&d_sum_disc_price, sizeof(double) * estimated_groups_size);
  cudaMemset(d_sum_disc_price, 0., sizeof(double) * estimated_groups_size);

  cudaMalloc(&d_sum_charge, sizeof(double) * estimated_groups_size);
  cudaMemset(d_sum_charge, 0., sizeof(double) * estimated_groups_size);

  cudaMalloc(&d_avg_qty, sizeof(double) * estimated_groups_size);
  cudaMemset(d_avg_qty, 0., sizeof(double) * estimated_groups_size);

  cudaMalloc(&d_avg_price, sizeof(double) * estimated_groups_size);
  cudaMemset(d_avg_price, 0., sizeof(double) * estimated_groups_size);

  cudaMalloc(&d_avg_disc, sizeof(double) * estimated_groups_size);
  cudaMemset(d_avg_disc, 0., sizeof(double) * estimated_groups_size);

  cudaMalloc(&d_sum_discount, sizeof(double) * estimated_groups_size);
  cudaMemset(d_sum_discount, 0., sizeof(double) * estimated_groups_size);

  // create the map to be aggregated in

  std::cout << "Line status dict: \n";
  for (auto e : l_linestatus->dict) {
    std::cout << e.first << ": " << (int32_t)e.second << "\n";
  }
  std::cout << "Size: " << l_linestatus->dict.size();
  std::cout << "\n";
  std::cout << "return flag dict: \n";
  for (auto e : l_returnflag->dict) {
    std::cout << e.first << ": " << (int32_t)e.second << "\n";
  }
  std::cout << "Size: " << l_returnflag->dict.size();
  std::cout << "\n";
  // launch the kernel to aggregate
  aggregate<<<std::ceil((float)lineitem_size / (float)TB), TB>>>(lineitem_size,
                                                                 d_l_returnflag,
                                                                 d_l_linestatus,
                                                                 d_l_shipdate,
                                                                 d_l_quantity,
                                                                 d_l_extendedprice,
                                                                 d_l_discount,
                                                                 d_l_tax,
                                                                 d_disc_price,
                                                                 d_charge,
                                                                 d_sum_charge,
                                                                 d_sum_base_price,
                                                                 d_sum_disc_price,
                                                                 d_sum_discount,
                                                                 d_sum_qty,
                                                                 d_count_order,
                                                                 l_linestatus->dict.size());
  aggregate_avg_columns<<<std::ceil((float)estimated_groups_size / (float)TB), TB>>>(
    estimated_groups_size,
    d_sum_base_price,
    d_avg_price,
    d_sum_qty,
    d_avg_qty,
    d_sum_discount,
    d_avg_disc,
    d_count_order);

  double* cpu_sum_base_price_2 = (double*)malloc(sizeof(double) * estimated_groups_size);
  memset(cpu_sum_base_price_2, 0., sizeof(double) * estimated_groups_size);
  for (int i = 0; i < lineitem_size; i++) {
    if (l_shipdate[i] > 10471) continue;
    int agg_key = ((int32_t)l_returnflag->column[i]) * l_linestatus->dict.size() +
                  ((int32_t)l_linestatus->column[i]);
    cpu_sum_base_price_2[agg_key] += l_ep_test[i];
  }

  // gather and print the results
  int64_t* sum_qty       = (int64_t*)malloc(sizeof(int64_t) * estimated_groups_size);
  double* sum_base_price = (double*)malloc(sizeof(double) * estimated_groups_size);
  double* sum_disc_price = (double*)malloc(sizeof(double) * estimated_groups_size);
  double* sum_charge     = (double*)malloc(sizeof(double) * estimated_groups_size);
  double* avg_qty        = (double*)malloc(sizeof(double) * estimated_groups_size);
  double* avg_price      = (double*)malloc(sizeof(double) * estimated_groups_size);
  double* avg_disc       = (double*)malloc(sizeof(double) * estimated_groups_size);
  int64_t* count_order   = (int64_t*)malloc(sizeof(int64_t) * estimated_groups_size);
  cudaMemcpy(sum_qty, d_sum_qty, sizeof(int64_t) * estimated_groups_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(
    count_order, d_count_order, sizeof(int64_t) * estimated_groups_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(sum_base_price,
             d_sum_base_price,
             sizeof(double) * estimated_groups_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(sum_disc_price,
             d_sum_disc_price,
             sizeof(double) * estimated_groups_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(
    sum_charge, d_sum_charge, sizeof(double) * estimated_groups_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(avg_qty, d_avg_qty, sizeof(double) * estimated_groups_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(
    avg_price, d_avg_price, sizeof(double) * estimated_groups_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(avg_disc, d_avg_disc, sizeof(double) * estimated_groups_size, cudaMemcpyDeviceToHost);
  std::cout << "sum_qty\tsum_base_price\tsum_disc_price\tsum_charge\tavg_qty\tavg_price\tavg_"
               "disc\tcount_order\n";
  for (int i = 0; i < estimated_groups_size; i++) {
    std::cout << sum_qty[i] << "\t" << sum_base_price[i] << "\t" << sum_disc_price[i] << "\t"
              << sum_charge[i] << "\t" << avg_qty[i] << "\t" << avg_price[i] << "\t" << avg_disc[i]
              << "\t" << count_order[i] << "\n";
  }
}
