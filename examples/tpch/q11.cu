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

#define TILE_SIZE 1
namespace cg = cooperative_groups;
template <typename Map>
__global__ void build_hash(Map map_ref, int64_t* column, int32_t* filter, size_t column_size) {
  int tid = (threadIdx.x + blockIdx.x*blockDim.x) / TILE_SIZE;
  auto this_thread = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  
  if (tid >= column_size) return;
  if (filter[tid] < 9204) return;
  map_ref.insert(this_thread, cuco::pair{column[tid], tid});
}

struct StringColumn {
  int* sizes;
  char** stringAddresses;
  char* data;
  StringColumn() {}
};
struct StringDictEncodedColumn {
  std::unordered_map<std::string, int32_t> dict;
  int32_t* column;
};

template<typename T>
T* read_column (std::shared_ptr<arrow::Table> &table, const std::string &column) {
  // TODO: add error handling for column not present in the schema
  T* carr = (T*)malloc(sizeof(T)*table->num_rows());
  auto arrow_col = table->GetColumnByName(column);
  for (auto chunk: arrow_col->chunks()) {
    if (std::is_same<T, int64_t>::value) {
      auto intArr = std::static_pointer_cast<arrow::Int64Array>(chunk);
      for (int i=0; i<intArr->length(); i++) {
        carr[i] = intArr->Value(i);
      }
    } else if (std::is_same<T, int32_t>::value) { // use int32 for date32 for now
      // date32 type represents the number of days since UNIX epoch 1970-01-01
      auto dateArr = std::static_pointer_cast<arrow::Date32Array>(chunk);

      for (int i=0; i<dateArr->length(); i++) {
        carr[i] = dateArr->Value(i);
      }
    } else if (std::is_same<T, double>::value) {
      auto doubleArr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
      for (int i=0; i<doubleArr->length(); i++) {
       carr[i] = doubleArr->Value(i);
      }
    }
  }
  return carr;
}

StringDictEncodedColumn* read_string_dict_encoded_column(std::shared_ptr<arrow::Table> &table, const std::string &column) {
  auto arrow_col = table->GetColumnByName(column);
  StringDictEncodedColumn* result = new StringDictEncodedColumn();
  int32_t uid = 0;
  for (auto chunk: arrow_col->chunks()) {
    auto string_arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
    for (int i=0; i < string_arr->length(); i++) {
      auto str = string_arr->GetString(i);
      if (result->dict.find(str) == result->dict.end()) {
        result->dict[str] = uid; uid++;
      }
    }
  }
  result->column = new int32_t[table->num_rows()];
  int j = 0; 
  for (auto chunk: arrow_col->chunks()) {
    auto string_arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
    for (int i=0; i < string_arr->length(); i++) {
      auto str = string_arr->GetString(i);
      result->column[j++] = result->dict[str];
    }
  }
  return result;
}
StringColumn* read_string_column(std::shared_ptr<arrow::Table> &table, const std::string &column) {

  //TODO: add all kinds of error handling
  auto arrow_col = table->GetColumnByName(column);
  StringColumn* result = new StringColumn();

  // calculate the size of data
  result->sizes = (int*)malloc(sizeof(int)*table->num_rows());
  result->stringAddresses = (char**)malloc(sizeof(char*)*table->num_rows());
  int data_size = 0;
  int j=0;
  for (auto chunk: arrow_col->chunks()) {
    auto string_arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
    for (int i=0; i < string_arr->length(); i++) {
      auto str = string_arr->GetString(i);
      data_size += str.size();
      result->sizes[j++] = str.size();
    }
  }
  result->data = (char*)malloc(sizeof(char)*data_size);
  j = 0;
  char* straddr = result->data;
  for (auto chunk: arrow_col->chunks()) {
    auto string_arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
    for (int i=0; i < string_arr->length(); i++) {
      auto str = string_arr->GetString(i);
      result->stringAddresses[i] = straddr;
      straddr += str.size();
      for (auto c: str) {
        result->data[j++] = c;
      }
    }
  }
  return result;
}
    
arrow::Status read_parquet(std::string path_to_file, std::shared_ptr<arrow::Table> &table) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  // open file
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(path_to_file));
  // initialize file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, pool, &arrow_reader));

  ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));

  return arrow::Status::OK();
}
 
std::shared_ptr<arrow::Table> getArrowTable(std::string path_to_file) {
  std::shared_ptr<arrow::Table> lineitem_table;
  arrow::Status st = read_parquet(path_to_file, lineitem_table);
  if (st != arrow::Status::OK()) {
    std::cerr << st.ToString();
    return nullptr;
  }
  return lineitem_table;
}

int main() {
  std::string dbDir = "/media/ajayakar/space/src/tpch/data/tables/scale-10.0/";
  std::string lineitem_file = dbDir + "lineitem.parquet";
  std::string orders_file = dbDir + "orders.parquet";
  std::string customer_file = dbDir + "customer.parquet";

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

  std::size_t const capacity = orders_size*2;
  auto o_orderkey_map = cuco::static_map{
    capacity,
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    thrust::equal_to<int>{},
    cuco::linear_probing<TILE_SIZE, cuco::default_hash_function<int>>()
  };  
  int64_t *d_o_orderkey; int32_t *d_o_orderdate;
  cudaMalloc(&d_o_orderkey, orders_size*sizeof(int64_t));
  cudaMalloc(&d_o_orderdate, orders_size*sizeof(int32_t));
  cudaMemcpy(d_o_orderkey, o_orderkey, orders_size*sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_o_orderdate, o_orderdate, orders_size*sizeof(int32_t), cudaMemcpyHostToDevice);
  build_hash<<<(int)std::ceil(((float)orders_size)/1024.0), 1024>>> (
    o_orderkey_map.ref(cuco::insert),
    d_o_orderkey,
    d_o_orderdate,
    orders_size
  );
}
