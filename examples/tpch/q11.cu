#include<iostream>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include <parquet/arrow/reader.h>


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

struct StringColumn {
  int* sizes;
  char** stringAddresses;
  char* data;
  StringColumn() {}
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
    }
  }
  return carr;
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
 

int main() {
    std::string path_to_file = "/media/ajayakar/space/src/tpch/data/tables/scale-1.0/lineitem.parquet";
    
    std::shared_ptr<arrow::Table> table;
    arrow::Status st = read_parquet(path_to_file, table);
    if (st != arrow::Status::OK()) {
        std::cerr << "Some error in reading file\n" << st.ToString();
    }
    int64_t* l_orderkey = read_column<int64_t>(table, "l_orderkey");
    auto l_shipmode = read_string_column(table, "l_shipmode");

    for (int i=0; i<table->num_rows(); i++) {
      std::cout << l_shipmode->sizes[i];
      std::cout << std::endl;
      for (int j=0; j<l_shipmode->sizes[i]; j++) {
        std::cout << l_shipmode->stringAddresses[i][j];
      }
      std::cout << std::endl;
    }
}
