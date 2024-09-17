#include <unordered_map>

#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/record_batch.h>
#include <arrow/array.h>
#include <iostream>
#include <parquet/arrow/reader.h>

struct StringColumn {
  int* sizes;
  char** stringAddresses;
  char* data;
  StringColumn() {}
};
struct StringDictEncodedColumn {
  std::unordered_map<std::string, int8_t> dict;
  int8_t* column;
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
    // TODO: come up with better type castings
      // date32 type represents the number of days since UNIX epoch 1970-01-01
      auto dateArr = std::static_pointer_cast<arrow::Date32Array>(chunk);
      for (int i=0; i<dateArr->length(); i++) {
        carr[i] = dateArr->Value(i);
      }
    } else if (std::is_same<T, int16_t>::value) {
    // TODO: come up with better type castings
      // same for this
      auto dateArr = std::static_pointer_cast<arrow::Date32Array>(chunk);

      for (int i=0; i<dateArr->length(); i++) {
        carr[i] = (T)dateArr->Value(i);
      }
    } 
    else if (std::is_same<T, double>::value) {
      auto doubleArr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
      for (int i=0; i<doubleArr->length(); i++) {
       carr[i] = doubleArr->Value(i);
      }
    }
  }
  return carr;
}

template<typename T>
T* read_column_typecasted(std::shared_ptr<arrow::Table> &table, const std::string &column) {
  T* carr = (T*)malloc(sizeof(T)*table->num_rows());
  auto arrow_col = table->GetColumnByName(column);
  for (auto chunk: arrow_col->chunks()) {
    if (std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value) {
      auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
      for (int i=0; i<arr->length(); i++) {
        carr[i] = (T)arr->Value(i);
      }
    } else if (std::is_same<T, float>::value) {
      auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
      for (int i=0; i<arr->length(); i++) {
        carr[i] = (T)arr->Value(i);
      }
    }
  }
  return carr;
}

StringDictEncodedColumn* read_string_dict_encoded_column(std::shared_ptr<arrow::Table> &table, const std::string &column) {
  auto arrow_col = table->GetColumnByName(column);
  StringDictEncodedColumn* result = new StringDictEncodedColumn();
  int8_t uid = 0;
  for (auto chunk: arrow_col->chunks()) {
    auto string_arr = std::static_pointer_cast<arrow::LargeStringArray>(chunk);
    for (int i=0; i < string_arr->length(); i++) {
      auto str = string_arr->GetString(i);
      if (result->dict.find(str) == result->dict.end()) {
        result->dict[str] = uid; uid++;
      }
    }
  }
  result->column = new int8_t[table->num_rows()];
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
int getGridSize(size_t column_size, int thread_block_size) {
  return (int)std::ceil(((float)column_size)/((float)thread_block_size));
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
