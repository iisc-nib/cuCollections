#=============================================================================
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

# find_package(Arrow REQUIRED)
# find_package(Parquet REQUIRED)

# Optionally specify paths to custom builds
option(ARROW_USE_CUSTOM_PATH "Use a custom Arrow build" OFF)
option(PARQUET_USE_CUSTOM_PATH "Use a custom Parquet build" OFF)

# If the user wants to specify custom paths, provide CMake options for them
if(ARROW_USE_CUSTOM_PATH)
    set(ARROW_ROOT "" CACHE PATH "Path to custom Arrow build")
endif()

if(PARQUET_USE_CUSTOM_PATH)
    set(PARQUET_ROOT "" CACHE PATH "Path to custom Parquet build")
endif()

# Try to find Arrow and Parquet packages (global/system installation)
if(NOT ARROW_USE_CUSTOM_PATH)
    find_package(Arrow CONFIG REQUIRED)
endif()

if(NOT PARQUET_USE_CUSTOM_PATH)
    find_package(Parquet CONFIG REQUIRED)
endif()

# If custom paths are specified, manually set the include and library directories
if(ARROW_USE_CUSTOM_PATH AND ARROW_ROOT)
    message(STATUS "Using custom Arrow build at ${ARROW_ROOT}")
    include_directories(${ARROW_ROOT}/include)
    link_directories(${ARROW_ROOT}/lib)
    set(ARROW_LIBS ${ARROW_ROOT}/lib/libarrow.so)
else()
    message(STATUS "Using system-installed Arrow")
    set(ARROW_LIBS libarrow.so)  # Automatically found by find_package
endif()

if(PARQUET_USE_CUSTOM_PATH AND PARQUET_ROOT)
    message(STATUS "Using custom Parquet build at ${PARQUET_ROOT}")
    include_directories(${PARQUET_ROOT}/include)
    link_directories(${PARQUET_ROOT}/lib)
    set(PARQUET_LIBS ${PARQUET_ROOT}/lib/libparquet.so)
else()
    message(STATUS "Using system-installed Parquet")
    set(PARQUET_LIBS libparquet.so)  # Automatically found by find_package
endif()

function(ConfigureExample EXAMPLE_NAME EXAMPLE_SRC)
    add_executable(${EXAMPLE_NAME} "${EXAMPLE_SRC}")
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
                                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    target_include_directories(${EXAMPLE_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}"
                                              ${arrow_SOURCE_DIR}/cpp/src
                                              ${arrow_BINARY_DIR}/cpp/src)
    target_compile_options(${EXAMPLE_NAME} PRIVATE --compiler-options=-Wall --compiler-options=-Wextra
      --expt-extended-lambda -Xcompiler -Wno-subobject-linkage)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE cuco CUDA::cudart ${ARROW_LIBS} ${PARQUET_LIBS})
endfunction(ConfigureExample)

###################################################################################################
### Example sources ###############################################################################
###################################################################################################

# ConfigureExample(STATIC_SET_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/host_bulk_example.cu")
# ConfigureExample(STATIC_SET_DEVICE_REF_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/device_ref_example.cu")
# ConfigureExample(STATIC_SET_DEVICE_SUBSETS_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/device_subsets_example.cu")
# ConfigureExample(STATIC_SET_SHARED_MEMORY_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/shared_memory_example.cu")
# ConfigureExample(STATIC_SET_MAPPING_TABLE_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_set/mapping_table_example.cu")
# ConfigureExample(STATIC_MAP_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/host_bulk_example.cu")
# ConfigureExample(STATIC_MAP_DEVICE_SIDE_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/device_ref_example.cu")
# ConfigureExample(STATIC_MAP_CUSTOM_TYPE_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/custom_type_example.cu")
# ConfigureExample(STATIC_MAP_COUNT_BY_KEY_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_map/count_by_key_example.cu")
# ConfigureExample(STATIC_MULTIMAP_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/static_multimap/host_bulk_example.cu")
# ConfigureExample(DISTINCT_COUNT_ESTIMATOR_HOST_BULK_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/distinct_count_estimator/host_bulk_example.cu")
# ConfigureExample(DISTINCT_COUNT_ESTIMATOR_DEVICE_REF_EXAMPLE "${CMAKE_CURRENT_SOURCE_DIR}/distinct_count_estimator/device_ref_example.cu")
ConfigureExample(TPCH_Q1 "${CMAKE_CURRENT_SOURCE_DIR}/tpch/q1.cu") 
ConfigureExample(TPCH_Q3 "${CMAKE_CURRENT_SOURCE_DIR}/tpch/q3.cu") 
ConfigureExample(TPCH_Q6 "${CMAKE_CURRENT_SOURCE_DIR}/tpch/q6.cu") 
ConfigureExample(TPCH_Q9 "${CMAKE_CURRENT_SOURCE_DIR}/tpch/q9.cu") 

