cmake_minimum_required(VERSION 3.10.2)

project(deconvolution CXX CUDA)

if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_C_FLAGS "-Wall -Wextra")
	set(CMAKE_C_FLAGS_DEBUG "-g -O0")
	set(CMAKE_C_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

find_package(CUDA REQUIRED)

#! FLAGS FOR CUDA
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-std=c++11 -arch=sm_60 -Xptxas -v -O3 -restrict -D_FORCE_INLINES -D_DEBUG -lineinfo
)

cuda_add_executable(deconvolution main.cpp deconvolution.cu)
target_link_libraries(deconvolution m)

#! Unit testing project
project(unit_test CXX CUDA)

cuda_add_executable(unit_test unit_test.cpp deconvolution.cu)
target_link_libraries(unit_test m)