# Standalone C++ build of the SUKP MMAS runner (no Python needed). From the repo root:
#   cmake -S mmas -B mmas/build && cmake --build mmas/build -j
#   ./mmas/build/mmas_run --help
cmake_minimum_required(VERSION 3.18)
project(sukp_mmas CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Threads REQUIRED)
add_executable(mmas_run main.cpp)
# instance.hpp (the benchmark file parser) is shared with the solver in cpp_solver/
target_include_directories(mmas_run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_solver)
target_link_libraries(mmas_run PRIVATE Threads::Threads)
