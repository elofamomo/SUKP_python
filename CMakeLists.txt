cmake_minimum_required(VERSION 3.18)
project(sukp_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(sukp_solver cpp_solver/main.cpp)
target_include_directories(sukp_solver PRIVATE cpp_solver)

# Python module sukp_cpp, written to the project root so play.py and train_ranker.py can import it.
# Configure with -DPython_EXECUTABLE=<python of your venv> so it matches the interpreter you run.
option(SUKP_BUILD_PYTHON "Build the sukp_cpp Python module" ON)
if(SUKP_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  # Prefer the pybind11 pip-installed in that interpreter's environment
  execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                  OUTPUT_VARIABLE PYBIND11_PIP_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(PYBIND11_PIP_CMAKE_DIR)
    set(pybind11_DIR "${PYBIND11_PIP_CMAKE_DIR}")
  endif()
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(sukp_cpp cpp_solver/bindings.cpp)
  target_include_directories(sukp_cpp PRIVATE cpp_solver)
  set_target_properties(sukp_cpp PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()
