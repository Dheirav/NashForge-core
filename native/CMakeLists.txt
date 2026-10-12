cmake_minimum_required(VERSION 3.18)
project(pokerbot_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native)

# The interpreter that will import the module, so headers and ABI match. Passed
# in by build.sh rather than found, because this repo has a venv and CMake's
# search would happily pick the system Python and produce a module that imports
# and then segfaults.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(pokerbot_native STABLE_ABI NB_STATIC src/bindings.cpp)
# The solver's worker threads (MCCFR::train with threads > 1).
find_package(Threads REQUIRED)
target_link_libraries(pokerbot_native PRIVATE Threads::Threads)

install(TARGETS pokerbot_native LIBRARY DESTINATION .)
