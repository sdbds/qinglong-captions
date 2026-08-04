cmake_minimum_required(VERSION 3.20)

project(auto_rig_spine_runtime LANGUAGES CXX)

if(NOT DEFINED SPINE_CPP_ROOT)
  message(FATAL_ERROR "Configure with -DSPINE_CPP_ROOT=<official spine-cpp CMake root>.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory("${SPINE_CPP_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/spine-cpp")

add_executable(auto_rig_spine_runtime main.cpp)
target_compile_definitions(auto_rig_spine_runtime PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
if(MSVC)
  target_compile_options(auto_rig_spine_runtime PRIVATE /W4 /permissive- /MP)
else()
  target_compile_options(auto_rig_spine_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(auto_rig_spine_runtime PRIVATE spine-cpp)
