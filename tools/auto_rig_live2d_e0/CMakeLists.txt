cmake_minimum_required(VERSION 3.20)

project(auto_rig_live2d_e0 LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "The E0 render harness currently requires Windows and MSVC.")
endif()
if(NOT DEFINED CUBISM_SDK_ROOT)
  message(FATAL_ERROR "Configure with -DCUBISM_SDK_ROOT=<CubismSdkForNative root>.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

set(CUBISM_CORE_ROOT "${CUBISM_SDK_ROOT}/Core")
set(CUBISM_FRAMEWORK_ROOT "${CUBISM_SDK_ROOT}/Framework")
set(FRAMEWORK_SOURCE D3D11)

add_library(Live2DCubismCore STATIC IMPORTED GLOBAL)
set_target_properties(
  Live2DCubismCore
  PROPERTIES
    IMPORTED_LOCATION_DEBUG
      "${CUBISM_CORE_ROOT}/lib/windows/x86_64/143/Live2DCubismCore_MDd.lib"
    IMPORTED_LOCATION_RELEASE
      "${CUBISM_CORE_ROOT}/lib/windows/x86_64/143/Live2DCubismCore_MD.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CUBISM_CORE_ROOT}/include"
)

add_subdirectory("${CUBISM_FRAMEWORK_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/Framework")
target_link_libraries(Framework PUBLIC Live2DCubismCore d3d11.lib d3dcompiler.lib)

add_executable(auto_rig_live2d_e0 main.cpp)
target_compile_definitions(auto_rig_live2d_e0 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(auto_rig_live2d_e0 PRIVATE /W4 /permissive- /MP)
target_link_libraries(
  auto_rig_live2d_e0
  PRIVATE Framework d3d11.lib d3dcompiler.lib dxguid.lib windowscodecs.lib ole32.lib
)

add_custom_command(
  TARGET auto_rig_live2d_e0
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    "${CUBISM_FRAMEWORK_ROOT}/src/Rendering/D3D11/Shaders"
    "$<TARGET_FILE_DIR:auto_rig_live2d_e0>/FrameworkShaders"
)
