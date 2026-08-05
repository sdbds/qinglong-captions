cmake_minimum_required(VERSION 3.20)

project(auto_rig_live2d_e0 LANGUAGES C CXX)

if(NOT DEFINED CUBISM_SDK_ROOT)
  message(FATAL_ERROR "Configure with -DCUBISM_SDK_ROOT=<CubismSdkForNative root>.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUBISM_CORE_ROOT "${CUBISM_SDK_ROOT}/Core")
set(CUBISM_FRAMEWORK_ROOT "${CUBISM_SDK_ROOT}/Framework")

if(WIN32 AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
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

  add_executable(
    qinglong_live2d_validator
    common/validator_common.cpp
    windows/main_d3d11.cpp
  )
  target_include_directories(qinglong_live2d_validator PRIVATE common)
  target_compile_definitions(qinglong_live2d_validator PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(qinglong_live2d_validator PRIVATE /W4 /permissive- /MP)
  target_link_libraries(
    qinglong_live2d_validator
    PRIVATE Framework d3d11.lib d3dcompiler.lib dxguid.lib windowscodecs.lib ole32.lib
  )
  set(FRAMEWORK_SHADER_PATH "${CUBISM_FRAMEWORK_ROOT}/src/Rendering/D3D11/Shaders")
elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_UTILS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    glew
    URL https://downloads.sourceforge.net/project/glew/glew/2.2.0/glew-2.2.0.tgz
    URL_HASH SHA256=d4fc82893cfb00109578d0a1a2337fb8ca335b3ceccf97b97e5cc7f08e4353e1
  )
  FetchContent_GetProperties(glew)
  if(NOT glew_POPULATED)
    FetchContent_Populate(glew)
  endif()

  find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
  add_library(qinglong_glew STATIC "${glew_SOURCE_DIR}/src/glew.c")
  target_include_directories(qinglong_glew PUBLIC "${glew_SOURCE_DIR}/include")
  target_compile_definitions(qinglong_glew PUBLIC GLEW_STATIC GLEW_EGL GLEW_NO_GLU)
  target_link_libraries(qinglong_glew PUBLIC OpenGL::GL OpenGL::EGL ${CMAKE_DL_LIBS})

  add_library(Live2DCubismCore STATIC IMPORTED GLOBAL)
  set_target_properties(
    Live2DCubismCore
    PROPERTIES
      IMPORTED_LOCATION "${CUBISM_CORE_ROOT}/lib/linux/x86_64/libLive2DCubismCore.a"
      INTERFACE_INCLUDE_DIRECTORIES "${CUBISM_CORE_ROOT}/include"
  )

  set(FRAMEWORK_SOURCE OpenGL)
  add_subdirectory("${CUBISM_FRAMEWORK_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/Framework")
  target_compile_definitions(Framework PUBLIC CSM_TARGET_LINUX_GL)
  target_include_directories(Framework PUBLIC "${glew_SOURCE_DIR}/include")
  target_link_libraries(Framework PUBLIC Live2DCubismCore qinglong_glew)

  add_executable(
    qinglong_live2d_validator
    common/validator_common.cpp
    linux/main_egl.cpp
  )
  target_include_directories(
    qinglong_live2d_validator
    PRIVATE
      common
      "${CUBISM_SDK_ROOT}/Samples/OpenGL/thirdParty/stb"
  )
  target_compile_options(qinglong_live2d_validator PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(
    qinglong_live2d_validator
    PRIVATE Framework qinglong_glew OpenGL::GL OpenGL::EGL ${CMAKE_DL_LIBS}
  )
  set(FRAMEWORK_SHADER_PATH "${CUBISM_FRAMEWORK_ROOT}/src/Rendering/OpenGL/Shaders/Standard")
else()
  message(FATAL_ERROR "The Live2D validator supports Windows x86_64 and Linux x86_64 only.")
endif()

add_custom_command(
  TARGET qinglong_live2d_validator
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    "${FRAMEWORK_SHADER_PATH}"
    "$<TARGET_FILE_DIR:qinglong_live2d_validator>/FrameworkShaders"
)
