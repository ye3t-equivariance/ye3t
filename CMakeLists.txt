cmake_minimum_required(VERSION 3.18)

project(libye3t_runtime LANGUAGES CXX)

include(CTest)

option(YE3T_BUILD_TORCH_ADAPTER "Build the PyTorch execution-plan adapter" ON)
option(YE3T_BUILD_CUDA_ADAPTER "Add CUDA execution-plan operators" ON)
set(
  YE3T_TORCH_DISCOVERY
  "PYTHON"
  CACHE STRING
  "PyTorch discovery mode for the adapter: PYTHON or LIBTORCH"
)
set_property(
  CACHE YE3T_TORCH_DISCOVERY
  PROPERTY STRINGS PYTHON LIBTORCH
)

add_library(
  ye3t_runtime_core
  STATIC
  ye3t/runtime/csrc/ye3t_runtime_core.cpp
)
target_include_directories(
  ye3t_runtime_core
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ye3t/runtime/csrc>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(ye3t_runtime_core PUBLIC cxx_std_17)
set_target_properties(
  ye3t_runtime_core
  PROPERTIES
  OUTPUT_NAME ye3t_runtime
  POSITION_INDEPENDENT_CODE ON
)

if(BUILD_TESTING)
  add_executable(
    ye3t_runtime_core_standalone_test
    tests/cpp/test_runtime_core_standalone.cpp
  )
  target_link_libraries(
    ye3t_runtime_core_standalone_test
    PRIVATE
    ye3t_runtime_core
  )
  add_test(
    NAME ye3t_runtime_core_standalone
    COMMAND
      ye3t_runtime_core_standalone_test
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/shared_monomial_plan_v1.txt"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/pace_radial_spline_v1.txt"
  )
endif()

if(YE3T_BUILD_TORCH_ADAPTER)
  if(YE3T_BUILD_CUDA_ADAPTER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
  endif()
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  if(YE3T_TORCH_DISCOVERY STREQUAL "PYTHON")
    set(
      YE3T_TORCH_QUERY_OUTPUT
      "${CMAKE_CURRENT_BINARY_DIR}/ye3t_torch_python.cmake"
    )
    execute_process(
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/query_torch.py"
        "${YE3T_TORCH_QUERY_OUTPUT}"
      RESULT_VARIABLE YE3T_TORCH_QUERY_RESULT
      ERROR_VARIABLE YE3T_TORCH_QUERY_ERROR
    )
    if(NOT YE3T_TORCH_QUERY_RESULT EQUAL 0)
      message(
        FATAL_ERROR
        "Could not query the target Python PyTorch installation: "
        "${YE3T_TORCH_QUERY_ERROR}"
      )
    endif()
    include("${YE3T_TORCH_QUERY_OUTPUT}")
    set(
      YE3T_TORCH_LINK_LIBRARIES
      c10
      torch
      torch_cpu
      torch_python
    )
    message(
      STATUS
      "Building YE3T Torch adapter against Python PyTorch "
      "${YE3T_TORCH_VERSION}"
    )
  elseif(YE3T_TORCH_DISCOVERY STREQUAL "LIBTORCH")
    find_package(Torch REQUIRED)
    set(YE3T_TORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIRS}")
    set(YE3T_TORCH_LINK_LIBRARIES "${TORCH_LIBRARIES}")
    set(YE3T_TORCH_LIBRARY_DIRS "")
    set(YE3T_PYTHON_EXTENSION_SUFFIX "")
  else()
    message(
      FATAL_ERROR
      "YE3T_TORCH_DISCOVERY must be PYTHON or LIBTORCH"
    )
  endif()
  add_library(
    ye3t_runtime_torch
    MODULE
    ye3t/runtime/csrc/execution_plan_torch.cpp
  )
  if(YE3T_BUILD_CUDA_ADAPTER)
    target_sources(
      ye3t_runtime_torch
      PRIVATE
      ye3t/runtime/csrc/execution_plan_cuda.cu
    )
    target_compile_definitions(
      ye3t_runtime_torch
      PRIVATE
      YE3T_HAS_CUDA=1
    )
    target_compile_options(
      ye3t_runtime_torch
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_link_libraries(
      ye3t_runtime_torch
      PRIVATE
      c10_cuda
      torch_cuda
      CUDA::cudart
    )
    set_target_properties(
      ye3t_runtime_torch
      PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )
  endif()
  target_include_directories(
    ye3t_runtime_torch
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ye3t/runtime/csrc
    ${YE3T_TORCH_INCLUDE_DIRS}
  )
  target_link_directories(
    ye3t_runtime_torch
    PRIVATE
    ${YE3T_TORCH_LIBRARY_DIRS}
  )
  target_link_libraries(
    ye3t_runtime_torch
    PRIVATE
    ye3t_runtime_core
    ${YE3T_TORCH_LINK_LIBRARIES}
    Python3::Module
  )
  target_compile_features(ye3t_runtime_torch PRIVATE cxx_std_20)
  target_compile_definitions(
    ye3t_runtime_torch
    PRIVATE
    TORCH_EXTENSION_NAME=_execution_plan_native
    TORCH_API_INCLUDE_EXTENSION_H
  )
  if(YE3T_TORCH_DISCOVERY STREQUAL "PYTHON")
    target_compile_definitions(
      ye3t_runtime_torch
      PRIVATE
      _GLIBCXX_USE_CXX11_ABI=${YE3T_TORCH_CXX11_ABI}
      PYBIND11_COMPILER_TYPE="${YE3T_TORCH_PYBIND11_COMPILER_TYPE}"
      PYBIND11_STDLIB="${YE3T_TORCH_PYBIND11_STDLIB}"
      PYBIND11_BUILD_ABI="${YE3T_TORCH_PYBIND11_BUILD_ABI}"
    )
  endif()
  set_target_properties(
    ye3t_runtime_torch
    PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_execution_plan_native"
    BUILD_RPATH "${YE3T_TORCH_LIBRARY_DIRS}"
    INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
  )
  if(APPLE)
    set_target_properties(
      ye3t_runtime_torch
      PROPERTIES
      INSTALL_RPATH "@loader_path/../../torch/lib"
    )
  elseif(UNIX)
    set_target_properties(
      ye3t_runtime_torch
      PROPERTIES
      INSTALL_RPATH "$ORIGIN/../../torch/lib"
    )
  endif()
  if(YE3T_PYTHON_EXTENSION_SUFFIX)
    set_target_properties(
      ye3t_runtime_torch
      PROPERTIES
      SUFFIX "${YE3T_PYTHON_EXTENSION_SUFFIX}"
    )
  endif()
  if(BUILD_TESTING)
    if(YE3T_BUILD_CUDA_ADAPTER)
      add_test(
        NAME ye3t_runtime_torch_cuda_smoke
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/smoke_torch_adapter.py"
          "$<TARGET_FILE:ye3t_runtime_torch>"
          cuda
      )
    else()
      add_test(
        NAME ye3t_runtime_torch_smoke
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/smoke_torch_adapter.py"
          "$<TARGET_FILE:ye3t_runtime_torch>"
      )
    endif()
  endif()
endif()

install(
  TARGETS ye3t_runtime_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(
  FILES ye3t/runtime/csrc/ye3t_runtime_core.h
  DESTINATION include/ye3t
)
if(YE3T_BUILD_TORCH_ADAPTER)
  install(
    TARGETS ye3t_runtime_torch
    LIBRARY DESTINATION ye3t/runtime
    RUNTIME DESTINATION ye3t/runtime
  )
  if(UNIX AND NOT APPLE)
    install(CODE [=[
      find_program(YE3T_INSTALL_PATCHELF patchelf)
      if(YE3T_INSTALL_PATCHELF)
        set(
          YE3T_INSTALLED_TORCH_ADAPTER
          "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/ye3t/runtime/$<TARGET_FILE_NAME:ye3t_runtime_torch>"
        )
        execute_process(
          COMMAND
            "${YE3T_INSTALL_PATCHELF}"
            --set-rpath
            "$ORIGIN/../../torch/lib"
            "${YE3T_INSTALLED_TORCH_ADAPTER}"
          COMMAND_ERROR_IS_FATAL ANY
        )
      endif()
    ]=])
  endif()
endif()
