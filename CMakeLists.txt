cmake_minimum_required(VERSION 3.22)
project(statecut LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(statecut_host_interval cuda/host_interval_cli.cpp)
target_compile_options(statecut_host_interval PRIVATE -O2 -fno-fast-math -ffp-contract=off)
option(STATECUT_CUDA "Build SM90 research CUDA kernels and arithmetic checks" OFF)
if(STATECUT_CUDA)
  enable_language(CUDA)
  add_executable(statecut_cuda_smoke cuda/smoke.cu)
  set_target_properties(statecut_cuda_smoke PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 90)
  target_compile_options(statecut_cuda_smoke PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>)
endif()
if(STATECUT_CUDA)
  add_executable(statecut_device_residual cuda/device_residual_cli.cu)
  set_target_properties(statecut_device_residual PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 90)
  target_compile_options(statecut_device_residual PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>)
endif()
add_executable(statecut_host_residual cuda/host_residual_cli.cpp)
target_compile_options(statecut_host_residual PRIVATE -O2 -fno-fast-math -ffp-contract=off)
if(STATECUT_CUDA)
  add_library(statecut_residual_cuda SHARED cuda/residual_frontier.cu)
  set_target_properties(statecut_residual_cuda PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 90)
  target_compile_options(statecut_residual_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>)
endif()
