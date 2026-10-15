cmake_minimum_required(VERSION 3.16)
project(monogeo_depth_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

option(MONOGEO_BUILD_TESTS "Build the CTest parity tests" ON)
# LibTorch is opt-in. The ONNX export this repository produces is consumed by
# cv::dnn, which is always available and is what results/cpp_timing.json was
# measured with. Enable the LibTorch backend explicitly with
#   -DMONOGEO_USE_LIBTORCH=ON -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
# Auto-detecting was tried and rejected: a system-wide Torch package can be
# missing shared-library dependencies, which produces a binary that links but
# cannot start.
option(MONOGEO_USE_LIBTORCH "Compile the LibTorch inference backend (requires Torch_DIR)" OFF)

# --------------------------------------------------------------------------
# OpenCV is always required: it provides the ONNX runtime (cv::dnn) and the
# image preprocessing (resize / BGR->RGB / colour mapping for the output PNG).
# --------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn imgcodecs)

message(STATUS "monogeo: OpenCV ${OpenCV_VERSION} at ${OpenCV_DIR}")

add_library(monogeo_infer STATIC src/depth_infer_lib.cpp)
target_include_directories(monogeo_infer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(monogeo_infer PUBLIC ${OpenCV_LIBS})

# --------------------------------------------------------------------------
# Optional LibTorch backend. Only compiled when a Torch package is found, so a
# machine without LibTorch still builds the OpenCV DNN path (which is what this
# repository actually measures -- see results/cpp_timing.json).
# --------------------------------------------------------------------------
if(MONOGEO_USE_LIBTORCH)
  find_package(Torch REQUIRED)
  message(STATUS "monogeo: LibTorch at ${Torch_DIR}; enabling the LibTorch backend")
  target_compile_definitions(monogeo_infer PRIVATE MONOGEO_HAVE_LIBTORCH=1)
  target_link_libraries(monogeo_infer PUBLIC "${TORCH_LIBRARIES}")
  target_include_directories(monogeo_infer PUBLIC ${TORCH_INCLUDE_DIRS})
else()
  message(STATUS "monogeo: using the cv::dnn backend (pass -DMONOGEO_USE_LIBTORCH=ON for LibTorch)")
endif()

add_executable(depth_infer src/depth_infer.cpp)
target_link_libraries(depth_infer PRIVATE monogeo_infer)

if(MONOGEO_BUILD_TESTS)
  enable_testing()
  add_executable(test_depth_infer tests/test_depth_infer.cpp)
  target_link_libraries(test_depth_infer PRIVATE monogeo_infer)

  # Paths are passed in so the tests run against the artifacts the Python side
  # produced; configure with -DMONOGEO_ARTIFACTS=<dir> to point elsewhere.
  set(MONOGEO_ARTIFACTS "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts" CACHE PATH "Directory holding the exported model and reference tensors")
  file(TO_NATIVE_PATH "${MONOGEO_ARTIFACTS}/depth_net.onnx" MONOGEO_ONNX)
  file(TO_NATIVE_PATH "${MONOGEO_ARTIFACTS}/sample_input.bin" MONOGEO_INPUT)
  file(TO_NATIVE_PATH "${MONOGEO_ARTIFACTS}/python_output.bin" MONOGEO_PYOUT)
  file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/cpp_output.bin" MONOGEO_CPPOUT)
  file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/cpp_depth.png" MONOGEO_CPPPNG)

  add_test(NAME depth_infer_parity
           COMMAND test_depth_infer "${MONOGEO_ONNX}" "${MONOGEO_INPUT}" "${MONOGEO_PYOUT}"
                   "${MONOGEO_CPPOUT}" "${MONOGEO_CPPPNG}")
  set_tests_properties(depth_infer_parity PROPERTIES
                       FAIL_REGULAR_EXPRESSION "PARITY FAILED")
endif()
