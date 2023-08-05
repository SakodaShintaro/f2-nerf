cmake_minimum_required(VERSION 3.18)
project(f2nerf
        VERSION 0.1
        LANGUAGES CUDA CXX)

add_compile_options(-w)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 17)

###############################################################################
# CUDA compiler setup
###############################################################################
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

add_definitions(-DHALF_PRECISION)

# tiny-cuda-nn
add_definitions(-DJSON_HAS_CPP_14)
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(External/tiny-cuda-nn)
include_directories(${TCNN_INCLUDE_DIRECTORIES})
include_directories(External/tiny-cuda-nn/include)
include_directories(External/tiny-cuda-nn/dependencies)

set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

# zlib, for compiling cnpy
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# Libtorch
set(CMAKE_PREFIX_PATH ${DCMAKE_PREFIX_PATH}$ ${CMAKE_SOURCE_DIR}/External/libtorch/)
find_package(Torch REQUIRED PATHS ${CMAKE_SOURCE_DIR}/External/libtorch NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
link_directories(${CMAKE_SOURCE_DIR}/External/libtorch/lib/)
include_directories(${TORCH_INCLUDE_DIRS})

# YAML
add_subdirectory(External/yaml-cpp)
include_directories(External/yaml-cpp/include)

# eigen
include_directories(External/eigen-3.4.0)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# GTSAM
find_package(GTSAM REQUIRED)

include(src/Field/field.cmake)
include(src/Dataset/dataset.cmake)
include(src/PtsSampler/pts_sampler.cmake)
include(src/Renderer/renderer.cmake)
include(src/Shader/shader.cmake)
include(src/Utils/utils.cmake)

list(APPEND SRC_ALL
        ${SRC_FIELD}
        ${SRC_DATASET}
        ${SRC_GEOMETRY}
        ${SRC_PTS_SAMPLER}
        ${SRC_RENDERER}
        ${SRC_SHADER}
        ${SRC_UTILS})

set(ALL_LIBS ${TORCH_LIBRARIES}
        ${TORCH_CUDA_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${OpenCV_LIBS}
        yaml-cpp
        fmt::fmt
        tiny-cuda-nn
        stdc++fs
        gtsam)

# ------------------------------------ Add executables ---------------------------------------
add_executable(main main.cpp
        src/ExpRunner.cpp
        src/Infer.cpp
        src/Test.cpp
        src/TestField.cpp
        src/Walk.cpp
        ros2/src/ros2-f2-nerf/src/localizer_core.cpp
        ${SRC_ALL})
target_link_libraries(main ${ALL_LIBS})
set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
