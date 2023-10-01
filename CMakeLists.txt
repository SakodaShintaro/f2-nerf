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

# Libtorch
set(CMAKE_PREFIX_PATH ${DCMAKE_PREFIX_PATH}$ ${CMAKE_SOURCE_DIR}/External/libtorch/)
find_package(Torch REQUIRED PATHS ${CMAKE_SOURCE_DIR}/External/libtorch NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
link_directories(${CMAKE_SOURCE_DIR}/External/libtorch/lib/)
include_directories(${TORCH_INCLUDE_DIRS})

# eigen
include_directories(External/eigen-3.4.0)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# GTSAM
find_package(GTSAM REQUIRED)

include(src/field/field.cmake)
include(src/dataset/dataset.cmake)
include(src/points_sampler/points_sampler.cmake)
include(src/renderer/renderer.cmake)
include(src/shader/shader.cmake)
include(src/utils/utils.cmake)
include(src/localizer/localizer.cmake)

list(APPEND SRC_ALL
        ${SRC_FIELD}
        ${SRC_DATASET}
        ${SRC_GEOMETRY}
        ${SRC_PTS_SAMPLER}
        ${SRC_RENDERER}
        ${SRC_SHADER}
        ${SRC_UTILS}
        ${SRC_LOCALIZER})

set(ALL_LIBS ${TORCH_LIBRARIES}
        ${TORCH_CUDA_LIBRARIES}
        ${OpenCV_LIBS}
        stdc++fs
        gtsam)

# ------------------------------------ Add executables ---------------------------------------
add_executable(main main.cpp
        src/main_functions/train_manager.cpp
        src/main_functions/infer.cpp
        src/main_functions/test.cpp
        src/main_functions/walk.cpp
        ${SRC_ALL})
target_link_libraries(main ${ALL_LIBS})
set_property(TARGET main PROPERTY CXX_STANDARD 17)
