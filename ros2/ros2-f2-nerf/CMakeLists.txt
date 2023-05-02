cmake_minimum_required(VERSION 3.8)
project(ros2-f2-nerf)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

###############################################################################
# setup for F2-NeRF start
###############################################################################
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")
message(STATUS "ROOT_DIR: ${ROOT_DIR}")

add_definitions(-DHALF_PRECISION)

# tiny-cuda-nn
add_definitions(-DJSON_HAS_CPP_14)
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(${ROOT_DIR}/External/tiny-cuda-nn ${CMAKE_CURRENT_BINARY_DIR}/tiny-cuda-nn)
include_directories(${TCNN_INCLUDE_DIRECTORIES})
include_directories(${ROOT_DIR}/External/tiny-cuda-nn/include)
include_directories(${ROOT_DIR}/External/tiny-cuda-nn/dependencies)

set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

# zlib, for compiling cnpy
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# Libtorch
set(CMAKE_PREFIX_PATH ${DCMAKE_PREFIX_PATH}$ ${ROOT_DIR}/External/libtorch/)
find_package(Torch REQUIRED PATHS ${ROOT_DIR}/External/libtorch NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
link_directories(${ROOT_DIR}/External/libtorch/lib/)
include_directories(${TORCH_INCLUDE_DIRS})

# YAML
add_subdirectory(${ROOT_DIR}/External/yaml-cpp ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp)
include_directories(${ROOT_DIR}/External/yaml-cpp/include)

# eigen
include_directories(${ROOT_DIR}/External/eigen-3.4.0)

set(ALL_LIBS ${TORCH_LIBRARIES}
             ${TORCH_CUDA_LIBRARIES}
             ${ZLIB_LIBRARIES}
             yaml-cpp
             fmt::fmt
             tiny-cuda-nn
             stdc++fs)

set(PROJECT_SOURCE_DIR ${ROOT_DIR})
include(${ROOT_DIR}/src/Field/field.cmake)
include(${ROOT_DIR}/src/Dataset/dataset.cmake)
include(${ROOT_DIR}/src/PtsSampler/pts_sampler.cmake)
include(${ROOT_DIR}/src/Renderer/renderer.cmake)
include(${ROOT_DIR}/src/Shader/shader.cmake)
include(${ROOT_DIR}/src/Utils/utils.cmake)

list(APPEND SRC_ALL
        ${SRC_FIELD}
        ${SRC_DATASET}
        ${SRC_GEOMETRY}
        ${SRC_PTS_SAMPLER}
        ${SRC_RENDERER}
        ${SRC_SHADER}
        ${SRC_UTILS})
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "SRC_ALL: ${SRC_ALL}")

###############################################################################
# setup for F2-NeRF end
###############################################################################

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

add_executable(nerf_based_localizer src/main.cpp
                                    src/nerf_based_localizer.cpp
                                    src/localizer_core.cpp
                                    ${SRC_ALL}
)
ament_target_dependencies(nerf_based_localizer
    rclcpp
    geometry_msgs
    sensor_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    )
install(TARGETS
  nerf_based_localizer
  DESTINATION lib/${PROJECT_NAME}
)
target_link_libraries(nerf_based_localizer ${ALL_LIBS})
