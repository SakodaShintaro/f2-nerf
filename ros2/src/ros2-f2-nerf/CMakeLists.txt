cmake_minimum_required(VERSION 3.8)
project(ros2-f2-nerf)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-w)
endif()

###############################################################################
# setup for F2-NeRF start
###############################################################################
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../")
message(STATUS "ROOT_DIR: ${ROOT_DIR}")

add_definitions(-DHALF_PRECISION)

# Libtorch
set(CMAKE_PREFIX_PATH ${DCMAKE_PREFIX_PATH}$ ${ROOT_DIR}/External/libtorch/)
find_package(Torch REQUIRED PATHS ${ROOT_DIR}/External/libtorch NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
link_directories(${ROOT_DIR}/External/libtorch/lib/)
include_directories(${TORCH_INCLUDE_DIRS})

# eigen
include_directories(${ROOT_DIR}/External/eigen-3.4.0)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# GTSAM
find_package(GTSAM REQUIRED)

set(ALL_LIBS ${TORCH_LIBRARIES}
             ${TORCH_CUDA_LIBRARIES}
             ${OpenCV_LIBS}
             stdc++fs
             gtsam)

set(PROJECT_SOURCE_DIR ${ROOT_DIR})
include(${ROOT_DIR}/src/field/field.cmake)
include(${ROOT_DIR}/src/dataset/dataset.cmake)
include(${ROOT_DIR}/src/points_sampler/points_sampler.cmake)
include(${ROOT_DIR}/src/Renderer/renderer.cmake)
include(${ROOT_DIR}/src/Shader/shader.cmake)
include(${ROOT_DIR}/src/Utils/utils.cmake)
include(${ROOT_DIR}/src/localizer/localizer.cmake)

list(APPEND SRC_ALL
        ${SRC_FIELD}
        ${SRC_DATASET}
        ${SRC_GEOMETRY}
        ${SRC_PTS_SAMPLER}
        ${SRC_RENDERER}
        ${SRC_SHADER}
        ${SRC_UTILS}
        ${SRC_LOCALIZER})
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
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(tier4_localization_msgs REQUIRED)

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
                                    ${SRC_ALL}
)
ament_target_dependencies(nerf_based_localizer
    rclcpp
    geometry_msgs
    sensor_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    tier4_localization_msgs
    std_srvs
    )
install(TARGETS
  nerf_based_localizer
  DESTINATION lib/${PROJECT_NAME}
)
target_link_libraries(nerf_based_localizer ${ALL_LIBS})

## Test
ament_add_gtest(test_localizer_core test/test_localizer_core.cpp
                                    ${SRC_ALL}
                                    )
target_link_libraries(test_localizer_core ${ALL_LIBS})
