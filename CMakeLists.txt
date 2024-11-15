# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(fast-las-classification)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2)
else()
  add_compile_options(-O3 -march=native)
endif()

# Find PDAL, OpenMP, Get Nannoflann
find_package(PDAL REQUIRED CONFIG)
find_package(OpenMP REQUIRED)
include(FetchContent)
FetchContent_Declare(
    nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG v1.6.2  # Use the latest stable version
)
FetchContent_MakeAvailable(nanoflann)

# Add executable
add_executable(fast_point_cloud_classifier main.cpp)

target_link_libraries(fast_point_cloud_classifier 
    PRIVATE 
    ${PDAL_LIBRARIES}
    nanoflann::nanoflann
    OpenMP::OpenMP_CXX
)

target_include_directories(fast_point_cloud_classifier PRIVATE
    ${PDAL_INCLUDE_DIRS}
    ${PDAL_INCLUDE_DIRS}/pdal)

if(WIN32)
    target_compile_definitions(fast_point_cloud_classifier PRIVATE NOMINMAX)
endif()

# Installation rules
install(TARGETS fast_point_cloud_classifier
        RUNTIME DESTINATION bin)