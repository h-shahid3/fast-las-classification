# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(copy-classifications-fast)

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

# Find PDAL
find_package(PDAL REQUIRED CONFIG)

# Add executable
add_executable(fast_point_cloud_classifier main.cpp)

# Link libraries
target_link_libraries(fast_point_cloud_classifier 
    PRIVATE 
    ${PDAL_LIBRARIES}
)

if(WIN32)
    target_compile_definitions(fast_point_cloud_classifier PRIVATE NOMINMAX)
endif()

# Installation rules
install(TARGETS fast_point_cloud_classifier
        RUNTIME DESTINATION bin)