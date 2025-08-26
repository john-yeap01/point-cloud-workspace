cmake_minimum_required(VERSION 3.18)
project(ransac CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.12 REQUIRED COMPONENTS common io filters segmentation kdtree search visualization)

add_executable(ransac src/ransac.cpp)
target_include_directories(ransac PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(ransac PRIVATE ${PCL_LIBRARIES})

# point to umbrella-level data folder
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(DATA_DIR "${ROOT_DIR}/data")

# Injects #define DATA_DIR "/Users/at/pointclouds/data" 
target_compile_definitions(ransac PRIVATE DATA_DIR="${DATA_DIR}")


add_definitions(${PCL_DEFINITIONS})