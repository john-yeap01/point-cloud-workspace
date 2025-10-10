cmake_minimum_required(VERSION 3.18)
project(trunk_detection CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.12 REQUIRED COMPONENTS common io filters segmentation kdtree search visualization)



add_executable(trunk_detection src/trunk_detection.cpp)
target_include_directories(trunk_detection PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(trunk_detection PRIVATE ${PCL_LIBRARIES})


add_executable(region_growing src/region_growing.cpp)
target_include_directories(region_growing PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(region_growing PRIVATE ${PCL_LIBRARIES})

add_executable(conditional_clustering src/conditional_clustering.cpp)
target_include_directories(conditional_clustering PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(conditional_clustering PRIVATE ${PCL_LIBRARIES})

# point to umbrella-level data folder
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(DATA_DIR "${ROOT_DIR}/data")

# Injects #define DATA_DIR "/Users/at/pointclouds/data" 
target_compile_definitions(trunk_detection PRIVATE DATA_DIR="${DATA_DIR}")
target_compile_definitions(region_growing PRIVATE DATA_DIR="${DATA_DIR}")
target_compile_definitions(conditional_clustering PRIVATE DATA_DIR="${DATA_DIR}")

add_definitions(${PCL_DEFINITIONS})