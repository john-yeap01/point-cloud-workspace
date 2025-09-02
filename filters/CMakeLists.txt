cmake_minimum_required(VERSION 3.18)
project(filters CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.12 REQUIRED COMPONENTS common io filters segmentation kdtree search visualization)
list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

# --- CSF (Cloth Simulation Filter) ---
find_path(CSF_INCLUDE_DIR
  NAMES CSF.h
  HINTS "$ENV{HOME}/.local/include"
  NO_DEFAULT_PATH
)

find_library(CSF_LIBRARY
  NAMES CSF
  HINTS "$ENV{HOME}/.local/lib"
  NO_DEFAULT_PATH
)

if (NOT CSF_INCLUDE_DIR OR NOT CSF_LIBRARY)
  message(FATAL_ERROR "CSF not found in ~/.local (headers or lib missing).")
endif()

add_executable(voxel_sampling src/voxel_sampling.cpp)
target_include_directories(voxel_sampling PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(voxel_sampling PRIVATE ${PCL_LIBRARIES})

add_executable(ground_removal src/ground_removal.cpp)
target_include_directories(ground_removal PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(ground_removal PRIVATE ${PCL_LIBRARIES})

add_executable(cloth_filter src/cloth_filter.cpp)
target_include_directories(cloth_filter PRIVATE ${PCL_INCLUDE_DIRS} ${CSF_INCLUDE_DIR})
target_link_libraries(cloth_filter PRIVATE ${PCL_LIBRARIES} ${CSF_LIBRARY})

# point to umbrella-level data folder
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(DATA_DIR "${ROOT_DIR}/data")

# Injects #define DATA_DIR "/Users/at/pointclouds/data" 
target_compile_definitions(voxel_sampling PRIVATE DATA_DIR="${DATA_DIR}")
target_compile_definitions(ground_removal PRIVATE DATA_DIR="${DATA_DIR}")
target_compile_definitions(cloth_filter PRIVATE DATA_DIR="${DATA_DIR}")

add_definitions(${PCL_DEFINITIONS})