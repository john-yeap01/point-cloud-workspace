cmake_minimum_required(VERSION 3.18)
project(basics CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.12 REQUIRED COMPONENTS common io filters segmentation kdtree search visualization)


# --- Library you can link from ransac ---
add_library(basics STATIC
    src/viewer.cpp
)
target_include_directories(basics
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${PCL_INCLUDE_DIRS}
)
target_link_libraries(basics
    PUBLIC
      ${PCL_LIBRARIES}
)

add_executable(pcl_viewer src/pcl_viewer_demo.cpp)
target_link_libraries(pcl_viewer PRIVATE basics)
target_include_directories(pcl_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
)

# FOR PASSTHROUGH
# add_executable(passthrough src/passthrough.cpp)
# target_link_libraries(passthrough PRIVATE ${PCL_LIBRARIES})
# target_include_directories(passthrough PRIVATE ${PCL_INCLUDE_DIRS})


add_definitions(${PCL_DEFINITIONS})