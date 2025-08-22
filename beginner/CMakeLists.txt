cmake_minimum_required(VERSION 3.18)
project(beginner CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL 1.12 REQUIRED COMPONENTS common io filters segmentation kdtree search visualization)

add_executable(pcl_app src/main.cpp)
target_include_directories(pcl_app PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_app PRIVATE ${PCL_LIBRARIES})

add_executable(pcl_viewer src/viewer.cpp)
target_link_libraries(pcl_viewer PRIVATE ${PCL_LIBRARIES})
target_include_directories(pcl_viewer PRIVATE ${PCL_INCLUDE_DIRS})

add_executable(passthrough src/passthrough.cpp)
target_link_libraries(passthrough PRIVATE ${PCL_LIBRARIES})
target_include_directories(passthrough PRIVATE ${PCL_INCLUDE_DIRS})



add_definitions(${PCL_DEFINITIONS})