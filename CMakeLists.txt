cmake_minimum_required(VERSION 3.18)
project(pointclouds CXX)

# Each subfolder is its own self-contained CMake project
add_subdirectory(basics)
add_subdirectory(features)
add_subdirectory(filters)
add_subdirectory(ransac)
