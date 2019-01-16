cmake_minimum_required(VERSION 3.2)

# Project name: ShadowDet
project(ShadowDet)

# OpenCV required
find_package(OpenCV REQUIRED)

# Include OpenCV and FindShadow.h
include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(ShadowDet src/Main.cpp src/FindShadow.cpp)

target_link_libraries(ShadowDet ${OpenCV_LIBS} -lpthread)
