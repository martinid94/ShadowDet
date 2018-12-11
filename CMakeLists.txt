cmake_minimum_required(VERSION 3.2)
project(ShadowDet)

find_package(OpenCV REQUIRED)

include_directories(
		    ${OpenCV_INCLUDE_DIRS}
)

add_executable(ShadowDet
		src/Main.cpp
)

target_link_libraries(ShadowDet 
			${OpenCV_LIBS}
)
