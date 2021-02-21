cmake_minimum_required(VERSION 3.10)

# set the project name
project(CSI_Camera)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-fPIC -g -pedantic -Wall -Wextra")

if(LOG) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLOG=1")
endif()

find_package(OpenCV 4.5 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})
include_directories(src)

# Build the actual library
add_library(CSI_Camera SHARED src/FrameTimeChecker.cpp src/CSI_Camera.cpp src/CSI_StereoCamera.cpp)
target_link_libraries(CSI_Camera ${OpenCV_LIBRARIES})

# add the test application
add_executable(test_CSI_Camera tests/cam_test.cpp)
target_link_libraries(test_CSI_Camera CSI_Camera ${OpenCV_LIBRARIES} nvvpi)

file(COPY config DESTINATION ./)
