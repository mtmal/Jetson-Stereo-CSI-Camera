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

# make sure we have the necessary submodule
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

find_package(OpenCV 4.5 REQUIRED)

# add OpenCV directory
include_directories(${OpenCV_INCLUDE_DIR})
# add the main source directory
include_directories(src)

# add the jetracer utils submodule
add_subdirectory(JetracerUtils)
include_directories(JetracerUtils/src)

# Build the actual library
add_library(CSI_Camera SHARED src/FrameTimeChecker.cpp src/CSI_Camera.cpp src/CSI_StereoCamera.cpp)
target_link_libraries(CSI_Camera JetracerUtils ${OpenCV_LIBRARIES})

# add the test application
add_executable(test_CSI_Camera tests/cam_test.cpp)
target_link_libraries(test_CSI_Camera CSI_Camera ${OpenCV_LIBRARIES})

file(COPY config DESTINATION ./)
