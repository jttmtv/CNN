cmake_minimum_required (VERSION 3.8)
project ("CNN")

message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fopenmp -mavx")
set(CMAKE_BUILD_TYPE "Release")


set(CMAKE_INCLUDE_CURRENT_DIR ON)
aux_source_directory(. source_list)
find_package(OpenCV 3.1 REQUIRED)
include_directories(
	${PROJECT_SOURCE_DIR}/
	${OpenCV_INCLUDE_DIRS}
)
add_executable(CNN ${source_list})
target_link_libraries(CNN ${OpenCV_LIBS})
