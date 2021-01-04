cmake_minimum_required (VERSION 3.1)
project ("CNN")

message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fopenmp -mavx")
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(LOCAL_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/src SRC_FLIES)

message(STATUS "Checking OpenMP")
find_package(OpenMP)
IF(OPENMP_FOUND)
    message("Found OpenMP!")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    add_definitions(-DENABLE_OPENMP)
ELSE()
    message("Missed OpenMP!")
ENDIF()

message(STATUS "Checking OpenCV")
find_package(OpenCV REQUIRED)
IF(OPENCV_FOUND)
    message("Found OpenCV!")
    include_directories(
            ${LOCAL_INCLUDE_DIR}
            ${OpenCV_INCLUDE_DIRS}
    )
ELSE()
    message(FATAL_ERROR "Missed OpenCV! CMake will exit")
ENDIF()

add_executable(${PROJECT_NAME} main.cpp ${SRC_FLIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
