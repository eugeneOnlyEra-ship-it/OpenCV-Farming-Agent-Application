cmake_minimum_required(VERSION 3.18)
project(pod_capture_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # Release matters: this is the "faster" part
endif()

# Your from-source OpenCV 5 build; only used if that folder exists.
# Override with -DOpenCV_DIR=... if needed.
if(EXISTS "/run/media/ugeonlyera/Drive2/opencv5/lib64/cmake/opencv5")
  set(OpenCV_DIR "/run/media/ugeonlyera/Drive2/opencv5/lib64/cmake/opencv5" CACHE PATH "OpenCV install location")
endif()

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(pod_capture src/pod_capture_bindings.cpp)
target_include_directories(pod_capture PRIVATE ${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pod_capture PRIVATE ${OpenCV_LIBS})

# Put the built module in the Python project root (one level up from cpp/)
# so "import pod_capture" works from main.py without sys.path changes.
set_target_properties(pod_capture PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)

add_executable(Testing src/Testing.cpp)
target_include_directories(Testing PRIVATE ${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Testing PRIVATE ${OpenCV_LIBS})
set_target_properties(Testing PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
