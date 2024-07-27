cmake_minimum_required(VERSION 3.10)

project(YOLOv10TRT)

set(CMAKE_CXX_STANDARD 14)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set the path to TensorRT installation
set(TENSORRT_PATH "F:/Program Files/TensorRT-8.6.1.6")  # Update this to the actual path

# Include TensorRT directories
include_directories(${TENSORRT_PATH}/include)

# Link TensorRT libraries
link_directories(${TENSORRT_PATH}/lib)

# Include directory for your project
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define the source files
set(SOURCES
    src/main.cpp
    src/YOLOv10.cpp
)

# Add the executable target
add_executable(YOLOv10Project ${SOURCES})

# Link libraries to the target
target_link_libraries(YOLOv10Project
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
)
