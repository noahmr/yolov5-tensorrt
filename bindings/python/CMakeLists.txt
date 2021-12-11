pybind11_add_module(yolov5tensorrt pybind.cpp)
target_include_directories(yolov5tensorrt PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(yolov5tensorrt PRIVATE
    yolov5-tensorrt 
    nvinfer
    nvonnxparser
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBRARIES}
)