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

# Determine the appropriate location to install the library. In
# newer versions of CMake this can be done through FindPython, but it is
# not yet widely available.
#
# See https://cmake.org/cmake/help/latest/module/FindPython.html
execute_process(
    COMMAND python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_VARIABLE YOLOV5_PYTHON_SITE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# install rule
install(TARGETS yolov5tensorrt
        LIBRARY DESTINATION ${YOLOV5_PYTHON_SITE_PATH})