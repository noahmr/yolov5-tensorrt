#
# process_image
#

add_executable(process_image 
    process_image.cpp
)
target_include_directories(process_image PUBLIC 
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(process_image 
    yolov5_tensorrt 
    nvinfer
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBRARIES}
)
