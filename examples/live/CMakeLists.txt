#
# process_live
#

add_executable(process_live 
    process_live.cpp
)
target_include_directories(process_live PUBLIC 
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(process_live 
    yolov5_tensorrt 
    nvinfer
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBRARIES}
)
