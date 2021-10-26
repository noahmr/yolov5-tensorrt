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
    yolov5-tensorrt 
    nvinfer
    nvonnxparser
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBRARIES}
)

install(TARGETS process_live
        DESTINATION ${CMAKE_INSTALL_BINDIR})