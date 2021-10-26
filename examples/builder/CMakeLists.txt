#
# build_engine
#

add_executable(build_engine 
    build_engine.cpp
)
target_include_directories(build_engine PUBLIC 
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(build_engine 
    yolov5-tensorrt 
    nvinfer
    nvonnxparser
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBRARIES}
)

install(TARGETS build_engine 
        DESTINATION ${CMAKE_INSTALL_BINDIR})