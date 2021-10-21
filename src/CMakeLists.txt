add_library(yolov5_tensorrt 
    yolov5_detection.cpp
    yolov5_detector.cpp
    yolov5_detector_internal.cpp
    yolov5_builder.cpp
    yolov5_common.cpp
    yolov5_logging.cpp
)
target_include_directories(yolov5_tensorrt PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(yolov5_tensorrt)