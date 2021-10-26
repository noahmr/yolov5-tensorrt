add_library(yolov5-tensorrt SHARED
    yolov5_detection.cpp
    yolov5_detector.cpp
    yolov5_detector_internal.cpp
    yolov5_builder.cpp
    yolov5_common.cpp
    yolov5_logging.cpp
)
target_include_directories(yolov5-tensorrt PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
set_target_properties(yolov5-tensorrt PROPERTIES PUBLIC_HEADER "${YOLOV5_INCLUDE_FILES}")
target_link_libraries(yolov5-tensorrt)

## install rules
##
## Install the library to the default lib destination,
## but the header files in a dedicated subdirectory called 'yolov5-tensorrt'
install(TARGETS yolov5-tensorrt
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/yolov5-tensorrt)