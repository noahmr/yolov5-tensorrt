### YOLOv5-TensorRT

The goal of this library is to provide an accessible and robust method for 
performing efficient, real-time inference with 
[YOLOv5](https://github.com/ultralytics/yolov5) using NVIDIA TensorRT. 
The library is extensively documented and comes with various guided
examples.

This library was originally developed for VDL RobotSports,
an industrial team based in the Netherlands participating in the RoboCup Middle
Size League, and currently sees active use on the soccer robots.


## Features

- FP32 and FP16 inference
- Batch inference
- Support for varying input dimensions
- ONNX support
- CUDA-accelerated pre-processing
- Integration with OpenCV (with optionally also the OpenCV-CUDA module)
- Extensive documentation available on all classes, methods and functions


### Platforms

- Most modern linux distributions
- NVIDIA L4T (Jetson platform)

Dependencies:
- TensorRT >=8 (libnvinfer libnvonnxparsers-dev)
- OpenCV


## Tools / Examples

Various **documented** examples can be found in the <examples> directory.

In order to **build** a TensorRT engine based on an ONNX model, the following
tool/example is available:
- [build_engine](examples/builder): build a TensorRT engine based on your ONNX model


For **object detection**, the following tools/examples are available:
- [process_image](examples/image): detect objects in a single image
- [process_live](examples/live): detect objects live in a video stream (e.g. webcam)
- [process_batch](examples/batch): detect objects in multiple images (batch inference)


### Example Usage
To get started quickly, build the library (using the steps below) and use the build_engine and process_image tools as following
```bash
./build_engine --input INPUT_ONNX --output yolov5.engine
./process_image --engine yolov5.engine --input INPUT_IMAGE --output OUTPUT_IMAGE
```
where you replace "INPUT_ONNX" with the path to your ONNX model, set "INPUT_IMAGE" to your input image, and "OUTPUT_IMAGE" to the desired output


## Code example

Build the TensorRT engine in just three lines of C++ code:
```cpp
yolov5::Builder builder;
builder.init();
builder.build("yolov5.onnx", "yolov5.engine");
```

Next, efficiently detect objects using YoloV5 in six lines of C++ code:
```cpp
yolov5::Detector detector;
detector.init();
detector.loadEngine("yolov5.engine");

cv::Mat image = cv::imread("image.png");

std::vector<yolov5::Detection> detections;
detector.detect(image, &detections);
```


## Building the library

The software can be compiled using CMake and a modern C++ compiler (e.g. GCC)
with support for C++14, using the following steps:

```bash
mkdir build
cd build
cmake ..
make
```


## Citing

If you like this library and would like to cite it, please use the following:

```tex
@misc{yolov5tensorrt,
  author       = {van der Meer, Noah and van Hoof, Charel},
  title        = {{yolov5-tensorrt}: Real-time object detection with {YOLOv5} and {TensorRT}},
  howpublished = {GitHub},
  year         = {2021},
  note         = {\url{https://github.com/noahmr/yolov5-tensorrt}}
}
```

## License

Copyright (c) 2021, Noah van der Meer

This software is licenced under the MIT license, see [LICENCE.md](LICENCE.md).
