## build_engine

The ```build_engine``` tool takes the YOLOv5 model expressed in ONNX form, and creates a TensorRT engine based on it. This TensorRT engine is optimized for the
specific hardware being used, and can be used repeadetly for object detection. Note that you can export your trained YOLOv5 model to ONNX using the [official YOLOv5 guide](https://github.com/ultralytics/yolov5/issues/251).

Basic usage:
```
./build_engine --model ONNX_MODEL --output OUTPUT_ENGINE
```

Arguments:
- ```--model```: path to the input ONNX model
- ```--output```: path at which the output engine should be written
- ```--precision```: (optional) the precision that should be used for the engine. The available options are "fp32" and "fp16". This argument is optional, and if it is not specified, the default "fp32" is used


### Example Usage

Assuming that your YOLOv5 model is <em>yolov5s.onnx</em> (i.e. the S variant of YOLOv5), you can build an engine with FP16 inference as following:
```
./build_engine --model yolov5s.onnx --output yolov5s.engine --precision fp16
```
The resulting engine will be stored to disk as <em>yolov5s.engine</em>. After this, you may perform inference using one of the other tools, such as [process_image](examples/image).
