## process_image

The ```process_image``` tool takes a YOLOv5 TensorRT engine, and performs object detection on on a single input image. The tool measures the time the overall detection process takes (i.e. pre-processing, inference, post-processing together) and prints the result.

Basic usage:
```
./process_image --engine ENGINE_FILE --input INPUT_IMAGE --output OUTPUT_IMAGE
```

Arguments:
- ```--engine```: path to the YOLOv5 TensorRT engine
- ```--input```: path to the input image
- ```--output```: path at which the visualized output image should be written
- ```--classes```: (optional) path to a file containing the class names


### Class names

By default, the ```process_image``` will attach numbers representing the class to all of the detections. For instance, when using the COCO dataset, this ranges from 0 to 79. By specifying the class names corresponding to each class id, human-readable names (e.g. "car", "truck") are displayed instead.

For convenience, a file containing the class names for COCO dataset can be found [here](../coco.txt).

### Example Usage

Assuming that your YOLOv5 TensorRT engine is <em>yolov5s.engine</em> and your input image is <em>image.png</em>, you can detect objects using:
```
./process_image --engine yolov5s.engine --input image.png --output result.png
```
A visualization will be stored to disk as <em>result.png</em>.
