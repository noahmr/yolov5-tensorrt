## process_live

The ```process_live``` tool takes a YOLOv5 TensorRT engine, and performs object detection on a live source such as a camera. The result is visualized through a graphical user interface (highgui).

Basic usage:
```
./process_live --engine ENGINE_FILE
```

Arguments:
- ```--engine```: path to the YOLOv5 TensorRT engine
- ```--camera```: (optional) camera index. The default is 0
- ```--classes```: (optional) path to a file containing the class names


### Class names

By default, the ```process_live``` will attach numbers representing the class to all of the detections. For instance, when using the COCO dataset, this ranges from 0 to 79. By specifying the class names corresponding to each class id, human-readable names (e.g. "car", "truck") are displayed instead.

For convenience, a file containing the class names for COCO dataset can be found [here](../coco.txt).

### Example Usage

Assuming that your YOLOv5 TensorRT engine is <em>yolov5s.engine</em>, you can detect objects using:
```
./process_live --engine yolov5s.engine
```
A graphical user interface will show up, displaying the detections live.
