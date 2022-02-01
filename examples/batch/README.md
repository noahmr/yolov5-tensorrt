## process_batch

The ```process_batch``` tool loads a YOLOv5 TensorRT engine, and performs object detection on a batch of images at once, i.e. <em>batch inference</em>. Note that the YOLOv5 model should have been exported with an explicit batch size.

Basic usage:
```
./process_batch --engine ENGINE_FILE --inputs INPUT_DIRECTORY --outputs OUTPUT_DIRECTORY
```

Arguments:
- ```--engine```: path to the YOLOv5 TensorRT engine
- ```--inputs```: path to the directory in which the images are stored
- ```--outputs```: path to the directory in which the results should be written (directory should exist already)
- ```--classes```: (optional) path to a file containing the class names


### Class names

By default, the ```process_batch``` program will attach numbers representing the class to all of the detections. For instance, when using the COCO dataset, this ranges from 0 to 79. By specifying the class names corresponding to each class id, human-readable names (e.g. "car", "truck") are displayed instead.

For convenience, a file containing the class names for COCO dataset can be found [here](../coco.txt).

### Example Usage

Given that your YOLOv5 TensorRT engine is <em>yolov5s.engine</em> and your input images are stored in directory <em>input_images</em>, you can detect objects using:
```
./process_batch --engine yolov5s.engine --inputs input_images --outputs output_images
```
Visualizations will be stored to disk in the directory <em>output_images</em>.
