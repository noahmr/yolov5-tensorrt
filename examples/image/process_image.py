#!/usr/bin/python

#
# Author:           Noah van der Meer
# Description:      YoloV5-TensorRT example: inference on a single image
# 
#
# Copyright (c) 2021, Noah van der Meer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to 
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# 
#

# note: import cv2 _before_ yolov5tensorrt; Otherwise it may lead to
#   issues see https://github.com/opencv/opencv/issues/14884
import cv2
import yolov5tensorrt

import argparse
import time

def main(args):

    #
    # Create the YoloV5 Detector object
    #
    detector = yolov5tensorrt.Detector()


    #       
    # Initialize the YoloV5 Detector. This should be done first, before
    # loading the engine.
    #
    # The init() method (like most of the methods) returns a result code,
    # of the type yolov5tensorrt.Result. If initialization was successfull, this
    # will be Result.SUCCESS. If unsuccessfull, it will be set to one of the
    # error codes, and you can get a description through the 
    # yolov5tensorrt.result_to_string() function.
    #
    # Note that the Detector also performs extensive logging itself,
    # so in case of failure, you will see a more detailed description of
    # the problem in the console output
    #
    r = detector.init()
    if r != yolov5tensorrt.Result.SUCCESS:
        print("init() failed:", yolov5tensorrt.result_to_string(r))
        return 1


    #
    # Load the engine from file.
    #
    r = detector.loadEngine(args.engine)
    if r != yolov5tensorrt.Result.SUCCESS:
        print("loadEngine() failed:", yolov5tensorrt.result_to_string(r))
        return 1


    #
    # Load the Class names from file, and pass these on to the Detector
    #
    if args.classes is not None:
        classes = yolov5tensorrt.Classes()
        r = classes.loadFromFile(args.classes)
        if r != yolov5tensorrt.Result.SUCCESS:
            print("classes.loadFromFile() failed:", 
                    yolov5tensorrt.result_to_string(r))
            return 1
        detector.setClasses(classes)


    #
    # Load an image from disk and store it in CPU memory.
    #
    # Note that by default, OpenCV will represent the image in BGR format.
    #
    image = cv2.imread(args.input)
    if image is None:
        print("Failed to load input image")
        return 1

    #
    # The first one/two runs of the engine typically take significantly
    # longer. To get an accurate timing for inference, first do two
    # runs. These can of course also be performed on other representative
    # images
    #
    detector.detect(image)
    detector.detect(image)


    ts = time.perf_counter()

    #
    # Detect objects in the image using the detect(...) method.
    #
    # The detect(...) method can optionally also take flags. Through these
    # flags, the type of input image can be specified specified, e.g. BGR
    # or RGB. By default, the detect(...) method assumes that the input
    # is stored as BGR. Thus while not necessary in this case, for clarity 
    # we specifically specify that the input is BGR here.
    #
    # Note that the detect() method might also fail in some
    # cases, which can be checked through the returned result code.
    #
    r, detections = detector.detect(image, flags = yolov5tensorrt.DetectorFlag.INPUT_BGR)
    if r != yolov5tensorrt.Result.SUCCESS:
        print("detect() failed:", yolov5tensorrt.result_to_string(r))
        return 1

    # timing
    duration = time.perf_counter() - ts
    print("detect() took:", duration*1000, "milliseconds")


    #
    # Visualize all of the detections
    #
    # The detections are provided in the form of yolov5tensorrt.Detection 
    # objects. These contain information regarding the location in the image,
    # confidence, and class.
    #
    magenta = (255, 51, 153)    # BGR
    for d in detections:
        yolov5tensorrt.visualizeDetection(d, image, magenta, 1.0)


    #
    # Store the visualization to disk again.
    #
    cv2.imwrite(args.output, image)

    return 0


if __name__ == '__main__':
    #
    # Handle arguments
    #
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--engine',
                        required = True,
                        dest ='engine',
                        type = str,
                        help = '[mandatory] specify the engine file')
    parser.add_argument('--input',
                        required = True,
                        dest ='input',
                        type = str,
                        help = '[mandatory] specify the input image file')
    parser.add_argument('--output',
                        required = True,
                        dest ='output',
                        type = str,
                        help = '[mandatory] specify the output image file')
    parser.add_argument('--classes',
                        dest ='classes',
                        type = str,
                        help = '[optional] specify list of class names')
    args = parser.parse_args()

    main(args)