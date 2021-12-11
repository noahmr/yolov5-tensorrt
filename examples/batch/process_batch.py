#!/usr/bin/python

#
# Author:           Noah van der Meer
# Description:      YoloV5-TensorRT example: inference on a batch of images
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
import os

def main(args):

    #
    # Create the YoloV5 Detector object
    #
    detector = yolov5tensorrt.Detector()


    #       
    # Initialize the YoloV5 Detector.
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
    # List all files in the specified directory
    #
    filenames = os.listdir(args.inputs)
    print("Found", len(filenames), "files in specified input directory")


    images = []
    for f in filenames:
        image = cv2.imread(args.inputs + "/" + f)
        if image is not None:
            images.append(image)
        else:
            print("Could not load file ", f)
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
    # Detect objects in the images using the detectBatch(...) method.
    #
    r, detections = detector.detectBatch(images)
    if r != yolov5tensorrt.Result.SUCCESS:
        print("detectBatch() failed:", yolov5tensorrt.result_to_string(r))
        return 1

    # timing
    duration = time.perf_counter() - ts
    print("detectBatch() took:", duration*1000, "milliseconds")


    #
    # Visualize all of the detections & store to disk
    #
    magenta = (255, 51, 153)    # BGR
    visualization = []
    for i in range(0, len(detections)):
        img = images[i]

        lst = detections[i]
        for d in lst:
            yolov5tensorrt.visualizeDetection(d, img, magenta, 1.0)

        #
        # Store the visualization to disk again.
        #
        outputName = args.outputs + "/" + filenames[i]
        cv2.imwrite(outputName, img)

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
    parser.add_argument('--inputs',
                        required = True,
                        dest ='inputs',
                        type = str,
                        help = '[mandatory] specify the input directory')
    parser.add_argument('--outputs',
                        required = True,
                        dest ='outputs',
                        type = str,
                        help = '[mandatory] specify the output directory')
    parser.add_argument('--classes',
                        dest ='classes',
                        type = str,
                        help = '[optional] specify list of class names')
    args = parser.parse_args()

    main(args)