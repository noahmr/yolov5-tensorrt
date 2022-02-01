#!/usr/bin/python

#
# Author:           Noah van der Meer
# Description:      YoloV5-TensorRT example: inference on a live video source
# 
#
# Copyright (c) 2020, Noah van der Meer
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
    # Set up the GUI 
    #
    cv2.namedWindow("live")


    #
    # Set up the Camera 
    #
    capture = cv2.VideoCapture()
    if not capture.open(args.camera, cv2.CAP_ANY):
        print("failure: could not open capture device")
        return 1


    #
    # Start Inference
    #
    while True:

        ret, image = capture.read()
        if not ret:
            print("failure: could not read new frames")
            break

        r, detections = detector.detect(image, flags = yolov5tensorrt.DetectorFlag.INPUT_BGR)
        if r != yolov5tensorrt.Result.SUCCESS:
            print("detect() failed:", yolov5tensorrt.result_to_string(r))
            return 1
        
        #
        # Visualize the detections
        #
        magenta = (255, 51, 153)    # BGR
        for d in detections:
            yolov5tensorrt.visualizeDetection(d, image, magenta, 1.0)
        cv2.imshow("live", image)

        cv2.waitKey(1)

    capture.release()

    cv2.destroyAllWindows()

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
    parser.add_argument('--camera',
                        dest ='camera',
                        type = int,
                        default = 0,
                        help = '[optional] camera index')
    parser.add_argument('--classes',
                        dest ='classes',
                        type = str,
                        help = '[optional] specify list of class names')
    args = parser.parse_args()

    main(args)