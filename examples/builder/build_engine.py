#!/usr/bin/python

#
# Author:           Noah van der Meer
# Description:      YoloV5-TensorRT example: build a TensorRT engine
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

import yolov5tensorrt
import argparse

def main(args):

    precision = yolov5tensorrt.Precision.FP32
    if args.precision is not None:
        if args.precision == "fp32":
            precision = yolov5tensorrt.Precision.FP32
        elif args.precision == 'fp16':
            precision = yolov5tensorrt.Precision.FP16
        else:
            print("Invalid precision specified:", args.precision)
            return 1


    #
    # Create the YoloV5 Builder object.
    #
    builder = yolov5tensorrt.Builder()


    #       
    # Initialize the YoloV5 Builder. This should be done first, before
    # loading the engine.
    #
    # The init() method (like most of the methods) returns a result code,
    # of the type yolov5tensorrt.Result. If initialization was successfull, this
    # will be Result.SUCCESS. If unsuccessfull, it will be set to one of the
    # error codes, and you can get a description through the 
    # yolov5tensorrt.result_to_string() function.
    #
    # Note that the Builder also performs extensive logging itself,
    # so in case of failure, you will see a more detailed description of
    # the problem in the console output
    #
    r = builder.init()
    if r != yolov5tensorrt.Result.SUCCESS:
        print("init() failed:", yolov5tensorrt.result_to_string(r))
        return 1


    #
    # Build the TensorRT engine
    #
    r = builder.buildEngine(args.model, args.output, precision)
    if r != yolov5tensorrt.Result.SUCCESS:
        print("buildEngine() failed:", yolov5tensorrt.result_to_string(r))
        return 1
        

    return 0

if __name__ == '__main__':
    #
    # Handle arguments
    #
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model',
                        required = True,
                        dest ='model',
                        type = str,
                        help = '[mandatory] specify the ONNX model file')
    parser.add_argument('--output',
                        required = True,
                        dest ='output',
                        type = str,
                        help = '[mandatory] specify the engine output file')
    parser.add_argument('--precision',
                        dest ='precision',
                        type = str,
                        help = '[optional] specify the precision')
    args = parser.parse_args()

    main(args)