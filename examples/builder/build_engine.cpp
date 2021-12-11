/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5-TensorRT example: build a TensorRT engine
 * 
 * Copyright (c) 2021, Noah van der Meer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to 
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 * 
 */

#include "yolov5_builder.hpp"

#include <iostream>
#include <algorithm>


char* getCmdOption(char** begin, char** end, const std::string& option)
{
    /*  From https://stackoverflow.com/questions/865668/parsing-
        command-line-arguments-in-c */
    char** itr = std::find(begin, end, option);
    if(itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option, 
                    bool value = false)
{
    /*  From https://stackoverflow.com/questions/865668/parsing-
        command-line-arguments-in-c */
    char** itr = std::find(begin, end, option);
    if(itr == end)
    {
        return false;
    }
    if(value && itr == end-1)
    {
        std::cout << "Warning: option '" << option << "'"
                << " requires a value" << std::endl;
        return false;
    }
    return true;
}

void printHelp()
{
    std::cout << "Options:\n"
                "-h --help :       show this help menu\n"
                "--model :          [mandatory] specify the ONNX model file\n"
                "--output :         [mandatory] specify the engine output "
                "file\n"
                "--precision :      [optional] specify the precision. "
                "Options: fp32, fp16\n\n"
                "Example usage:\n"
                "build_engine --model yolov5.onnx --output yolov5.engine "
                "--precision fp32" << std::endl;
}

int main(int argc, char* argv[])
{
    /*  
        Handle arguments
    */
    if(cmdOptionExists(argv, argv+argc, "--help") || 
        cmdOptionExists(argv, argv+argc, "-h"))
    {
        printHelp();
        return 0;
    }

    if(!cmdOptionExists(argv, argv+argc, "--model", true) ||
        !cmdOptionExists(argv, argv+argc, "--output", true))
    {
        std::cout << "Missing mandatory argument" << std::endl;
        printHelp();
        return 1;
    }

    const std::string modelFile(getCmdOption(argv, argv+argc, "--model"));
    const std::string outputFile(getCmdOption(argv, argv+argc, "--output"));
    
    yolov5::Precision precision = yolov5::PRECISION_FP32;   /*  default */
    if(cmdOptionExists(argv, argv+argc, "--precision", true))
    {
        const std::string s = getCmdOption(argv, argv+argc, "--precision");
        if(s == "fp32")
        {
        }
        else if(s == "fp16")
        {
            precision = yolov5::PRECISION_FP16;
        }
        else
        {
            std::cout << "Invalid precision specified: " << s << std::endl;
            printHelp();
            return 1;
        }
    }



    /*
        Create the YoloV5 Builder object.
    */
    yolov5::Builder builder;


    /*
        Initialize the YoloV5 Builder. This should be done first, before
        building the engine.

        The init() method (like most of the methods) returns a result code,
        of the type yolov5::Result. If initialization was successfull, this
        will be RESULT_SUCCESS. If unsuccessfull, it will be set to one of the
        error codes, and you can get a description through the 
        yolov5::result_to_string() function.

        Note that the Builder also performs extensive logging itself,
        so in case of failure, you will see a more detailed description of
        the problem in the console output
    */
    yolov5::Result r = builder.init();
    if(r != yolov5::RESULT_SUCCESS)
    {
        std::cout << "init() failed: " << yolov5::result_to_string(r) 
                    << std::endl;
        return 1;
    }


    /*
        Build the TensorRT engine
    */
    r = builder.buildEngine(modelFile, outputFile, precision);
    if(r != yolov5::RESULT_SUCCESS)
    {
        std::cout << "buildEngine() failed: " << yolov5::result_to_string(r) 
                    << std::endl;
        return 1;
    }

    return 0;
}