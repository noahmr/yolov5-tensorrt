/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5-TensorRT example: inference on a live video source
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

#include "yolov5_detector.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <chrono>

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
                "--engine :        [mandatory] specify the engine file\n"
                "--camera :        [optional] camera index\n"
                "--classes :       [optional] specify list of class names\n\n"
                "Example usage:\n"
                "process_live --engine yolov5s.engine --camera 0" << std::endl;
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

    if(!cmdOptionExists(argv, argv+argc, "--engine", true))
    {
        std::cout << "Missing mandatory argument" << std::endl;
        printHelp();
        return 1;
    }
    const std::string engineFile(getCmdOption(argv, argv+argc, "--engine"));

    int cameraIndex = 0;
    if(cmdOptionExists(argv, argv+argc, "--camera", true))
    {
        const std::string option = 
                getCmdOption(argv, argv+argc, "--camera");
        cameraIndex = std::atoi(option.c_str());
    }

    std::string classesFile;
    if(cmdOptionExists(argv, argv+argc, "--classes", true))
    {
        classesFile = getCmdOption(argv, argv+argc, "--classes");
    }


    /*
        Create the YoloV5 Detector object.
    */
    yolov5::Detector detector;


    /*
        Initialize the YoloV5 Detector. This should be done first, before
        loading the engine.
    */
    yolov5::Result r = detector.init();
    if(r != yolov5::RESULT_SUCCESS)
    {
        std::cout << "init() failed: " << yolov5::result_to_string(r) 
                    << std::endl;
        return 1;
    }


    /*
        Load the engine from file.
    */
    r = detector.loadEngine(engineFile);
    if(r != yolov5::RESULT_SUCCESS)
    {
        std::cout << "loadEngine() failed: " << yolov5::result_to_string(r)
                    << std::endl;
        return 1;
    }

    
    /*
        Load the Class names from file, and pass these on to the Detector
    */
    if(classesFile.length() > 0)
    {
        yolov5::Classes classes;
        classes.setLogger(detector.logger());
        r = classes.loadFromFile(classesFile);
        if(r != yolov5::RESULT_SUCCESS)
        {
            std::cout << "classes.loadFromFile() failed: " 
                    << yolov5::result_to_string(r) << std::endl;
            return 1;
        }
        detector.setClasses(classes);
    }


    /*
        Set up the GUI
    */
    cv::namedWindow("live");


    /*
        Set up the Camera
    */
    cv::VideoCapture capture;
    if(!capture.open(cameraIndex, cv::CAP_ANY))
    {
        std::cout << "failure: could not open capture device" << std::endl;
        return 1;
    }

    /*
        Start Inference
    */
    cv::Mat image;
    std::vector<yolov5::Detection> detections;
    while(true)
    {
        if(!capture.read(image))
        {
            std::cout << "failure: could not read new frames" << std::endl;
            break;
        }

        r = detector.detect(image, &detections, yolov5::INPUT_BGR);
        if(r != yolov5::RESULT_SUCCESS)
        {
            std::cout << "detect() failed: " << yolov5::result_to_string(r) 
                        << std::endl;
            return 1;
        }

        /*
            Visualize the detections
        */
        for(unsigned int i = 0; i < detections.size(); ++i)
        {
            const cv::Scalar magenta(255, 51, 153); /*  BGR */
            yolov5::visualizeDetection(detections[i], &image, magenta, 1.0);
        }
        cv::imshow("live", image);

        cv::waitKey(1);
    }
    capture.release();

    cv::destroyAllWindows();

    return 0;
}