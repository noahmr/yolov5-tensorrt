/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5-TensorRT example: inference on a single image
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
                "--input :         [mandatory] specify the input image file\n"
                "--output :        [mandatory] specify the output image file\n"
                "--classes :       [optional] specify list of class names\n\n"
                "Example usage:\n"
                "process_image --engine yolov5s.engine --input test_image.png "
                "--output result.png" << std::endl;
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

    if(!cmdOptionExists(argv, argv+argc, "--engine", true) || 
        !cmdOptionExists(argv, argv+argc, "--input", true) ||
        !cmdOptionExists(argv, argv+argc, "--output", true))
    {
        std::cout << "Missing mandatory argument" << std::endl;
        printHelp();
        return 1;
    }
    const std::string engineFile(getCmdOption(argv, argv+argc, "--engine"));
    const std::string inputFile(getCmdOption(argv, argv+argc, "--input"));
    const std::string outputFile(getCmdOption(argv, argv+argc, "--output"));

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

        The init() method (like most of the methods) returns a result code,
        of the type yolov5::Result. If initialization was successfull, this
        will be RESULT_SUCCESS. If unsuccessfull, it will be set to one of the
        error codes, and you can get a description through the 
        yolov5_result_to_string() function.

        Note that the Detector also performs extensive logging itself,
        so in case of failure, you will see a more detailed description of
        the problem in the console output
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
        Load an image from disk and store it in CPU memory.

        Note that by default, OpenCV will represent the image in BGR
        format.
    */
    cv::Mat image;
    try
    {
        image = cv::imread(inputFile);
    }
    catch(const std::exception& e)
    {
        std::cout << "Failed to load input image: " << e.what() << std::endl;
        return 1;   /*  fatal   */
    }
    if(image.empty())
    {
        std::cout << "Failed to load input image" << std::endl;
        return 1;
    }


    /*
        The first one/two runs of the engine typically take significantly
        longer. To get an accurate timing for inference, first do two
        runs. These can of course also be performed on other representative
        images
    */
    detector.detect(image, nullptr);
    detector.detect(image, nullptr);


    auto ts = std::chrono::high_resolution_clock::now();    /*  timing  */

    /*
        Detect objects in the image using the detect(...) method. The
        detections are inserted into the 'detections' vector.

        The detect(...) method can optionally also take flags. Through these
        flags, the type of input image can be specified specified, e.g. BGR
        or RGB. By default, the detect(...) method assumes that the input
        is stored as BGR. Thus while not necessary in this case, for clarity 
        we specifically specify that the input is BGR here.

        Note that the detect() method might also fail in some
        cases, which can be checked through the returned result code.
    */
    std::vector<yolov5::Detection> detections;
    r = detector.detect(image, &detections, yolov5::INPUT_BGR);
    if(r != yolov5::RESULT_SUCCESS)
    {
        std::cout << "detect() failed: " << yolov5::result_to_string(r) 
                    << std::endl;
        return 1;
    }

    /*  timing  */
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - ts);
    std::cout << "detect() took: " << duration.count() << "ms" << std::endl;


    /*
        Visualize all of the detections

        The detections are provided in the form of yolov5::Detection 
        objects. These contain information regarding the location in the image,
        confidence, and class.
    */
    const cv::Scalar magenta(255, 51, 153); /*  BGR */
    for(unsigned int i = 0; i < detections.size(); ++i)
    {
        yolov5::visualizeDetection(detections[i], &image, magenta, 1.0);
    }


    /*
        Store the visualization to disk again.
    */
    try
    {
        cv::imwrite(outputFile, image);
    }
    catch(const std::exception& e)
    {
        std::cout << "Failed to write output image: " << e.what() << std::endl;
    }

    return 0;
}