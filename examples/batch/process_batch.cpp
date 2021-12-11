/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5-TensorRT example: inference on a batch of images
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

#include <dirent.h>     

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

bool listFiles(const std::string& directory, std::vector<std::string>* out)
{
    DIR* dir = opendir(directory.c_str());
    if(dir == nullptr)
    {
        std::cout << "listFiles() failed, could not open directory: "
                << std::strerror(errno) << std::endl;
        return false;
    }

    struct dirent* i = readdir(dir);
    while(i != nullptr)
    {
        const std::string name(i->d_name);
        if(name != "." && name != "..")
        {
            std::cout << "Input image: " << name << std::endl;
            out->push_back(name);
        }
        i = readdir(dir);
    }
    return true;
}

void printHelp()
{
    std::cout << "Options:\n"
                "-h --help :       show this help menu\n"
                "--engine :        [mandatory] specify the engine file\n"
                "--inputs :        [mandatory] specify the input directory\n"
                "--outputs :       [mandatory] specify the output directory\n"
                "--classes :       [optional] specify list of class names\n\n"
                "Example usage:\n"
                "process_batch --engine yolov5s.engine --inputs input_dir "
                "--outputs output_dir" << std::endl;
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
        !cmdOptionExists(argv, argv+argc, "--inputs", true) ||
        !cmdOptionExists(argv, argv+argc, "--outputs", true))
    {
        std::cout << "Missing mandatory argument" << std::endl;
        printHelp();
        return 1;
    }
    const std::string engineFile(getCmdOption(argv, argv+argc, "--engine"));
    const std::string inputDir(getCmdOption(argv, argv+argc, "--inputs"));
    const std::string outputDir(getCmdOption(argv, argv+argc, "--outputs"));

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
        Initialize the YoloV5 Detector.
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
        List all files in the specified directory
    */
    std::vector<std::string> filenames;
    if(!listFiles(inputDir, &filenames))
    {
        return 1;
    }
    std::cout << "Found " << filenames.size() << " files in specified input "
            << "directory" << std::endl;

    /*
        Load the images from disk and store in CPU memory.
    */
    std::vector<cv::Mat> images;
    for(unsigned int i = 0; i < filenames.size(); ++i)
    {
        cv::Mat image;
        try
        {
            image = cv::imread(inputDir + "/" + filenames[i]);
        }
        catch(const std::exception& e)
        {
            std::cout << "Could not load image " << filenames[i] 
                        << " . Exception: " << e.what() << std::endl;
            return 1;
        }

        if(!image.empty())
        {
            images.push_back(image);
        }
        else
        {
            std::cout << "Could not load image: empty image " 
                        << filenames[i] << std::endl;
            return 1;
        }
    }


    /*
        The first one/two runs of the engine typically take significantly
        longer. To get an accurate timing for inference, first do two
        runs. These can of course also be performed on other representative
        images
    */
    detector.detectBatch(images, nullptr);
    detector.detectBatch(images, nullptr);


    auto ts = std::chrono::high_resolution_clock::now();    /*  timing  */

    /*
        Detect objects in the images using the detectBatch(...) method. The
        detections are inserted into the 'detections' vector.
    */
    std::vector<std::vector<yolov5::Detection>> detections;
    r = detector.detectBatch(images, &detections, yolov5::INPUT_BGR);
    if(r != yolov5::RESULT_SUCCESS)
    {
        std::cout << "detectBatch() failed: " << yolov5::result_to_string(r) 
                    << std::endl;
        return 1;
    }

    /*  timing  */
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - ts);
    std::cout << "detectBatch() took: " << duration.count() << "ms" 
                << std::endl;


    /*
        Visualize all of the detections & store to disk
    */
    cv::Mat visualization;
    const cv::Scalar magenta(255, 51, 153); /*  BGR */
    for(unsigned int i = 0; i < detections.size(); ++i)
    {
        images[i].copyTo(visualization);

        const std::vector<yolov5::Detection>& lst = detections[i];
        for(unsigned int j = 0; j < lst.size(); ++j)
        {
            yolov5::visualizeDetection(lst[j], &visualization, magenta, 1.0);
        }

        /*
            Store the visualization to disk again.
        */
        const std::string outputName = outputDir + "/" + filenames[i];
        try
        {
            cv::imwrite(outputName, visualization);
        }
        catch(const std::exception& e)
        {
            std::cout << "Warning: could not save image '" << outputName 
                        << "'. Exception: " << e.what() << std::endl;
        }
    }

    return 0;
}