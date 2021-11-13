/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5 inference through TensorRT
 * 
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

#include "yolov5_detection.hpp"

#include <fstream>

namespace yolov5
{

Detection::Detection() noexcept : _classId(-1), _score(0)
{
}

Detection::Detection(const int& classId, 
        const cv::Rect& boundingBox, const double& score) noexcept
    : _classId(classId), _boundingBox(boundingBox), _score(score)
{
}

Detection::~Detection() noexcept
{
}

const int32_t& Detection::classId() const noexcept
{
    return _classId;
}

const cv::Rect& Detection::boundingBox() const noexcept
{
    return _boundingBox;
}

const double& Detection::score() const noexcept
{
    return _score;
}

const std::string& Detection::className() const noexcept
{
    return _className;
}

bool Detection::setClassName(const std::string& name) noexcept
{
    try
    {
        _className = name;
    }
    catch(const std::exception& e)
    {
        return false;
    }
    return true;
}

Result visualizeDetection(const Detection& detection, cv::Mat* image,
                            const cv::Scalar& color, 
                            const double& fontScale) noexcept
{
    if(image == nullptr)
    {
        return RESULT_SUCCESS;
    }

    try
    {
        /*  Draw bounding box around the detection  */
        const int bboxThickness = 2;
        const cv::Rect& bbox = detection.boundingBox();
        cv::rectangle(*image, bbox, color, bboxThickness);

        /*  "className: score"

            or alternatively "classId: score"
            if no class name is known

            display the score with 2 decimal places
        */
        std::string className = detection.className();
        if(className.length() == 0)
        {
            className = std::to_string(detection.classId());
        }
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << detection.score();
        const std::string label = className + ": " + ss.str();

        /*  Draw a rectangle above the bounding box, in which the
            label will be written   */
        const int textThickness = 1;

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(label, 
                                        cv::FONT_HERSHEY_PLAIN,
                                        fontScale, textThickness, &baseline);
        const cv::Point tl(bbox.x - bboxThickness/2.0, bbox.y-textSize.height);
        const cv::Rect labelRect(tl, textSize);
        cv::rectangle(*image, labelRect, color, -1);  /*  filled rectangle */

        /*  white text on top of the previously drawn rectangle */
        const cv::Point bl(tl.x, bbox.y - bboxThickness/2.0);
        cv::putText(*image, label, bl, cv::FONT_HERSHEY_PLAIN,
                        fontScale, cv::Scalar(255, 255, 255), textThickness);
    }
    catch(const std::exception& e)
    {
        return RESULT_FAILURE_OPENCV_ERROR;
    }
    return RESULT_SUCCESS;
}

Classes::Classes() noexcept
{
}

Classes::~Classes() noexcept
{
}

Result Classes::load(const std::vector<std::string>& names) noexcept
{
    if(names.size() == 0 && _logger)
    { 
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Classes] load() warning: specified "
                                "list of class names is empty!");
        }
        return RESULT_FAILURE_INVALID_INPUT;
    }

    try
    {
        _names = names;
    }
    catch(const std::exception& e)
    {
        if(_logger)
        {
            _logger->logf(LOGGING_ERROR, "[Classes] load() failure: got "
                        "exception trying to copy names: %s", e.what());
        }
        return RESULT_FAILURE_ALLOC;
    }

    if(_logger)
    {
        _logger->logf(LOGGING_INFO, "[Classes] Loaded %d classes", 
                        (unsigned int)names.size());
    }
    return RESULT_SUCCESS;
}

Result Classes::loadFromFile(const std::string& filepath) noexcept
{
    std::vector<std::string> names;

    std::ifstream file(filepath, std::ios::binary);
    if(!file.good())
    {
        if(_logger)
        {
            _logger->logf(LOGGING_ERROR, "[Classes] loadFromFile() failure: "
                    "could not open file '%s'", filepath.c_str());
        }
        return RESULT_FAILURE_FILESYSTEM_ERROR;
    }    

    try
    {
        std::string line;
        while(std::getline(file, line))
        {
            if(line.length() > 0)
            {
                names.push_back(line);
            }
        }
    }
    catch(const std::exception& e)
    {
        if(_logger)
        {
            _logger->logf(LOGGING_ERROR, "[Classes] loadFromFile() failure: "
                        "got exception while reading classes from file: %s",
                        e.what());
        }
        return RESULT_FAILURE_ALLOC;
    }
    file.close();

    if(names.size() == 0)
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Classes] loadFromFile() failure: "
                            "could not load any classes");
        }
        return RESULT_FAILURE_OTHER;
    }

    names.swap(_names);
    if(_logger)
    {
        _logger->logf(LOGGING_INFO, "[Classes] Loaded %d classes",
                        (unsigned int)_names.size());
    }
    return RESULT_SUCCESS;
}

bool Classes::isLoaded() const noexcept
{
    return (_names.size() > 0);
}

Result Classes::getName(const int& classId, std::string* out) const noexcept
{
    if((unsigned int)classId >= _names.size() || classId < 0)
    {
        if(_logger)
        {
            _logger->logf(LOGGING_ERROR, "[Classes] getName() failure: no "
                        "info about specified classId '%i'", classId);
        }
        return RESULT_FAILURE_INVALID_INPUT;
    }

    if(out != nullptr)
    {
        try
        {
            *out = _names[classId];
        }
        catch(const std::exception& e)
        {
            if(_logger)
            {
                _logger->logf(LOGGING_ERROR, "[Classes] getName() failure: got"
                            " exception when setting output: %s", e.what());
            }
            return RESULT_FAILURE_ALLOC;
        }
    }
    return RESULT_SUCCESS;
}

void Classes::setLogger(std::shared_ptr<Logger> logger) noexcept
{
    _logger = logger;
}

}   /*  namespace yolov5    */