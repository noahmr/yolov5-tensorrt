/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5 inference through TensorRT (logging)
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

#include "yolov5_logging.hpp"

#include <cstring>
#include <cstdarg>

const char* yolov5::loglevel_to_string(const LogLevel& l) noexcept
{
    if(l == LOGGING_DEBUG)
    {
        return "debug";
    }
    else if(l == LOGGING_INFO)
    {
        return "info";
    }
    else if(l == LOGGING_WARNING)
    {
        return "warning";
    }
    else if(l == LOGGING_ERROR)
    {
        return "error";
    }
    else
    {
        return "";
    }
}

bool yolov5::loglevel_to_string(const LogLevel& l, std::string* out) noexcept
{
    const char* str = loglevel_to_string(l);
    if(std::strlen(str) == 0)
    {
        return false;
    }

    if(out != nullptr)
    {
        try
        {
            *out = std::string(str);
        }
        catch(const std::exception& e)
        {
            return false;
        }
    }
    return true;
}

yolov5::Logger::Logger() noexcept
{
}

yolov5::Logger::~Logger()
{
}

void yolov5::Logger::print(const LogLevel& level, const char* msg)
{
    std::printf("|yolov5|%s|%s\n", loglevel_to_string(level), msg);
}

void yolov5::Logger::log(const LogLevel& level, const char* msg) noexcept
{
    try
    {
        this->print(level, msg);
    }
    catch(...)
    {
    }
}

void yolov5::Logger::logf(const LogLevel& level, 
                        const char* fmt, ...) noexcept
{
    char* msg = nullptr;

    va_list args;
    va_start(args, fmt);

    if(vasprintf(&msg, fmt, args) < 0)
    {
        msg = nullptr;
    }
    va_end(args);

    if(msg)
    {
        try
        {
            this->print(level, msg);
        }
        catch(...)
        {
        }
        std::free(msg);
    }
}

yolov5::TensorRT_Logger::TensorRT_Logger() noexcept
{
}

yolov5::TensorRT_Logger::TensorRT_Logger(
        std::shared_ptr<Logger> logger) noexcept
    : _logger(logger)   /*  shared_ptr copy constructor is marked noexcept  */
{
}

yolov5::TensorRT_Logger::~TensorRT_Logger()
{
}

void yolov5::TensorRT_Logger::setLogger(
        std::shared_ptr<Logger> logger) noexcept
{
    _logger = logger;
}

void yolov5::TensorRT_Logger::log(nvinfer1::ILogger::Severity severity, 
                        const char* msg) noexcept
{
    if(!_logger)
    {
        return;
    }

    LogLevel level = LOGGING_DEBUG;
    if(severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR ||
        severity == nvinfer1::ILogger::Severity::kERROR)
    {
        level = LOGGING_ERROR;
    }
    else if(severity == nvinfer1::ILogger::Severity::kWARNING)
    {
        level = LOGGING_WARNING;
    }
    else if(severity == nvinfer1::ILogger::Severity::kINFO)
    {
        level = LOGGING_INFO;
    }
    else if(severity == nvinfer1::ILogger::Severity::kVERBOSE)
    {
        /*  LOGGING_DEBUG   */
    }


    try
    {
        _logger->logf(level, "[TensorRT] %s", msg);
    }
    catch(...)
    {
    }
}