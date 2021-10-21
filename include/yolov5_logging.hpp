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

/*  include guard   */
#ifndef _YOLOV5_LOGGING_HPP_
#define _YOLOV5_LOGGING_HPP_

#include "yolov5_common.hpp"

#include <memory>

#include <NvInfer.h>



namespace yolov5
{

enum LogLevel
{
    LOGGING_DEBUG   = 0,        /**<  verbose, low-level details   */
    
    LOGGING_INFO    = 1,        /**<  informational messages  */

    LOGGING_WARNING = 2,        /**<  warning messages    */

    LOGGING_ERROR   = 3         /**<  error messages  */
};

/**
 * @brief           Convert the LogLevel to string
 * 
 * If the specified value 'l' is not a valid YoloV5 loglevel, an empty
 * string is returned. Note that the methods and functions of this library
 * always use proper loglevels when logging.
 * 
 * Outputs:
 * - LOGGING_DEBUG:     "debug"
 * - LOGGING_INFO:      "info"
 * - LOGGING_WARNING:   "warning"
 * - LOGGING_ERROR:     "error"
 * 
 * @param l         Log level
 *
 * @return          Log level string
 */
const char* loglevel_to_string(const LogLevel& l) noexcept;


/**
 * @brief           Convert the LogLevel to string
 * 
 * See yolov5_loglevel_to_string(YoloV5_LogLevel) for more information.
 * 
 * If 'r' is not a valid loglevel, False is returned and 'out' is left
 * untouched. Note that the methods and functions of this library
 * always use proper loglevels when logging.
 * 
 * 
 * @param l         Log level
 * @param out       Output. Can be nullptr
 *
 * @return          True on success, False otherwise
 */
bool loglevel_to_string(const LogLevel& l, std::string* out) noexcept;


/**
 * The main logger used in the yolov5-tensorrt library
 *
 * You can use this class to integrate the yolov5-tensorrt logging into
 * your own preferred logging facilities. To do so, create your own
 * class that inherits from this class, and override the print(...)
 * method.
 */
class Logger
{
public:
    Logger() noexcept;


    virtual ~Logger();

public:

    /**
     * @brief                   Print/Log a message. Override this method to
     *                          integrate logging with your own preferred
     *                          logging mechanism.
     * 
     * The default implementation prints all messages to stdout, and appends
     * a newline at the end of all messages.
     * 
     * 
     * This method is not marked as 'noexcept' intentionally, in case a user
     * does not (want to)deal with exceptions properly in a derived class.
     * 
     * @param level             Logging level
     * @param msg               Message to be printed.
     */
    virtual void print(const LogLevel& level, const char* msg);


    /**
     * @brief                   Log a message
     * 
     * Internally, this method will forward the message to print().
     * 
     * @param level             Logging level
     * @param msg               Message to be printed
     */
    void log(const LogLevel& level, const char* msg) noexcept;


    /**
     * @brief                   Log a formatted message
     * 
     * Internally, this method will forward the message to print().
     * 
     * @param level             Logging level
     * @param fmt               Format string
     */
    void logf(const LogLevel& level, const char* fmt, ...) noexcept
                        __attribute__ ((format (printf, 3, 4)));
    
private:
};


/**
 * Logger used to integrate TensorRT and yolov5-tensorrt logging
 *
 * This logger forwards all messages from the TensorRT logger
 * to a yolov5::Logger.
 * 
 * Normally, it is not necessary for a user of the library to worry about
 * this class, unless you are using TensorRT in other places as well and wish
 * to integrate logging further.
 */
class TensorRT_Logger : public nvinfer1::ILogger
{
public:
    TensorRT_Logger() noexcept;


    /**
     * @brief                   Construct a new TensorRT_Logger object
     * 
     * @param logger            Pointer to Logger. Can be nullptr
     */
    TensorRT_Logger(std::shared_ptr<Logger> logger) noexcept;


    ~TensorRT_Logger();
public:

    /**
     * @brief                   Set the YoloV5-TensorRT logger
     * 
     * @param logger            Pointer to Logger. Can be nullptr
     */
    void setLogger(std::shared_ptr<Logger> logger) noexcept;


    virtual void log(nvinfer1::ILogger::Severity severity, 
                        const char* msg) noexcept override;

private:
    std::shared_ptr<Logger>      _logger;
};

}   /*  namespace yolov5    */

#endif  /*  include guard   */