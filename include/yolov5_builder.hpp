/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5 inference through NVIDIA TensorRT (builder)
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
#ifndef _YOLOV5_BUILDER_HPP_
#define _YOLOV5_BUILDER_HPP_

#include <memory>
#include <vector>
#include <cstring>

#include "yolov5_logging.hpp"

namespace yolov5
{

/**
 * Build the YoloV5 TensorRT engine, which can be used for detection
 * 
 * Before building a TensorRT engine, you should first initialize the
 * builder by using the init() method.
 * 
 * 
 * ### Basic usage example
 *  yolov5::Builder builder;
 *  builder.init();
 *  builder.buildEngine("yolov5.onnx", "yolov5.engine");
 */
class Builder
{
public:
	/// ***
	///	Constructor / Destructor
    /// ***

    Builder() noexcept;


    ~Builder();

public:
	/// ***
	///	Initialization
    /// ***

    /**
     * @brief                   Initialize the Builder.
     * 
     * @return                  Result code
     */
    Result init() noexcept;



    /// ***
	///	Building
    /// ***

    /**
     * @brief                   Build an engine from ONNX model input, save it
     *                          to disk
     * 
     * The Builder should have been initialized already through
     * the init() method.
     * 
     * 
     * @param inputFilePath     Path to ONNX model
     * @param outputFilePath    Path where output (engine) should be written
     *
     * @param precision         (optional) Desired precision
     * 
     * @return                  Result code
     */
    Result buildEngine(const std::string& inputFilePath,
                        const std::string& outputFilePath,
                        Precision precision = PRECISION_FP32)
                        const noexcept;
    
    /**
     * @brief                   Build an engine from ONNX model input, store in
     *                          memory
     * 
     * The Builder should have been initialized already through
     * the init() method.  
     * 
     * 
     * @param inputFilePath     Path to ONNX model
     * @param output            Output data
     * 
     * @param precision         (optional) Desired precision
     * 
     * @return Result           Result code
     */
    Result buildEngine(const std::string& inputFilePath,
                        std::vector<char>* output,
                        Precision precision = PRECISION_FP32)
                        const noexcept;


    /// ***
	///	Logging
    /// ***
    
    /**
     * @brief                   Set the logger to be used by the Builder
     * 
     * Note that you can potentially use this method _before_ initializing
     * the Builder.
     * 
     * @param logger            New logger; Should NOT be a nullptr
     * 
     * @return                  Result code
     */
    Result setLogger(std::shared_ptr<Logger> logger) noexcept;


    /**
     * @brief                   Retrieve the logger used by the Builder
     * 
     * @return                  Logger. Could be a nullptr
     */
    std::shared_ptr<Logger> logger() const noexcept;


private:
    Result _buildEngine(const std::string& inputFilePath,
                        std::shared_ptr<nvinfer1::IHostMemory>* output,
                        Precision precision) const noexcept;

private:
    bool                        _initialized;

    std::shared_ptr<Logger>     _logger;

    std::unique_ptr<TensorRT_Logger>                _trtLogger;
};

}   /*  namespace yolov5    */

#endif  /*  include guard   */