/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5 inference through NVIDIA TensorRT (common utilities)
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
#ifndef _YOLOV5_COMMON_HPP_
#define _YOLOV5_COMMON_HPP_

/*  C/C++   */
#include <string>

/*  macro's for internal use    */
#define YOLOV5_UNUSED(x) (void)x;

namespace yolov5
{

enum Result
{
    /*      Invalid input specified. This typically indicates a programming
            error in your software (i.e. a bug in your software).
     */
    RESULT_FAILURE_INVALID_INPUT                = -100,


    /*      Not initialized yet */
    RESULT_FAILURE_NOT_INITIALIZED              = -90,


    /*      Not loaded yet (e.g. no engine loaded yet)    */
    RESULT_FAILURE_NOT_LOADED                   = -80,

    /*      Issue with the loaded model (e.g. input binding is missing) */
    RESULT_FAILURE_MODEL_ERROR                  = -70,              

    /*      Indicates that you are trying to use a function that OpenCV-CUDA,
            but your OpenCV has no support for this. This typically indicates a
            programming error in your software  */
    RESULT_FAILURE_OPENCV_NO_CUDA               = -21,


    /*      Error related to filesystem (e.g. could not open file)    */
    RESULT_FAILURE_FILESYSTEM_ERROR             = -50,


    /*      Internal cuda error (e.g. could not allocate memory)    */
    RESULT_FAILURE_CUDA_ERROR                   = -40,

    /*      Internal TensorRT error (e.g. could not setup execution context  */
    RESULT_FAILURE_TENSORRT_ERROR               = -30,

    /*      Internal OpenCV error   */
    RESULT_FAILURE_OPENCV_ERROR                 = -20,


    /*      Memory-related error    */
    RESULT_FAILURE_ALLOC                        = -11,

    /*      Other error */
    RESULT_FAILURE_OTHER                        = -10,


    /*      Successfull execution   */
    RESULT_SUCCESS                              = 0
};


/**
 * @brief           Get a textual description of a result code
 * 
 * If the specified value 'r' is not a valid yolov5 result code, an empty
 * string is returned. Note that the methods and functions of this library
 * always return valid result codes.
 * 
 * Outputs:
 * - RESULT_FAILURE_INVALID_INPUT:              "invalid input"
 * - RESULT_FAILURE_NOT_INITIALIZED:            "not initialized"
 * - RESULT_FAILURE_NOT_LOADED:                 "not loaded"
 * - RESULT_FAILURE_MODEL_ERROR:                "model error"
 * - RESULT_FAILURE_OPENCV_NO_CUDA:             "opencv lacks cuda"
 * - RESULT_FAILURE_FILESYSTEM_ERROR:           "filesystem error"
 * - RESULT_FAILURE_CUDA_ERROR:                 "cuda error"
 * - RESULT_FAILURE_TENSORRT_ERROR:             "tensorrt error"
 * - RESULT_FAILURE_OPENCV_ERROR:               "opencv error"
 * - RESULT_FAILURE_ALLOC:                      "memory error"
 * - RESULT_FAILURE_OTHER:                      "other error"
 * - RESULT_SUCCESS:                            "success"
 * - In case of an invalid result code: "" (empty string)
 * 
 * 
 * @param r         Result code
 * 
 * @return          String
 */
const char* result_to_string(Result r) noexcept;

/**
 * @brief           Get a textual description of a result code
 * 
 * See result_to_string(Result) for more information.
 * 
 * If 'r' is not valid a valid result code, False is returned and 'out' is
 * left untouched. Note that the methods and functions f this library always
 * return valid result codes.
 * 
 * @param r         Result code
 * @param out       Output. Can be nullptr
 *
 * @return          True on success, False otherwise
 */
bool result_to_string(Result r, std::string* out) noexcept;



enum Precision
{
    PRECISION_FP32          = 0,    /**<    32-bit floating point mode  */

    PRECISION_FP16          = 1,    /**<    16-bit floating point mode  */
};

/**
 * @brief           Get a textual description of a precision code
 * 
 * If the specified value 'p' is not a valid yolov5 precision, an empty
 * string is returned.
 * 
 * Outputs:
 * - PRECISION_FP32:                "fp32"
 * - PRECISION_FP16:                "fp16"
 * - In case of an invalid input:   "" (empty string)
 * 
 * @param p         Precision
 * @return          String
 */
const char* precision_to_string(Precision p) noexcept;

/**
 * @brief           Get a textual description of a precision code
 * 
 * See precision_to_string(Precision) for more information.
 * 
 * If 'r' is not valid a valid precision code, False is returned and 'out' is
 * left untouched. 
 * 
 * @param p         Precision
 * @param out       Output. Can be nullptr
 *
 * @return          True on success, False otherwise
 */
bool precision_to_string(Precision p, std::string* out) noexcept;


/**
 * Additional flags that can be passed to the Detector
 */
enum DetectorFlag
{
    INPUT_BGR =         1,
    /**<    input image is in BGR colorspace(opencv default) */

    INPUT_RGB =         2,
    /**<    input image is in RGB colorspace */

    PREPROCESSOR_CVCUDA     =   4,
    /**<    OpenCV-CUDA pre-processing should be used */
    
    PREPROCESSOR_CVCPU      =   8
    /**<    OpenCV-CPU pre-processing should be used */
};



}   /*  namespace yolov5    */

#endif  /*  include guard   */