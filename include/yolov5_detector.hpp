/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5 inference through TensorRT (detector)
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
#ifndef _YOLOV5_DETECTOR_HPP_
#define _YOLOV5_DETECTOR_HPP_

#include <yolov5_detector_internal.hpp>

namespace yolov5
{


/**
 * The main class for YoloV5 detection using TensorRT.
 * 
 * Before loading a TensorRT engine or performing inference, you should first
 * initialize the detector by using the init() method.
 * 
 * 
 * Basic usage example
 * 
 *  yolov5::Detector detector;
 *  detector.init();
 *  detector.loadEngine("yolov5.engine");
 *
 *  cv::Mat image = cv::imread("image.png");
 *
 *  std::vector<yolov5::Detection> detections;
 *  detector.detect(image, &detections);
 *
 */
class Detector
{
public:
	/// ***
	///	Constructor / Destructor
    /// ***

    /**
     * @brief                   Construct a new Detector object, using default
     *                          options for everything
     */
    Detector() noexcept;


    /**
     * @brief                   Destroy Detector object. Frees up any resources
     */
    ~Detector() noexcept;

private:

    Detector(const Detector& src) noexcept;

public:
	/// ***
	///	Initialization
    /// ***

    /**
     * @brief                   Initialize the Detector.
     * 
     * The initialization consists of multiple steps. If a particular step 
     * fails, the method returns False and appropriate error messages are
     * logged, and later steps are not performed. In this case, the method
     * might be called again at a later time to complete the initialization.
     * 
     * If no logger has been set before, this method will create the
     * default logger provided by this library, which simply prints messages
     * to stdout.
     * 
     * 
     * This method will also set up the pre-processor that will be used for
     * object detection. By default, the OpenCV-CUDA pre-processor is picked if
     * it is available. If not (meaning either OpenCV was built without CUDA
     * support, or no CUDA devices are currently available), a CPU based
     * pre-processor is used. To change this behaviour, use the appropriate
     * flags.
     * 
     * 
     * Supported flags:
     * - PREPROCESSOR_CVCUDA : specify that the OpenCV-CUDA pre-processor
     *  should be used. If it is not available, this method fails and
     * the RESULT_FAILURE_OPENCV_NO_CUDA code is returned.
     * 
     * - PREPROCESSOR_CVCPU : specify that the OpenCV-CPU pre-processor
     *  should be used. This pre-processor is always available.
     * 
     * Any unsupported flags are ignored.
     * 
     * 
     * On success, RESULT_SUCCESS is returned and no messages are logged.
     * 
     * 
     * @param flags             (Optional) Additional flags for initialization
     * 
     * @return                  Result code
     */
    Result init(int flags = 0) noexcept;

    
    /**
     * @brief                   Query whether the Detector is initialized
     * 
     * @return                  True if initialized, False otherwise  
     */
    bool isInitialized() const noexcept;



	/// ***
	///	Engine
    /// ***

    /**
     * @brief                   Load a TensorRT engine from a file
     * 
     * The initialization should have been completed (i.e. through
     * the init() method).
     * 
     * If any code other than RESULT_SUCCESS is returned, this method has
     * no effect, and loading an engine may be attempted again at a later time,
     * for instance after freeing up memory on either the CUDA device or host.
     * 
     * If an engine is already loaded, this method will first fully load the
     * new engine, and only if this is successfull, the old engine is replaced.
     * 
     * 
     * @param filepath          Path to engine file in filesystem
     * 
     * @return                  Result code
     */
    Result loadEngine(const std::string& filepath) noexcept;


    /**
     * @brief                   Load a TensorRT engine from the provided data
     * 
     * The initialization should have been completed (i.e. through
     * the init() method).
     * 
     * If any code other than RESULT_SUCCESS is returned, this method has
     * no effect, and loading an engine may be attempted again at a later time,
     * for instance after freeing up memory on either the CUDA device or host.
     * 
     * If an engine is already loaded, this method will first fully load the
     * new engine, and only if this is successfull, the old engine is replaced.
     * 
     * 
     * @param data              Engine data
     *  
     * @return                  Result code
     */
    Result loadEngine(const std::vector<char>& data) noexcept;


    /**
     * @brief                   Query whether an inference engine has been
     *                          loaded already
     * 
     * @return                  True if loaded, False otherwise 
     */
    bool isEngineLoaded() const noexcept;



	/// ***
	///	Classes
    /// ***

    /**
     * @brief                   Retrieve the number of classes of 
     *                          the engine/network
     * 
     * An engine should have been loaded already. If not, an error message is
     * logged and 0 is returned.
     * 
     * @return                  Number of classes
     */
    int numClasses() const noexcept;


    /**
     * @brief                   Set the classes of the network
     * 
     * See the 'Classes' class for more information.
     * 
     * Note that it is NOT mandatory to set the Classes object. This is only
     * useful if you want ClassIds to be automatically mapped to
     * Class names (e.g. "human", "bike") in the detections that are given by
     * the Detector.
     * 
     * 
     * This method may be used at any point in time, e.g. before/after
     * initialization, before/after loading an engine etc.
     * 
     * 
     * @param classes           Classes (e.g. names)
     * 
     * @return                  Result code 
     */
    Result setClasses(const Classes& classes) noexcept;



    /// ***
	///	Detection
    /// ***

    /**
     * @brief                   Detect objects in the specified image using
     *                          the YoloV5 model
     * 
     * An engine should have been loaded already.
     * 
     * This method accepts input of any size, but providing an input
     * of the exact size for which the network was configured will result in
     * a lower detection time, since no pre-processing is required. The network
     * input size can be retrieved using the inferenceSize() method.
     * 
     * 
     * By default, this method assumes that your input is in BGR format. If
     * this is not the case, this can be specified by setting the appropriate
     * flags.
     * 
     * 
     * Supported flags:
     * - INPUT_BGR : specify that the input is in BGR format (opencv default)
     * - INPUT_RGB :specify that the input is in RGB format

     * Any unsupported flags are ignored.
     * 
     * 
     * If any code other than RESULT_SUCCESS is returned, the output 'out' is
     * left untouched.
     * 
     * 
     * @param img               Input images
     * @param out               Output; Can be nullptr
     * @param flags             (Optional) Additional flags for detection
     * 
     * @return                  True on success, False otherwise
     */
    Result detect(const cv::Mat& img, 
                    std::vector<Detection>* out,
                    int flags = 0) noexcept;

    /**
     * @brief                   Detect objects in the specified image (in CUDA
     *                          memory) using the YoloV5 model
     * 
     * See the documentation on detect(const cv::Mat&, std::vector<Detection>*, 
     *      int) for more information.
     */
    Result detect(const cv::cuda::GpuMat& img, 
                    std::vector<Detection>* out,
                    int flags = 0) noexcept;

    /**
     * @brief                   Detect objects in the specified images using
     *                          batch inference with the YoloV5 model
     * 
     * An engine should have been loaded already.
     * 
     * This method accepts inputs of any size, but providing inputs
     * of the exact size for which the network was configured will result in
     * a lower detection time, since no pre-processing is required. The network
     * input size can be retrieved using the inferenceSize() method.
     * 
     * Note that the inputs can potentially all have different sizes if this is
     * desired.
     * 
     * 
     * By default, this method assumes that your inputs are in BGR format. If
     * this is not the case, this can be specified by setting the appropriate
     * flags.
     * 
     * 
     * Supported flags:
     * - INPUT_BGR : specify that the inputs are all in BGR
     *  format (opencv default)
     * - INPUT_RGB : specify that the inputs are all in RGB format

     * Any unsupported flags are ignored.
     * 
     * 
     * If any code other than RESULT_SUCCESS is returned, the output 'out' is
     * left untouched.
     * 
     * 
     * @param images            Input images
     * @param out               Outputs for each image.
     * @param flags             (Optional) Additional flags for detection
     * 
     * @return                  True on success, False otherwise
     */
    Result detectBatch(const std::vector<cv::Mat>& images, 
                    std::vector<std::vector<Detection>>* out,
                    int flags = 0) noexcept;
    
    /**
     * @brief                   Detect objects in the specified images (in CUDA
     *                          memory) using batch inference with YoloV5

     * See the documentation on detectBatch(const std::vector<cv::Mat>&, 
     *      std::vector<Detection>*, int) for more information.
     */
    Result detectBatch(const std::vector<cv::cuda::GpuMat>& images, 
                    std::vector<std::vector<Detection>>* out,
                    int flags = 0) noexcept;


    /// ***
	///	Detection Parameters
    /// ***

    /**
     * @brief                   Obtain the score threshold
     */
    double scoreThreshold() const noexcept;


    /**
     * @brief                   Set the Score threshold: used to filter 
     *                          objects by score
     * 
     * @param v                 Score threshold. Should be in [0, 1]
     * @return                  Result code 
     */
    Result setScoreThreshold(const double& v) noexcept;


    /**
     * @brief                   Obtain the NMS threshold
     */
    double nmsThreshold() const noexcept;


    /**
     * @brief                   Set the NMS threshold
     * 
     * @param v                 NMS threshold. Should be in [0, 1]
     * @return                  Result code
     */
    Result setNmsThreshold(const double& v) noexcept;



    /// ***
	///	Engine/Network properties
    /// ***

    /**
     * @brief                   Retrieve the batch size of the engine/network
     * 
     * An engine should have been loaded already. If not, an error message is
     * logged and 0 is returned.
     * 
     * @return                  Batch size
     */
    int batchSize() const noexcept;
    
    /**
     * @brief                   Input size for which the network was configured
     * 
     * An engine should have been loaded already. If not, an error message is
     * logged and Size(0, 0) is returned.
     * 
     * @return                  Size
     */
    cv::Size inferenceSize() const noexcept;


	/// ***
	///	Logging
    /// ***
    
    /**
     * @brief                   Set a custom logger to be used by the Detector
     * 
     * This method can be called at any time, either before or after
     * initialization, and will take effect immediately.
     * 
     * 
     * @param logger            New logger; Should NOT be a nullptr
     * 
     * @return                  Result code
     */
    Result setLogger(std::shared_ptr<Logger> logger) noexcept;


    /**
     * @brief                   Retrieve the logger used by the Detector
     * 
     * @return                  Logger. Can potentially be a nullptr if
     *                          the Detector has not been initialized yet
     */
    std::shared_ptr<Logger> logger() const noexcept;


private:
    /**
     * @brief                   Not implemented
     */
    Detector& operator=(const Detector& rhs);

    /**
     * @brief                   Load the engine from data
     * 
     * @return                  Result code
     */
    Result _loadEngine(const std::vector<char>& data) noexcept;

    void _printBindings(const std::unique_ptr<nvinfer1::ICudaEngine>& engine)
        const noexcept;

    int _batchSize() const noexcept;

    int _numClasses() const noexcept;


    Result _detect(std::vector<Detection>* out);

    Result _detectBatch(const int& nrImages,
                    std::vector<std::vector<Detection>>* out);

    /**
     * @brief                   Run the TensorRT engine on the network inputs,
     *                          copy the output to host memory
     */
    Result _inference(const char* logid);

    /**
     * @brief                   Decode network output, convert to
     *                          proper Detection objects
     */
    Result _decodeOutput(const char* logid, const int& index, 
                    std::vector<Detection>* out);

private:
    bool                            _initialized;

    std::shared_ptr<Logger>         _logger;

    Classes                         _classes;
    double                          _scoreThreshold;
    double                          _nmsThreshold;


    /*  TensorRT    */
    std::unique_ptr<TensorRT_Logger>                _trtLogger;
    std::unique_ptr<nvinfer1::IRuntime>             _trtRuntime;

    /*  note: execution context depends on the engine, and should be destroyed
            _before_ the engine is destroyed */
    std::unique_ptr<nvinfer1::ICudaEngine>          _trtEngine;
    std::unique_ptr<nvinfer1::IExecutionContext>    _trtExecutionContext;

    
    /*  I/O  */
    internal::EngineBinding         _inputBinding;
    internal::EngineBinding         _outputBinding;

    std::unique_ptr<internal::Preprocessor>     _preprocessor;

    internal::DeviceMemory          _deviceMemory;

    std::vector<float>              _outputHostMemory;
};

}   /*  namespace yolov5    */

#endif  /*  include guard   */