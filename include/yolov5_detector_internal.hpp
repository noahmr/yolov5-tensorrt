/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       YoloV5 inference through TensorRT (detector internals)
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
#include "yolov5_logging.hpp"

#include <vector>

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <opencv2/opencv.hpp>

namespace yolov5
{

namespace internal
{

int32_t dimsVolume(const nvinfer1::Dims& dims) noexcept;


bool dimsToString(const nvinfer1::Dims& dims, std::string* out) noexcept;


/**
 * Used to store all (relevant) properties of an engine binding. This does
 * not include memory or any i/o.
 */
class EngineBinding
{
public:
    EngineBinding() noexcept;

    ~EngineBinding() noexcept;

public:

    void swap(EngineBinding& other) noexcept;


    const int& index() const noexcept;

    const std::string& name() const noexcept;

    const nvinfer1::Dims& dims() const noexcept;
    const int& volume() const noexcept;

    bool isDynamic() const noexcept;

    const bool& isInput() const noexcept;


    void toString(std::string* out) const noexcept;


    static bool setup(const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                    const std::string& name, EngineBinding* binding) noexcept;

    static bool setup(const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                    const int& index, EngineBinding* binding) noexcept;
private:
    int                 _index;

    std::string         _name;

    nvinfer1::Dims      _dims;
    int                 _volume;    /*  note: calculated based on dims  */

    bool                _isInput;
};


/**
 * Used to manage memory on the CUDA device, corresponding to the engine
 * bindings.
 */
class DeviceMemory
{
public:
    DeviceMemory() noexcept;

    ~DeviceMemory() noexcept;

private:
    DeviceMemory(const DeviceMemory&);

public:

    void swap(DeviceMemory& other) noexcept;

    /**
     * @brief           Get the beginning of the data. This can be passed onto
     *                  the TensorRT engine
     */
    void** begin() const noexcept;

    /**
     * @brief           Obtain a pointer to the device memory corresponding to
     *                  the specified binding
     * 
     * @param index     Index of the engine binding
     */
    void* at(const int& index) const noexcept;

    /**
     * @brief           Try setting up the Device Memory based on the TensorRT
     *                  engine
     * 
     * 
     * @param logger    Logger to be used
     * 
     * @param engine    TensorRT engine
     * @param output    Output
     * 
     * @return Result   Result code
     */
    static Result setup(const std::shared_ptr<Logger>& logger, 
                    std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                    DeviceMemory* output) noexcept;

private:
    std::vector<void*>  _memory;
};


/**
 * @brief               Check whether OpenCV-CUDA is supported
 */
bool opencvHasCuda() noexcept;


/**
 * Used to store the Letterbox parameters used for a particular image. These
 * can be used to transform the bounding boxes returned by the engine to use
 * coordinates in the original input image.
 */
class PreprocessorTransform
{
public:
    PreprocessorTransform() noexcept;

    PreprocessorTransform(const cv::Size& inputSize, 
                            const double& f, const int& leftWidth, 
                            const int& topHeight) noexcept;

    ~PreprocessorTransform() noexcept;

private:

public:

    /**
     * @brief               Transform bounding box from network space to input
     *                      space
     */
    cv::Rect transformBbox(const cv::Rect& input) const noexcept;

private:
    cv::Size    _inputSize;

    double      _f;
    int         _leftWidth;
    int         _topHeight;
};


/**
 * Used to perform pre-processing task, and to store intermediate buffers to
 * speed up repeated computations.
 * 
 * Note that this base class does not actually do any processing.
 */
class Preprocessor
{
public:
    Preprocessor() noexcept;

    virtual ~Preprocessor() noexcept;

    enum InputType
    {
        INPUTTYPE_BGR       = 0,
        INPUTTYPE_RGB
    };

public:

    void setLogger(std::shared_ptr<Logger> logger) noexcept;

    /**
     * @brief               Set up the Preprocessor
     * 
     * The implementation of the base class only manages the transforms.
     * 
     * @param inputDims     Engine input dimensions
     * @param flags         Additional flags
     * @param batchSize     Number of images that will be processed (i.e. in
     *                      batch mode)
     * @param inputMemory   Start of input on the CUDA device
     * 
     * @return              True on success, False otherwise
     */
    virtual bool setup(const nvinfer1::Dims& inputDims, 
                        const int& flags, const int& batchSize,
                        float* inputMemory) noexcept = 0;

    virtual void reset() noexcept = 0;

    /**
     * @brief               Process the input
     * 
     * @param index         Index in the input batch
     * @param input         Input image
     * @param last          Boolean indicating whether this is the last image
     *                      in the batch
     * 
     * @return              True on success, False otherwise
     */
    virtual bool process(const int& index, const cv::Mat& input, 
                const bool& last) noexcept;

    virtual bool process(const int& index, 
                    const cv::cuda::GpuMat& input, const bool& last) noexcept;

    virtual cudaStream_t cudaStream() const noexcept = 0;

    virtual bool synchronizeCudaStream() noexcept = 0;


    /**
     * @brief               Transform bounding box from network space to input
     *                      space, for a particular image in the batch
     * 
     * @param index         Index in the input batch
     * @param input         Input bounding box
     */
    cv::Rect transformBbox(const int& index, 
                    const cv::Rect& bbox) const noexcept;

protected:
    std::shared_ptr<Logger>                 _logger;

    std::vector<PreprocessorTransform>      _transforms;
};


/**
 * Preprocessing based on letterboxing with OpenCV CPU operations
 */
class CvCpuPreprocessor : public Preprocessor
{
public:
    CvCpuPreprocessor() noexcept;

    virtual ~CvCpuPreprocessor() noexcept;

public:

    virtual bool setup(const nvinfer1::Dims& inputDims,
                        const int& flags, const int& batchSize,
                        float* inputMemory) 
                        noexcept override;

    virtual void reset() noexcept override;

    virtual bool process(const int& index, 
                    const cv::Mat& input, const bool& last) noexcept override;

    virtual bool process(const int& index, 
                    const cv::cuda::GpuMat& input, const bool& last)
                    noexcept override;

    virtual cudaStream_t cudaStream() const noexcept override;

    virtual bool synchronizeCudaStream() noexcept override;

private:
    cudaStream_t    _cudaStream;

    InputType       _lastType;
    int             _lastBatchSize;

    int             _networkCols;
    int             _networkRows;

    cv::Mat         _buffer1;
    cv::Mat         _buffer2;
    cv::Mat         _buffer3;

    std::vector<std::vector<cv::Mat>>   _inputChannels;

    std::vector<float>                  _hostInputMemory;
    float*          _deviceInputMemory;
};


/**
 * Preprocessing based on letterboxing with OpenCV-CUDA operations. Note
 * that OpenCV-CUDA must be available for this. If not, this class will
 * not perform any actual operations.
 */
class CvCudaPreprocessor : public Preprocessor
{
public:

    CvCudaPreprocessor() noexcept;

    virtual ~CvCudaPreprocessor() noexcept;

public:

    virtual bool setup(const nvinfer1::Dims& inputDims,
                        const int& flags, const int& batchSize,
                        float* inputMemory) 
                        noexcept override;

    virtual void reset() noexcept override;

    virtual bool process(const int& index, 
                    const cv::Mat& input, const bool& last) noexcept override;

    virtual bool process(const int& index, 
                    const cv::cuda::GpuMat& input,
                    const bool& last) noexcept override;

    virtual cudaStream_t cudaStream() const noexcept override;

    virtual bool synchronizeCudaStream() noexcept override;

private:
    cv::cuda::Stream    _cudaStream;

    InputType           _lastType;
    int                 _lastBatchSize;

    int                 _networkCols;
    int                 _networkRows;

    cv::cuda::GpuMat    _buffer0;
    cv::cuda::GpuMat    _buffer1;
    cv::cuda::GpuMat    _buffer2;
    cv::cuda::GpuMat    _buffer3;

    std::vector<std::vector<cv::cuda::GpuMat>>      _inputChannels;
};


}   /*  namespace internal  */

}   /*  namespace yolov5    */