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

#include "yolov5_detector_internal.hpp"

#if __has_include(<opencv2/cudaarithm.hpp>) && __has_include(<opencv2/cudawarping.hpp>)

#define YOLOV5_OPENCV_HAS_CUDA 1

#include <opencv2/cudaarithm.hpp>   /*  cv::cuda::split(...)    */
#include <opencv2/cudawarping.hpp>  /*  cv::cuda::resize(...)   */
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

#include <cuda_runtime_api.h>

namespace yolov5
{

namespace internal
{

int32_t dimsVolume(const nvinfer1::Dims& dims) noexcept
{
    int32_t r = 0;
    if(dims.nbDims > 0)
    {
        r = 1;
    }

    for(int32_t i = 0; i < dims.nbDims; ++i)
    {
        r = r * dims.d[i];
    }
    return r;
}


bool dimsToString(const nvinfer1::Dims& dims, std::string* out) noexcept
{
    try
    {
        *out = "(";
        for(int32_t i = 0; i < dims.nbDims; ++i)
        {
            *out += std::to_string(dims.d[i]);
            if(i < dims.nbDims - 1)
            {
                out->push_back(',');
            }
        }
        out->push_back(')');
    }
    catch(const std::exception& e)
    {
        return false;
    }
    return true;
}

EngineBinding::EngineBinding() noexcept
{
}

EngineBinding::~EngineBinding() noexcept
{
}

void EngineBinding::swap(EngineBinding& other) noexcept
{
    std::swap(_index, other._index);
    std::swap(_name, other._name);
    std::swap(_volume, other._volume);

    nvinfer1::Dims tmp;
    std::memcpy(&tmp, &_dims, sizeof(nvinfer1::Dims));
    std::memcpy(&_dims, &other._dims, sizeof(nvinfer1::Dims));
    std::memcpy(&other._dims, &tmp, sizeof(nvinfer1::Dims));

    std::swap(_isInput, other._isInput);
}

const int& EngineBinding::index() const noexcept
{
    return _index;
}

const std::string& EngineBinding::name() const noexcept
{
    return _name;
}

const nvinfer1::Dims& EngineBinding::dims() const noexcept
{
    return _dims;
}

const int& EngineBinding::volume() const noexcept
{
    return _volume;
}

bool EngineBinding::isDynamic() const noexcept
{
    for(int i = 0; i < _dims.nbDims; ++i)
    {
        if(_dims.d[i] == -1)
        {
            return true;
        }
    }
    return false;
}

const bool& EngineBinding::isInput() const noexcept
{
    return _isInput;
}

void EngineBinding::toString(std::string* out) const noexcept
{
    std::string dimsStr;
    if(!dimsToString(_dims, &dimsStr))
    {
        return;
    }

    try
    {
        *out = "name: '" + _name + "'"
                    + " ;  dims: " + dimsStr 
                    + " ;  isInput: " + (_isInput ? "true" : "false")
                    + " ;  dynamic: " + (isDynamic() ? "true" : "false");
    }
    catch(const std::exception& e)
    {
    }
}

bool EngineBinding::setup(const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                    const std::string& name, EngineBinding* binding) noexcept
{
    try
    {
        binding->_name = name;
    }
    catch(const std::exception& e)
    {
        return false;
    }

    binding->_index = engine->getBindingIndex(name.c_str());
    if(binding->_index == -1)
    {
        return false;
    }

    binding->_dims = engine->getBindingDimensions(binding->_index);
    binding->_volume = dimsVolume(binding->_dims);

    binding->_isInput = engine->bindingIsInput(binding->_index);

    return true;
}

bool EngineBinding::setup(const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                    const int& index, EngineBinding* binding) noexcept
{
    binding->_index = index;
    const char* name = engine->getBindingName(index);
    if(name == nullptr)
    {
        return false;
    }

    try
    {
        binding->_name = std::string(name);
    }
    catch(const std::exception& e)
    {
        return false;
    }

    binding->_dims = engine->getBindingDimensions(binding->_index);
    binding->_volume = dimsVolume(binding->_dims);

    binding->_isInput = engine->bindingIsInput(binding->_index);

    return true;
}

DeviceMemory::DeviceMemory() noexcept
{
}

DeviceMemory::~DeviceMemory() noexcept
{
    for(unsigned int i = 0; i < _memory.size(); ++i)
    {
        if(_memory[i])
        {
            cudaFree(_memory[i]);
        }
    }
    _memory.clear();
}

void DeviceMemory::swap(DeviceMemory& other) noexcept
{
    std::swap(_memory, other._memory);
}

void** DeviceMemory::begin() const noexcept
{
    return (void**)_memory.data();
}

void* DeviceMemory::at(const int& index) const noexcept
{
    return _memory[index];
}

Result DeviceMemory::setup(const std::shared_ptr<Logger>& logger, 
                    std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                    DeviceMemory* output) noexcept
{
    const int32_t nbBindings = engine->getNbBindings();
    for(int i = 0; i < nbBindings; ++i)
    {
        const nvinfer1::Dims dims = engine->getBindingDimensions(i); 
        const int volume = dimsVolume(dims);

        try
        {
            output->_memory.push_back(nullptr);
        }
        catch(const std::exception& e)
        {
            logger->logf(LOGGING_ERROR, "[DeviceMemory] setup() failure: "
                        "exception: %s", e.what()); 
            return RESULT_FAILURE_ALLOC;
        }
        void** ptr = &output->_memory.back();

        auto r = cudaMalloc(ptr, volume * sizeof(float));
        if(r != 0 || *ptr == nullptr)
        {
            logger->logf(LOGGING_ERROR, "[DeviceMemory] setup() failure: "
                        "could not allocate device memory: %s", 
                        cudaGetErrorString(r));
            return RESULT_FAILURE_CUDA_ERROR;
        }

    }
    return RESULT_SUCCESS;
}

bool opencvHasCuda() noexcept
{
    int r = 0;
    try
    {
        r = cv::cuda::getCudaEnabledDeviceCount();
    }
    catch(const std::exception& e)
    {
        return 0;
    }
	return (r > 0);
}

PreprocessorTransform::PreprocessorTransform() noexcept
    : _inputSize(0, 0), _f(1), _leftWidth(0), _topHeight(0)
{
}

PreprocessorTransform::PreprocessorTransform(const cv::Size& inputSize, 
                        const double& f, const int& leftWidth, 
                        const int& topHeight) noexcept
    : _inputSize(inputSize), _f(f), 
        _leftWidth(leftWidth), _topHeight(topHeight)
{
}

PreprocessorTransform::~PreprocessorTransform() noexcept
{
}

cv::Rect PreprocessorTransform::transformBbox(
                                const cv::Rect& input) const noexcept
{
    cv::Rect r;
    r.x = (input.x - _leftWidth) / _f;
    r.x = MAX(0, MIN(r.x, _inputSize.width-1));

    r.y = (input.y - _topHeight) / _f;
    r.y = MAX(0, MIN(r.y, _inputSize.height-1));

    r.width = input.width / _f;
    if(r.x + r.width > _inputSize.width)
    {
        r.width = _inputSize.width - r.x;
    }
    r.height = input.height / _f;
    if(r.y + r.height > _inputSize.height)
    {
        r.height = _inputSize.height - r.y;
    }
    return r;
}


Preprocessor::Preprocessor() noexcept
{
}

Preprocessor::~Preprocessor() noexcept
{
}

void Preprocessor::setLogger(std::shared_ptr<Logger> logger) noexcept
{
    _logger = logger;
}

bool Preprocessor::process(const int& index, const cv::Mat& input,
                        const bool& last) noexcept
{
    YOLOV5_UNUSED(input);
    YOLOV5_UNUSED(last);

    if(_transforms.size() < (unsigned int)index+1)
    {
        try
        {
            _transforms.resize(index + 1);
        }
        catch(const std::exception& e)
        {
            _logger->logf(LOGGING_ERROR, "[Preprocessor] process() "
                            "failure: got exception setting up transforms: "
                            "%s", e.what());
            return false;
        }
    }
    return true;
}

bool Preprocessor::process(const int& index, 
            const cv::cuda::GpuMat& input, const bool& last) noexcept
{
    YOLOV5_UNUSED(input);
    YOLOV5_UNUSED(last);

    if(_transforms.size() < (unsigned int)index+1)
    {
        try
        {
            _transforms.resize(index + 1);
        }
        catch(const std::exception& e)
        {
            _logger->logf(LOGGING_ERROR, "[Preprocessor] process() "
                            "failure: got exception setting up transforms: "
                            "%s", e.what());
            return false;
        }
    }
    return true;
}

cv::Rect Preprocessor::transformBbox(const int& index, 
                    const cv::Rect& bbox) const noexcept
{
    return _transforms[index].transformBbox(bbox);
}

template <typename T>
static void setupChannels(const cv::Size& size, 
                        const Preprocessor::InputType& inputType, 
                        float* inputPtr,
                        std::vector<T>& channels)
{
    const int channelSize = size.area();
    if(inputType == Preprocessor::INPUTTYPE_BGR)  /*  INPUT_BGR   */
    {
        /*  B channel will go here  */
        channels.push_back(T(size, CV_32FC1, inputPtr + 2 * channelSize));
        /*  G channel will go here  */
        channels.push_back(T(size, CV_32FC1, inputPtr + 1 * channelSize));
        /*  R channel will go here  */
        channels.push_back(T(size, CV_32FC1, inputPtr));
    }
    else    /*  INPUTTYPE_RGB   */
    {
        /*  R channel will go here  */
        channels.push_back(T(size, CV_32FC1, inputPtr));
        /*  G channel will go here  */
        channels.push_back(T(size, CV_32FC1, inputPtr + 1 * channelSize));
        /*  B channel will go here  */
        channels.push_back(T(size, CV_32FC1, inputPtr + 2 * channelSize));
    }
}

CvCpuPreprocessor::CvCpuPreprocessor() noexcept 
    : _cudaStream(nullptr), _lastType((InputType)-1), _lastBatchSize(-1), 
    _networkCols(0), _networkRows(0)
{
}

CvCpuPreprocessor::~CvCpuPreprocessor() noexcept
{
}

bool CvCpuPreprocessor::setup(const nvinfer1::Dims& inputDims,
                            const int& flags, const int& batchSize,
                            float* inputMemory) noexcept
{
    if(!_cudaStream)
    {
        auto r = cudaStreamCreate(&_cudaStream);
        if(r != 0)
        {
            _logger->logf(LOGGING_ERROR, "[CvCpuPreprocessor] setup() "
                        "failure: could not create cuda stream: %s", 
                        cudaGetErrorString(r));
            return false;
        }
    }

    if((flags & INPUT_RGB) && (flags & INPUT_BGR))
    {
        _logger->log(LOGGING_ERROR, "[CvCpuPreprocessor] setup() "
                    "failure: both INPUT_RGB and INPUT_BGR flags specified"); 
        return false;
    }
    InputType inputType = INPUTTYPE_BGR;
    if(flags & INPUT_RGB)
    {
        inputType = INPUTTYPE_RGB;
    }
    
    if(_lastType == inputType && _lastBatchSize == batchSize)
    {
        return true;
    }
    _lastType = inputType;
    _lastBatchSize = batchSize;

    _networkRows = inputDims.d[2];
    _networkCols = inputDims.d[3];
    const cv::Size networkSize(_networkCols, _networkRows);

    _deviceInputMemory = inputMemory;

    _inputChannels.clear();
    try
    {
        /*  Set up host input memory    */
        _hostInputMemory.resize(dimsVolume(inputDims));

        _inputChannels.resize(batchSize);
        for(unsigned int i = 0; i < _inputChannels.size(); ++i)
        {
            float* inputPtr = _hostInputMemory.data() 
                        + i * networkSize.area() * 3;

            std::vector<cv::Mat>& channels = _inputChannels[i];
            setupChannels<cv::Mat>(networkSize, inputType, inputPtr, channels);
        }
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[CvCpuPreprocessor] setup() failure: "
                    "got exception while trying to set up input channels: %s",
                    e.what());
        return false;
    }
    return true;
}

void CvCpuPreprocessor::reset() noexcept
{
    /*  this will trigger setup() to take effect next time  */
    _lastType = (InputType)-1;
}

bool CvCpuPreprocessor::process(const int& index, 
                                const cv::Mat& input,
                                const bool& last) noexcept
{
    if(!Preprocessor::process(index, input, last))
    {
        return false;
    }

    PreprocessorTransform& transform = _transforms[index];
    try
    {
        if(input.rows == _networkRows && input.cols == _networkCols)
        {
            input.convertTo(_buffer3, CV_32FC3, 1.0f / 255.0f);
            transform = PreprocessorTransform(input.size(), 1.0, 0, 0);
        }
        else
        {
            const double f =
                    MIN((double)_networkRows / (double)input.rows,
                        (double)_networkCols / (double)input.cols);
            const cv::Size boxSize = cv::Size(input.cols * f,
                                            input.rows * f);

            const int dr = _networkRows - boxSize.height;
            const int dc = _networkCols - boxSize.width;
            const int topHeight = std::floor(dr / 2.0);
            const int bottomHeight = std::ceil(dr / 2.0);
            const int leftWidth = std::floor(dc / 2.0);
            const int rightWidth = std::ceil(dc / 2.0);

            transform = PreprocessorTransform(input.size(), 
                                            f, leftWidth, topHeight);

            cv::resize(input, _buffer1, boxSize, 0, 0, cv::INTER_LINEAR);
            cv::copyMakeBorder(_buffer1, _buffer2, 
                            topHeight, bottomHeight, leftWidth, rightWidth,
                            cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            _buffer2.convertTo(_buffer3, CV_32FC3, 1.0f / 255.0f);
        }
        cv::split(_buffer3, _inputChannels[index]);
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[CvCpuPreprocessor] process() "
                            "failure: got exception setting up input: "
                            "%s", e.what());
        return false;
    }

    /*  Copy from host to device    */
    if(last)
    {
        const int volume = _inputChannels.size()    /*  batch size  */
                * 3 * _buffer3.size().area();   /*  channels * rows*cols   */

        auto r = cudaMemcpyAsync(_deviceInputMemory, 
                (void*)_hostInputMemory.data(), 
                (int)(volume * sizeof(float)),
                cudaMemcpyHostToDevice, _cudaStream);
        if(r != 0)
        {
            _logger->logf(LOGGING_ERROR, "[CvCpuPreprocessor] process() "
                        "failure: could not set up host-to-device transfer "
                        "for input: %s", cudaGetErrorString(r));
            return false;
        }
    }
    return true;
}

bool CvCpuPreprocessor::process(const int& index, 
                                const cv::cuda::GpuMat& input,
                                const bool& last) noexcept
{
    YOLOV5_UNUSED(index);
    YOLOV5_UNUSED(input);
    YOLOV5_UNUSED(last);
    _logger->log(LOGGING_ERROR, "[CvCpuPreprocessor] process() failure: CUDA "
                    "method is not supported. Use CPU variant instead");
    /*  not supported   */
    return false;
}

cudaStream_t CvCpuPreprocessor::cudaStream() const noexcept
{
    return _cudaStream;
}

bool CvCpuPreprocessor::synchronizeCudaStream() noexcept
{
    auto r = cudaStreamSynchronize(_cudaStream); 
    if(r != 0)
    {
        _logger->logf(LOGGING_ERROR, "[CvCpuPreprocessor] "
                "synchronizeCudaStream() failure: %s", cudaGetErrorString(r));
        return false;
    }
    return true;
}

CvCudaPreprocessor::CvCudaPreprocessor() noexcept 
    : _lastType((InputType)-1), _lastBatchSize(-1), 
    _networkCols(0), _networkRows(0)
{
}

CvCudaPreprocessor::~CvCudaPreprocessor() noexcept
{
}

bool CvCudaPreprocessor::setup(const nvinfer1::Dims& inputDims,
                            const int& flags, const int& batchSize,
                            float* inputMemory) noexcept
{
#ifdef YOLOV5_OPENCV_HAS_CUDA
    if((flags & INPUT_RGB) && (flags & INPUT_BGR))
    {
        _logger->log(LOGGING_ERROR, "[CvCudaPreprocessor] setup() "
                    "failure: both INPUT_RGB and INPUT_BGR flags specified"); 
        return false;
    }
    InputType inputType = INPUTTYPE_BGR;
    if(flags & INPUT_RGB)
    {
        inputType = INPUTTYPE_RGB;
    }

    if(_lastType == inputType && _lastBatchSize == batchSize)
    {
        return true;
    }
    _lastType = inputType;
    _lastBatchSize = batchSize;

    _networkRows = inputDims.d[2];
    _networkCols = inputDims.d[3];
    const cv::Size networkSize(_networkCols, _networkRows);

    _inputChannels.clear();
    try
    {
        _inputChannels.resize(batchSize);
        for(unsigned int i = 0; i < _inputChannels.size(); ++i)
        {
            float* inputPtr = inputMemory + i * networkSize.area() * 3;

            std::vector<cv::cuda::GpuMat>& channels = _inputChannels[i];
            setupChannels<cv::cuda::GpuMat>(networkSize, inputType, inputPtr, 
                                            channels);
        }
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[CvCudaPreprocessor] setup() failure: "
                    "got exception while trying to set up input channels: %s",
                    e.what());
        return false;
    }
    return true;
#else
    YOLOV5_UNUSED(inputDims);
    YOLOV5_UNUSED(flags);
    YOLOV5_UNUSED(batchSize);
    YOLOV5_UNUSED(inputMemory);
    _logger->log(LOGGING_ERROR, "[CvCudaPreprocessor] setup() failure: "
                    "OpenCV without CUDA support");
    return false;
#endif
}

void CvCudaPreprocessor::reset() noexcept
{
    /*  this will trigger setup() to take effect next time  */
    _lastType = (InputType)-1;
}

bool CvCudaPreprocessor::process(const int& index, 
                        const cv::Mat& input, const bool& last) noexcept
{
#ifdef YOLOV5_OPENCV_HAS_CUDA
    try
    {
        _buffer0.upload(input);
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[CvCudaPreprocessor] process() "
                            "failure: got exception trying to upload input "
                            "to CUDA device: %s", e.what());
        return false;
    }
    return process(index, _buffer0, last);
#else
    YOLOV5_UNUSED(index);
    YOLOV5_UNUSED(input);
    YOLOV5_UNUSED(last);
    _logger->log(LOGGING_ERROR, "[CvCudaPreprocessor] process() failure: "
                    "OpenCV without CUDA support");
    return false;
#endif
}

bool CvCudaPreprocessor::process(const int& index, 
                        const cv::cuda::GpuMat& input,
                        const bool& last) noexcept
{
#ifdef YOLOV5_OPENCV_HAS_CUDA
    if(!Preprocessor::process(index, input, last))
    {
        return false;
    }

    if(index >= 1)
    {
        /*  If we're processing a batch, first process the previous image
            before changing the internal buffers    */
        if(!synchronizeCudaStream())
        {
            return false;
        }
    }

    PreprocessorTransform& transform = _transforms[index];
    try
    {
        if(input.rows == _networkRows && input.cols == _networkCols)
        {
            input.convertTo(_buffer3, CV_32FC3, 1.0f / 255.0f, _cudaStream);
            transform = PreprocessorTransform(input.size(), 1.0, 0, 0);
        }
        else
        {
            const double f =
                    MIN((double)_networkRows / (double)input.rows,
                        (double)_networkCols / (double)input.cols);
            const cv::Size boxSize = cv::Size(input.cols * f,
                                            input.rows * f);

            const int dr = _networkRows - boxSize.height;
            const int dc = _networkCols - boxSize.width;
            const int topHeight = std::floor(dr / 2.0);
            const int bottomHeight = std::ceil(dr / 2.0);
            const int leftWidth = std::floor(dc / 2.0);
            const int rightWidth = std::ceil(dc / 2.0);

            transform = PreprocessorTransform(input.size(), 
                                        f, leftWidth, topHeight);

            cv::cuda::resize(input, _buffer1, boxSize, 0, 0, cv::INTER_LINEAR,
                            _cudaStream);
            cv::cuda::copyMakeBorder(_buffer1, _buffer2, 
                            topHeight, bottomHeight, leftWidth, rightWidth,
                            cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0), 
                            _cudaStream);
            _buffer2.convertTo(_buffer3, CV_32FC3, 1.0f / 255.0f, _cudaStream);
        }
        cv::cuda::split(_buffer3, _inputChannels[index], _cudaStream);
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[CvCudaPreprocessor] process() "
                            "failure: got exception setting up input: "
                            "%s", e.what());
        return false;
    }
    return true;
#else
    YOLOV5_UNUSED(index);
    YOLOV5_UNUSED(input);
    YOLOV5_UNUSED(last);
    _logger->log(LOGGING_ERROR, "[CvCudaPreprocessor] process() failure: "
                    "OpenCV without CUDA support");
    return false;
#endif
}

cudaStream_t CvCudaPreprocessor::cudaStream() const noexcept
{
#ifdef YOLOV5_OPENCV_HAS_CUDA
    return cv::cuda::StreamAccessor::getStream(_cudaStream);
#else
    return nullptr;
#endif
}

bool CvCudaPreprocessor::synchronizeCudaStream() noexcept
{
    try
    {
        _cudaStream.waitForCompletion();
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[CvCudaPreprocessor] "
                    "synchronizeCudaStream() failure: %s", e.what());
        return false;
    }
    return true;
}


}   /*  namespace internal  */

}   /*  namespace yolov5    */