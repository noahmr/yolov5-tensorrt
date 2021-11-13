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

#include "yolov5_detector.hpp"

#include <fstream>
#include <iterator>
#include <memory>

#include <cstdio>
#include <cstddef>

/*  CUDA    */
#include <cuda_runtime_api.h>

namespace yolov5
{

Detector::Detector() noexcept 
    : _initialized(false), _scoreThreshold(0.4), _nmsThreshold(0.4)
{
}

Detector::~Detector() noexcept
{
}

Result Detector::init(int flags) noexcept
{
    /*  Initialize Logger */
    if(!_logger)
    {
        try
        {
            _logger = std::make_shared<Logger>();
        }
        catch(const std::exception& e)
        {
            /*  logging not available  */
            return RESULT_FAILURE_ALLOC;
        }
    }

    /*  Initialize TensorRT logger  */
    if(!_trtLogger)
    {
        try
        {
            _trtLogger = std::make_unique<TensorRT_Logger>(_logger);
        }
        catch(const std::exception& e)
        {
            _logger->logf(LOGGING_ERROR, "[Detector] init() failure: could not"
                            " create TensorRT logger: %s", e.what());
            return RESULT_FAILURE_ALLOC;
        }
    }

    /*  Set up Preprocessor */
    if(!_preprocessor)
    {
        try
        {
            const bool cvCudaAvailable = internal::opencvHasCuda();

            if((flags & PREPROCESSOR_CVCUDA) && (flags & PREPROCESSOR_CVCPU))
            {
                _logger->log(LOGGING_ERROR, "[Detector] init() failure: "
                            "both PREPROCESSOR_CVCUDA and PREPROCESSOR_CVCPU "
                            "flags specified");
                return RESULT_FAILURE_INVALID_INPUT;
            }
            
            /*  If the CVCUDA flag was specified, OpenCV-CUDA has to be
                available or fail   */
            if(flags & PREPROCESSOR_CVCUDA && !cvCudaAvailable)
            {
                _logger->log(LOGGING_ERROR, "[Detector] init() failure: "
                        "PREPROCESSOR_CVCUDA flag specified, but "
                        "OpenCV-CUDA  pre-processor is not available.");
                return RESULT_FAILURE_OPENCV_NO_CUDA;
            }

            bool useCudaPreprocessor = cvCudaAvailable;
            if(flags & PREPROCESSOR_CVCPU)
            {
                useCudaPreprocessor = false;
            }

            if(useCudaPreprocessor)
            {
                _logger->log(LOGGING_INFO, "[Detector] Using OpenCV-CUDA "
                            "pre-processor");
                _preprocessor = 
                        std::make_unique<internal::CvCudaPreprocessor>();
            }
            else
            {
                _logger->log(LOGGING_INFO, "[Detector] Using OpenCV-CPU "
                            "pre-processor");
                _preprocessor = 
                        std::make_unique<internal::CvCpuPreprocessor>();
            }
        }
        catch(const std::exception& e)
        {
            _logger->logf(LOGGING_ERROR, "[Detector] init() failure: "
                        "could not set up preprocessor: %s", e.what());
            return RESULT_FAILURE_ALLOC;
        }
        _preprocessor->setLogger(_logger);
    }

    /*  Initialize TensorRT runtime */
    if(!_trtRuntime)
    {
        nvinfer1::IRuntime* trtRuntime 
                        = nvinfer1::createInferRuntime(*_trtLogger);
        if(trtRuntime == nullptr)
        {
            _logger->log(LOGGING_ERROR, "[Detector] init() failure: could not"
                    "create TensorRT runtime"); 
            return RESULT_FAILURE_TENSORRT_ERROR;
        }
        std::unique_ptr<nvinfer1::IRuntime> ptr(trtRuntime);
        ptr.swap(_trtRuntime);
    }


    _initialized = true;
    return RESULT_SUCCESS;
}

bool Detector::isInitialized() const noexcept
{
    return _initialized;
}

Result Detector::loadEngine(const std::string& filepath) noexcept
{
    if(!_initialized)
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: "
                        "detector is not initialized yet");
        }
        return RESULT_FAILURE_NOT_INITIALIZED;
    }
    _logger->logf(LOGGING_INFO, "[Detector] Loading TensorRT engine "
            "from '%s'", filepath.c_str());

    std::ifstream file(filepath, std::ios::binary);
    if(!file.good())
    {
        _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: could "
                "not open specified file");
        return RESULT_FAILURE_FILESYSTEM_ERROR;
    }

    std::vector<char> data;
    try
    {
        file.seekg(0, file.end);
        const auto size = file.tellg();
        file.seekg(0, file.beg);

        /*  read entire file into vector    */
        data.resize(size);
        file.read(data.data(), size);
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[Detector] loadEngine() failure: could "
                "not load file into memory: %s", e.what());
        return RESULT_FAILURE_ALLOC;
    }
    file.close();
    return _loadEngine(data);
}

Result Detector::loadEngine(const std::vector<char>& data) noexcept
{
    if(!_initialized)
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: "
                        "detector is not initialized yet");
        }
        return RESULT_FAILURE_NOT_INITIALIZED;
    }
    return _loadEngine(data);
}

bool Detector::isEngineLoaded() const noexcept
{
    return (bool)_trtEngine;
}

int Detector::numClasses() const noexcept
{
    if(!isEngineLoaded())
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] numClasses() failure: no "
                        "engine loaded");
        }
        return 0;
    }
    return _numClasses();
}

Result Detector::setClasses(const Classes& classes) noexcept
{
    if(!classes.isLoaded())
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] setClasses() failure: "
                        "invalid input specified: classes not yet loaded");
        }   
        return RESULT_FAILURE_INVALID_INPUT;
    }

    try
    {
        _classes = classes;
    }
    catch(const std::exception& e)
    {
        if(_logger)
        {
            _logger->logf(LOGGING_ERROR, "[Detector] setClasses() failure: "
                        "could not set up classes: %s", e.what());
        }
        return RESULT_FAILURE_ALLOC;
    }
    return RESULT_SUCCESS;
}

Result Detector::detect(const cv::Mat& img, 
                    std::vector<Detection>* out,
                    int flags) noexcept
{
    if(!isEngineLoaded())
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] detect() failure: no "
                        "engine loaded");
        }
        return RESULT_FAILURE_NOT_LOADED;
    }

    /**     Pre-processing      **/
    if(!_preprocessor->setup( _inputBinding.dims(), flags, _batchSize(),
                            (float*)_deviceMemory.at(_inputBinding.index()) ))
    {
        _logger->log(LOGGING_ERROR, "[Detector] detect() failure: could not "
                        "set up pre-processor");
        return RESULT_FAILURE_OTHER;
    }
    if(!_preprocessor->process(0, img, true))
    {
        _logger->log(LOGGING_ERROR, "[Detector] detect() failure: could not "
                        "pre-process input");
        return RESULT_FAILURE_OTHER;
    }

    return _detect(out);
}

Result Detector::detect(const cv::cuda::GpuMat& img, 
                    std::vector<Detection>* out,
                    int flags) noexcept
{
    if(!isEngineLoaded())
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] detect() failure: no "
                        "engine loaded");
        }
        return RESULT_FAILURE_NOT_LOADED;
    }

    if(!internal::opencvHasCuda())
    {
        _logger->log(LOGGING_ERROR, "[Detector] detect() failure: this method "
                        "requires OpenCV-CUDA support, which is not "
                        "available. Use detect(const cv::Mat&, ...) instead");
        return RESULT_FAILURE_OPENCV_NO_CUDA;
    }

    /**     Pre-processing      **/
    if(!_preprocessor->setup(_inputBinding.dims(), flags, _batchSize(),
                            (float*)_deviceMemory.at(_inputBinding.index()) ))
    {
        _logger->log(LOGGING_ERROR, "[Detector] detect() failure: could not "
                        "set up pre-processor");
        return RESULT_FAILURE_OTHER;
    }
    if(!_preprocessor->process(0, img, true))
    {
        _logger->log(LOGGING_ERROR, "[Detector] detect() failure: could not "
                        "pre-process input");
        return RESULT_FAILURE_OTHER;
    }

    return _detect(out);
}

Result Detector::detectBatch(const std::vector<cv::Mat>& images, 
                    std::vector<std::vector<Detection>>* out,
                    int flags) noexcept
{
    if(!isEngineLoaded())
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] detectBatch() failure: no "
                        "engine loaded");
        }
        return RESULT_FAILURE_NOT_LOADED;
    }
    if(images.size() == 0)
    {
        _logger->log(LOGGING_ERROR, "[Detector] detectBatch() failure: list "
                        "of inputs is empty");
        return RESULT_FAILURE_INVALID_INPUT;
    }
    if((int)images.size() > _batchSize())
    {
        _logger->logf(LOGGING_ERROR, "[Detector] detectBatch() failure: "
                        "specified %d images, but batch size is %i",
                        (unsigned int)images.size(), _batchSize());
        return RESULT_FAILURE_INVALID_INPUT;
    }
    const int numProcessed = MIN((int)images.size(), _batchSize());

    /**     Pre-processing      **/
    if(!_preprocessor->setup(_inputBinding.dims(), flags, _batchSize(), 
                            (float*)_deviceMemory.at(_inputBinding.index()) ))
    {
        _logger->log(LOGGING_ERROR, "[Detector] detectBatch() failure: could "
                        "not set up pre-processor");
        return RESULT_FAILURE_OTHER;
    }

    for(int i = 0; i < numProcessed; ++i)
    {
        if(!_preprocessor->process(i, images[i], i == numProcessed - 1))
        {
            _logger->logf(LOGGING_ERROR, "[Detector] detectBatch() "
                            "failure: preprocessing for image %i failed", i);
            return RESULT_FAILURE_OTHER;
        }
    }

    return _detectBatch(numProcessed, out);
}
    
Result Detector::detectBatch(const std::vector<cv::cuda::GpuMat>& images, 
                    std::vector<std::vector<Detection>>* out,
                    int flags) noexcept
{
    if(!isEngineLoaded())
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] detectBatch() failure: no "
                        "engine loaded");
        }
        return RESULT_FAILURE_NOT_LOADED;
    }

    if(!internal::opencvHasCuda())
    {
        _logger->log(LOGGING_ERROR, "[Detector] detectBatch() failure: this "
                        "method requires OpenCV-CUDA support, which is not "
                        "available. Use detectBatch(const "
                        "std::vector<cv::Mat>&, ...) instead");
        return RESULT_FAILURE_OPENCV_NO_CUDA;
    }
    if(images.size() == 0)
    {
        _logger->log(LOGGING_ERROR, "[Detector] detectBatch() failure: list "
                        "of inputs is empty");
        return RESULT_FAILURE_INVALID_INPUT;
    }
    if((int)images.size() > _batchSize())
    {
        _logger->logf(LOGGING_ERROR, "[Detector] detectBatch() failure: "
                        "specified %d images, but batch size is %i",
                        (unsigned int)images.size(), _batchSize());
        return RESULT_FAILURE_INVALID_INPUT;
    }
    const int numProcessed = MIN((int)images.size(), _batchSize());

    /**     Pre-processing      **/
    if(!_preprocessor->setup(_inputBinding.dims(), flags, _batchSize(), 
                            (float*)_deviceMemory.at(_inputBinding.index()) ))
    {
        _logger->log(LOGGING_ERROR, "[Detector] detectBatch() failure: could "
                        "not set up pre-processor");
        return RESULT_FAILURE_OTHER;
    }

    for(int i = 0; i < numProcessed; ++i)
    {
        if(!_preprocessor->process(i, images[i], i == numProcessed - 1))
        {
            _logger->logf(LOGGING_ERROR, "[Detector] detectBatch() "
                            "failure: preprocessing for image %i failed", i);
            return RESULT_FAILURE_OTHER;
        }
    }

    return _detectBatch(numProcessed, out);
}

double Detector::scoreThreshold() const noexcept
{
    return _scoreThreshold;
}

Result Detector::setScoreThreshold(const double& v) noexcept
{
    if(v < 0 || v > 1)
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] setScoreThreshold() "
                        "failure: invalid value specified");
        }
        return RESULT_FAILURE_INVALID_INPUT;
    }
    _scoreThreshold = v;
    return RESULT_SUCCESS;
}

double Detector::nmsThreshold() const noexcept
{
    return _nmsThreshold;
}

Result Detector::setNmsThreshold(const double& v) noexcept
{
    if(v < 0 || v > 1)
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] setNmsThreshold() "
                        "failure: invalid value specified");
        }
        return RESULT_FAILURE_INVALID_INPUT;
    }
    _nmsThreshold = v;
    return RESULT_SUCCESS;
}

int Detector::batchSize() const noexcept
{
    if(!isEngineLoaded())
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] batchSize() failure: no "
                        "engine loaded");
        }
        return 0;
    }
    return _batchSize();
}

cv::Size Detector::inferenceSize() const noexcept
{
    if(!isEngineLoaded())
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] inferenceSize() failure: "
                        "no engine loaded");
        }
        return cv::Size(0, 0);
    }
    const auto& inputDims = _inputBinding.dims();
    const int rows = inputDims.d[2];
    const int cols = inputDims.d[3];
    return cv::Size(cols, rows);
}

Result Detector::setLogger(std::shared_ptr<Logger> logger) noexcept
{
    if(!logger)
    {
        if(_logger)
        {
            _logger->log(LOGGING_ERROR, "[Detector] setLogger() failure: "
                        "provided logger is nullptr");
        }
        return RESULT_FAILURE_INVALID_INPUT;
    }
    _logger = logger;   /*  note: operator= of shared_ptr is marked noexcept */

    if(_preprocessor)
    {
        _preprocessor->setLogger(_logger);
    }
    return RESULT_SUCCESS;
}

std::shared_ptr<Logger> Detector::logger() const noexcept
{
    return _logger;
}

Result Detector::_loadEngine(const std::vector<char>& data) noexcept
{
    /*  Try to deserialize engine */
    _logger->log(LOGGING_INFO, "[Detector] Deserializing inference engine. "
                    "This may take a while...");
    std::unique_ptr<nvinfer1::ICudaEngine> engine(
            _trtRuntime->deserializeCudaEngine(data.data(), data.size()));
    if(!engine)
    {
        _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: could "
                    "not deserialize engine");
        return RESULT_FAILURE_TENSORRT_ERROR;
    }

    /*  Create execution context    */
    std::unique_ptr<nvinfer1::IExecutionContext> executionContext(
        engine->createExecutionContext());
    if(!executionContext)
    {
        _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: could "
                    "not create execution context");
        return RESULT_FAILURE_TENSORRT_ERROR;
    }
    
    _printBindings(engine);

    /*  Determine input bindings & verify that it matches what is expected  */
    internal::EngineBinding input;
    if(!internal::EngineBinding::setup(engine, "images", &input))
    {
        _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: could "
                    "not set up input binding");
        return RESULT_FAILURE_MODEL_ERROR;
    }
    if(input.dims().nbDims != 4)
    {
        std::string str;
        internal::dimsToString(input.dims(), &str);
        _logger->logf(LOGGING_ERROR, "[Detector] loadEngine() failure: "
                    "unexpected input dimensions: %s", str.c_str());
        return RESULT_FAILURE_MODEL_ERROR;
    }
    if(input.isDynamic())
    {
        _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: "
                    "input binding has dynamic dimensions. This is not "
                    "supported at this time!");
        return RESULT_FAILURE_MODEL_ERROR;
    }


    /*  Determine output binding & verify that it matches what is expected   */
    internal::EngineBinding output;
    if(!internal::EngineBinding::setup(engine, "output", &output))
    {
        _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: could "
                    "not set up output binding");
        return RESULT_FAILURE_MODEL_ERROR;
    }
    if(output.dims().nbDims != 3)
    {
        std::string str;
        internal::dimsToString(output.dims(), &str);
        _logger->logf(LOGGING_ERROR, "[Detector] loadEngine() failure: "
                    "unexpected output dimensions: %s", str.c_str());
        return RESULT_FAILURE_MODEL_ERROR;
    }
    if(output.isDynamic())
    {
        _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: "
                    "output binding has dynamic dimensions. This is not "
                    "supported at this time!");
        return RESULT_FAILURE_MODEL_ERROR;
    }

    /*  Set up Device memory for input & output */
    internal::DeviceMemory memory;
    const Result r = internal::DeviceMemory::setup(_logger, engine, &memory);
    if(r != RESULT_SUCCESS)
    {
        _logger->log(LOGGING_ERROR, "[Detector] loadEngine() failure: "
                    "could not set up device memory");
        return r;
    }

    /*  Set up memory on host for post-processing */
    std::vector<float> outputHostMemory;
    try
    {
        outputHostMemory.resize(output.volume());
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[Detector] loadEngine() failure: "
                    "could not set up output host memory: %s", e.what());
        return RESULT_FAILURE_ALLOC;
    }


    /*  commit to the new engine; at this point, there is nothing that
        will fail anymore  */
    if(isEngineLoaded())
    {
        _logger->log(LOGGING_INFO, "[Detector] loadEngine() info: an engine "
                    "is already loaded; Replacing it");
    }

    engine.swap(_trtEngine);
    executionContext.swap(_trtExecutionContext);
    
    memory.swap(_deviceMemory);
    outputHostMemory.swap(_outputHostMemory);

    input.swap(_inputBinding);
    output.swap(_outputBinding);

    /*  Note: this is the PreProcessor::reset() method, not the reset()
        method of unique_ptr (!)    */
    _preprocessor->reset();

    _logger->log(LOGGING_INFO, "[Detector] Successfully loaded inference "
                "engine");
    return RESULT_SUCCESS;
}

void Detector::_printBindings(
            const std::unique_ptr<nvinfer1::ICudaEngine>& engine) 
            const noexcept
{
    const int32_t nbBindings = engine->getNbBindings();

    for(int i = 0; i < nbBindings; ++i)
    {
        internal::EngineBinding binding;
        internal::EngineBinding::setup(engine, i, &binding);

        std::string str;
        binding.toString(&str);

        _logger->logf(LOGGING_DEBUG, "[Detector] loadEngine() info: Binding "
                    "%i - %s", i, str.c_str());
    }
}

int Detector::_batchSize() const noexcept
{
    return _inputBinding.dims().d[0];
}

int Detector::_numClasses() const noexcept
{
    return _outputBinding.dims().d[2] - 5;
}

Result Detector::_detect(std::vector<Detection>* out)
{
    /**     Inference     **/
    Result r = _inference("detect()");
    if(r != RESULT_SUCCESS)
    {
        return r;
    }


    /**     Post-processing     **/
    std::vector<Detection> lst;
    r = _decodeOutput("detect()", 0, &lst);
    if(r != RESULT_SUCCESS)
    {
        return r;
    }

    if(out != nullptr)
    {
        std::swap(lst, *out);
    }
    return RESULT_SUCCESS;
}

Result Detector::_detectBatch(const int& nrImages,
                    std::vector<std::vector<Detection>>* out)
{
    /**     Inference     **/
    Result r = _inference("detectBatch()");
    if(r != RESULT_SUCCESS)
    {
        return r;
    }


    /**     Post-processing     **/
    std::vector<std::vector<Detection>> lst;
    try
    {
        lst.resize(nrImages);
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[Detector] detectBatch() failure: could "
                        "not allocate output space: %s", e.what());
        return RESULT_FAILURE_ALLOC;
    }

    for(int i = 0; i < nrImages; ++i)
    {
        r = _decodeOutput("detectBatch()", i, &lst[i]);
        if(r != RESULT_SUCCESS)
        {
            return r;
        }
    }

    if(out != nullptr)
    {
        std::swap(lst, *out);
    }
    return RESULT_SUCCESS;
}

Result Detector::_inference(const char* logid)
{
    /*  Enqueue for inference   */
    if(!_trtExecutionContext->enqueueV2(_deviceMemory.begin(),
                                        _preprocessor->cudaStream(), nullptr))
    {
        _logger->logf(LOGGING_ERROR, "[Detector] %s failure: could not enqueue "
                    "data for inference", logid);
        return RESULT_FAILURE_TENSORRT_ERROR;
    }

    /*  Copy output back from device memory to host memory  */
    auto r = cudaMemcpyAsync(_outputHostMemory.data(), 
            _deviceMemory.at(_outputBinding.index()),
            (int)(_outputBinding.volume() * sizeof(float)), 
            cudaMemcpyDeviceToHost, _preprocessor->cudaStream());
    if(r != 0)
    {
        _logger->logf(LOGGING_ERROR, "[Detector] %s failure: could not set up "
                    "device-to-host transfer for output: %s", 
                    logid, cudaGetErrorString(r));
        return RESULT_FAILURE_CUDA_ERROR;
    }

    /*  Synchronize */
    if(!_preprocessor->synchronizeCudaStream())
    {
        return RESULT_FAILURE_CUDA_ERROR;
    }
    return RESULT_SUCCESS;
}

Result Detector::_decodeOutput(const char* logid, const int& index, 
                            std::vector<Detection>* out)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classes;

    const int nrClasses = numClasses();

    /*  Decode YoloV5 output    */
    const int numGridBoxes = _outputBinding.dims().d[1];
    const int rowSize = _outputBinding.dims().d[2];

    float* begin = _outputHostMemory.data() + index * numGridBoxes * rowSize;

    for(int i = 0; i < numGridBoxes; ++i)
    {
        float* ptr = begin + i * rowSize;

        const float objectness = ptr[4];
        if(objectness < _scoreThreshold)
        {
            continue;
        }

        /*  Get the class with the highest score attached to it */
        double maxClassScore = 0.0;
        int maxScoreIndex = 0;
        for(int i = 0; i < nrClasses; ++i)
        {
            const float& v = ptr[5 + i];
            if(v > maxClassScore)
            {
                maxClassScore = v;
                maxScoreIndex = i;
            }
        }
        const double score = objectness * maxClassScore;
        if(score < _scoreThreshold)
        {
            continue;
        }

        const float w = ptr[2];
        const float h = ptr[3];
        const float x = ptr[0] - w / 2.0;
        const float y = ptr[1] - h / 2.0;

        try
        {
            boxes.push_back(cv::Rect(x, y, w, h));
            scores.push_back(score);
            classes.push_back(maxScoreIndex);
        }
        catch(const std::exception& e)
        {
            _logger->logf(LOGGING_ERROR, "[Detector] %s failure: got "
                    "exception setting up model detection: %s", 
                    logid, e.what());
            return RESULT_FAILURE_ALLOC;
        }
    }


    /*  Apply non-max-suppression   */
    std::vector<int> indices;
    try
    {
        cv::dnn::NMSBoxes(boxes, scores, _scoreThreshold, _nmsThreshold, 
                            indices);
    }
    catch(const std::exception& e)
    {
        _logger->logf(LOGGING_ERROR, "[Detector] %s failure: got exception "
                    "applying OpenCV non-max-suppression: %s", 
                    logid, e.what());
        return RESULT_FAILURE_OPENCV_ERROR;
    }

    /*  Convert to Detection objects    */
    for(unsigned int i = 0; i < indices.size(); ++i)
    {
        const int& j = indices[i];
        /*  transform bounding box from network space to input space    */
        const cv::Rect bbox = _preprocessor->transformBbox(index, boxes[j]);
        const double score = MAX(0.0, MIN(1.0, scores[j]));
        try
        {
            out->push_back(Detection(classes[j], bbox, score));
        }
        catch(const std::exception& e)
        {
            _logger->logf(LOGGING_ERROR, "[Detector] %s failure: got "
                    "exception setting up Detection output: %s", 
                    logid, e.what());
            return RESULT_FAILURE_ALLOC;
        }

        if(_classes.isLoaded())
        {
            Detection& det = out->back();
            
            std::string className;
            _classes.getName(det.classId(), &className);
            det.setClassName(className);
        }
    }
    return RESULT_SUCCESS;
}


}   /*  namespace yolov5    */