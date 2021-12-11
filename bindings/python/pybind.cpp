/**
 * @file
 * 
 * @author      Noah van der Meer
 * @brief       Python Bindings for the yolov5-tensorrt library
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "yolov5_builder.hpp"
#include "yolov5_detector.hpp"

using namespace yolov5;

inline cv::Mat arrayToMat(const pybind11::array_t<uint8_t, 
                            pybind11::array::c_style | 
                            pybind11::array::forcecast>& img)
{
    const auto rows = img.shape(0);
    const auto columns = img.shape(1);
    const auto channels = img.shape(2);
    return cv::Mat(rows, columns, CV_8UC(channels), 
                    (unsigned char*)img.data());
}

PYBIND11_MODULE(yolov5tensorrt, m) 
{
    m.doc() = "yolov5-tensorrt python binding";


    /*
        yolov5_common.hpp
    */
    pybind11::enum_<Result>(m, "Result")
        .value("FAILURE_INVALID_INPUT", RESULT_FAILURE_INVALID_INPUT)
        .value("FAILURE_NOT_INITIALIZED", 
                    RESULT_FAILURE_NOT_INITIALIZED)
        .value("FAILURE_NOT_LOADED", RESULT_FAILURE_NOT_LOADED)
        .value("FAILURE_MODEL_ERROR", RESULT_FAILURE_MODEL_ERROR)
        .value("FAILURE_OPENCV_NO_CUDA", RESULT_FAILURE_OPENCV_NO_CUDA)
        .value("FAILURE_FILESYSTEM_ERROR", 
                    RESULT_FAILURE_FILESYSTEM_ERROR)
        .value("FAILURE_CUDA_ERROR", RESULT_FAILURE_CUDA_ERROR)
        .value("FAILURE_TENSORRT_ERROR", RESULT_FAILURE_TENSORRT_ERROR)
        .value("FAILURE_OPENCV_ERROR", RESULT_FAILURE_OPENCV_ERROR)
        .value("FAILURE_ALLOC", RESULT_FAILURE_ALLOC)
        .value("FAILURE_OTHER", RESULT_FAILURE_OTHER)
        .value("SUCCESS", RESULT_SUCCESS);
    m.def("result_to_string", 
        pybind11::overload_cast<Result>(&result_to_string),
        "Get a textual description of a result code");

    pybind11::enum_<Precision>(m, "Precision")
        .value("FP32", PRECISION_FP32)
        .value("FP16", PRECISION_FP16);
    m.def("precision_to_string", 
        pybind11::overload_cast<Precision>(&precision_to_string),
        "Get a textual description of a precision code");

    pybind11::enum_<DetectorFlag>(m, "DetectorFlag")
        .value("INPUT_BGR", INPUT_BGR)
        .value("INPUT_RGB", INPUT_RGB)
        .value("PREPROCESSOR_CVCUDA", PREPROCESSOR_CVCUDA)
        .value("PREPROCESSOR_CVCPU", PREPROCESSOR_CVCPU);


    /*
        yolov5_builder.hpp
    */
    pybind11::class_<Builder>(m, "Builder")
        .def(pybind11::init<>())
        .def("init", &Builder::init, "Initialize the Builder.")
        .def("buildEngine",
            [](const Builder& builder, 
                const std::string& inputPath, const std::string& outputPath,
                Precision precision)
            {
                return builder.buildEngine(inputPath, outputPath, precision);
            },
            pybind11::arg("inputPath"), pybind11::arg("outputPath"), 
            pybind11::arg("precision") = PRECISION_FP32,
            "Build an engine from ONNX model input, save it to disk"
        );


    /*
        yolov5_detection.hpp
    */
    pybind11::class_<Detection>(m, "Detection")
        .def(pybind11::init<>())
        .def("classId", &Detection::classId, 
            "Retrieve the class id of the detection")
        .def("boundingBox",
            [](const Detection& detection)
            {
                const cv::Rect& r = detection.boundingBox();
                return pybind11::make_tuple(r.x, r.y, r.width, r.height);
            },
            "Retrieve a bounding box of the detection"
        )
        .def("score", &Detection::score, 
            "Retrieve the score assigned to this detection")
        .def("className", &Detection::className,
            "Retrieve the name of the class of this detection, if known.")
        .def("setClassName", &Detection::setClassName,
            "Set the class name");

    m.def("visualizeDetection", 
        [](const Detection& detection, 
            pybind11::array_t<uint8_t, pybind11::array::c_style |
                                    pybind11::array::forcecast>& img,
            const std::tuple<int, int, int>& color,
            const double& fontScale)
        {
            const cv::Scalar colorScalar(std::get<0>(color), 
                                    std::get<1>(color), std::get<2>(color));
            cv::Mat mat = arrayToMat(img);
            return visualizeDetection(detection, &mat, colorScalar, fontScale);
        },
        pybind11::arg("detection"), pybind11::arg("img"), 
        pybind11::arg("color"), pybind11::arg("fontScale"),
        "Helper method for visualizing a Detection in an image"
    );

    pybind11::class_<Classes>(m, "Classes")
        .def(pybind11::init<>())
        .def("loadFromFile", &Classes::loadFromFile,
            "Try loading the class names as a list from a file")
        .def("isLoaded", &Classes::isLoaded,
            "Query whether the classes have been loaded")
        .def("getName", 
            [](const Classes& classes, const int& classId)
            {
                std::string name;
                const Result r = classes.getName(classId, &name);
                return pybind11::make_tuple(r, name);
            },
            "Get the Class name corresponding to a ClassId"
        );
    

    /*
        yolov5_detector.hpp
    */
    pybind11::class_<Detector>(m, "Detector")
        .def(pybind11::init<>())
        .def("init", &Detector::init, 
            pybind11::arg("flags") = 0,
            "Initialize the Detector.")
        .def("isInitialized", &Detector::isInitialized,
            "Query whether the Detector is initialized")
        .def("loadEngine", 
            [](Detector& detector, const std::string& filepath)
            {
                return detector.loadEngine(filepath);
            },
            "Load a TensorRT engine from a file"
        )
        .def("isEngineLoaded", &Detector::isEngineLoaded,
            "Query whether an inference engine has been loaded already")
        .def("numClasses", &Detector::numClasses,
            "Retrieve the number of classes of the engine/network")
        .def("setClasses", &Detector::setClasses,
            "Set the classes of the network")
        .def("detect", 
            [](Detector& detector, 
                const pybind11::array_t<uint8_t, pybind11::array::c_style |
                                    pybind11::array::forcecast>& img, 
                int flags = 0)
            {
                std::vector<Detection> detections;
                const Result r = 
                        detector.detect(arrayToMat(img), &detections, flags);
                return pybind11::make_tuple(r, detections);
            },
            pybind11::arg("img"),
            pybind11::arg("flags") = 0,
            "Detect objects in the specified image using the YoloV5 model"
        )
        .def("detectBatch", 
            [](Detector& detector, 
                const std::vector<
                    pybind11::array_t<uint8_t, pybind11::array::c_style |
                                        pybind11::array::forcecast>>& images, 
                int flags = 0)
            {
                std::vector<std::vector<Detection>> detections;

                std::vector<cv::Mat> mats;
                for(unsigned int i = 0; i < images.size(); ++i)
                {
                    mats.push_back(arrayToMat(images[i]));
                }
                const Result r = 
                        detector.detectBatch(mats, &detections, flags);
                return pybind11::make_tuple(r, detections);
            },
            pybind11::arg("images"),
            pybind11::arg("flags") = 0,
            "Detect objects in the specified images using batch inference "
                "with the YoloV5 model"
        )
        .def("scoreThreshold", &Detector::scoreThreshold,
            "Obtain the score threshold")
        .def("setScoreThreshold", &Detector::setScoreThreshold,
            "Set the Score threshold: used to filter objects by score")
        .def("nmsThreshold", &Detector::nmsThreshold,
            "Obtain the NMS threshold")
        .def("setNmsThreshold", &Detector::setNmsThreshold,
            "Set the NMS threshold")
        .def("batchSize", &Detector::batchSize,
            "Retrieve the batch size of the engine/network")
        .def("inferenceSize",
            [](Detector& detector)
            {
                const cv::Size s = detector.inferenceSize();
                return pybind11::make_tuple(s.width, s.height);
            },
            "Retrieve the input size for which the network was configured");
}
