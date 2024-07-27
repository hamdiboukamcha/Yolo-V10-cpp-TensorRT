#ifndef YOLOV10_HPP
#define YOLOV10_HPP

#include "opencv2/opencv.hpp"
#include "cuda.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

/**
 * @brief Logger class for TensorRT
 * 
 * This class inherits from nvinfer1::ILogger and provides a logging mechanism for TensorRT.
 */
class Logger : public nvinfer1::ILogger {
public:
    /**
     * @brief Logs a message with a given severity.
     * 
     * @param severity The severity level of the message.
     * @param msg The message to log.
     */
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

/**
 * @brief Structure to hold detection results.
 * 
 * This structure holds the bounding box, confidence score, and label of a detection result.
 */
struct DetResult {
    cv::Rect bbox; ///< Bounding box of the detected object.
    float conf;    ///< Confidence score of the detection.
    int label;     ///< Label of the detected object.

    /**
     * @brief Constructs a DetResult object.
     * 
     * @param bbox Bounding box of the detected object.
     * @param conf Confidence score of the detection.
     * @param label Label of the detected object.
     */
    DetResult(cv::Rect bbox, float conf, int label) : bbox(bbox), conf(conf), label(label) {}
};

/**
 * @brief Class for YOLOv10 object detection.
 * 
 * This class provides methods for preprocessing input images, postprocessing detection results, 
 * drawing bounding boxes, creating execution contexts, and performing inference on video streams.
 */
class YOLOv10 {
public:
    /**
     * @brief Constructs a YOLOv10 object.
     */
    YOLOv10();

    /**
     * @brief Destructs a YOLOv10 object.
     */
    ~YOLOv10();

    /**
     * @brief Preprocesses an input image.
     * 
     * @param img Pointer to the input image.
     * @param length Length of the image.
     * @param factor Scaling factor.
     * @param data Vector to hold the preprocessed data.
     */
    void preProcess(cv::Mat* img, int length, float* factor, std::vector<float>& data);

    /**
     * @brief Postprocesses the detection results.
     * 
     * @param result Pointer to the detection results.
     * @param factor Scaling factor.
     * @param outputLength Length of the output results.
     * @return std::vector<DetResult> Vector of detection results.
     */
    std::vector<DetResult> postProcess(float* result, float factor, int outputLength);

    /**
     * @brief Draws bounding boxes on an image.
     * 
     * @param img Reference to the image.
     * @param res Vector of detection results.
     */
    void drawBbox(cv::Mat& img, std::vector<DetResult>& res);

    /**
     * @brief Creates an execution context from a model path.
     * 
     * @param modelPath Path to the model.
     * @return std::shared_ptr<nvinfer1::IExecutionContext> Shared pointer to the execution context.
     */
    std::shared_ptr<nvinfer1::IExecutionContext> createExecutionContext(const std::string& modelPath);

    /**
     * @brief Performs inference on a video stream.
     * 
     * @param videoPath Path to the video file.
     * @param enginePath Path to the TensorRT engine file.
     */
    void inferVideo(const std::string& videoPath, const std::string& enginePath);
	
	/**
	* @brief Performs inference on an image.
	* 
	* @param imagePath Path to the image file.
	* @param enginePath Path to the TensorRT engine file.
	*/
	void inferImage(const std::string& imagePath, const std::string& enginePath);

    /**
     * @brief Converts an ONNX model to a TensorRT engine.
     * 
     * @param onnxFile Path to the ONNX file.
     * @param memorySize Size of the memory allocated for the engine.
     */
    void convertOnnxToEngine(const std::string& onnxFile, int memorySize);

private:
    Logger logger; ///< Logger instance for TensorRT.
};

#endif // YOLOV10_HPP
