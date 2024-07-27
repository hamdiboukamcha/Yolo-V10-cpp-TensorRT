#include "YOLOv10.hpp"

YOLOv10::YOLOv10() {}

YOLOv10::~YOLOv10() {}

void YOLOv10::preProcess(cv::Mat* img, int length, float* factor, std::vector<float>& data) {
       // Create a new cv::Mat object for storing the image after conversion
    cv::Mat mat;

    // Get the dimensions and number of channels of the input image
    int rh = img->rows;  // Height of the input image
    int rw = img->cols;  // Width of the input image
    int rc = img->channels();  // Number of channels (e.g., 3 for RGB)

    // Convert the input image from BGR to RGB color space
    cv::cvtColor(*img, mat, cv::COLOR_BGR2RGB);

    // Determine the size of the new square image (largest dimension of the input image)
    int maxImageLength = rw > rh ? rw : rh;

    // Create a new square image filled with zeros (black) with dimensions maxImageLength x maxImageLength
    cv::Mat maxImage = cv::Mat::zeros(maxImageLength, maxImageLength, CV_8UC3);

    // Set all pixels to 255 (white)
    maxImage = maxImage * 255;

    // Define a Region of Interest (ROI) that covers the entire original image
    cv::Rect roi(0, 0, rw, rh);

    // Copy the original image into the ROI of the new square image
    mat.copyTo(cv::Mat(maxImage, roi));

    // Create a new cv::Mat object for storing the resized image
    cv::Mat resizeImg;

    // Resize the square image to the specified dimensions (length x length)
    cv::resize(maxImage, resizeImg, cv::Size(length, length), 0.0f, 0.0f, cv::INTER_LINEAR);

    // Calculate the scaling factor and store it in the 'factor' variable
    *factor = (float)((float)maxImageLength / (float)length);

    // Convert the resized image to floating-point format with values in range [0, 1]
    resizeImg.convertTo(resizeImg, CV_32FC3, 1 / 255.0);

    // Update the height, width, and number of channels for the resized image
    rh = resizeImg.rows;
    rw = resizeImg.cols;
    rc = resizeImg.channels();

    // Extract each channel of the resized image and store it in the 'data' vector
    for (int i = 0; i < rc; ++i) {
        // Extract the i-th channel and store it in the appropriate part of the 'data' vector
        cv::extractChannel(resizeImg, cv::Mat(rh, rw, CV_32FC1, data.data() + i * rh * rw), i);
    }
}

std::vector<DetResult> YOLOv10::postProcess(float* result, float factor, int outputLength) {
    // Vectors to store the detection results
std::vector<cv::Rect> positionBoxes;  // Stores bounding boxes of detected objects
std::vector<int> classIds;            // Stores class IDs for detected objects
std::vector<float> confidences;       // Stores confidence scores for detected objects

// Process each detection result from the output
for (int i = 0; i < outputLength; i++) {
    // Compute the starting index for the current detection result in the 'result' array
    int s = 6 * i;
    
    // Check if the confidence score of the detection is above a threshold (0.2 in this case)
    if ((float)result[s + 4] > 0.2) {
        // Extract the coordinates and dimensions of the bounding box (normalized values)
        float cx = result[s + 0];  // Center x-coordinate
        float cy = result[s + 1];  // Center y-coordinate
        float dx = result[s + 2];  // Bottom-right x-coordinate
        float dy = result[s + 3];  // Bottom-right y-coordinate
        
        // Convert normalized coordinates and dimensions to pixel values using the scaling factor
        int x = (int)((cx) * factor);           // Top-left x-coordinate of the bounding box
        int y = (int)((cy) * factor);           // Top-left y-coordinate of the bounding box
        int width = (int)((dx - cx) * factor);  // Width of the bounding box
        int height = (int)((dy - cy) * factor); // Height of the bounding box
        
        // Create a cv::Rect object to represent the bounding box
        cv::Rect box(x, y, width, height);
        
        // Store the bounding box, class ID, and confidence score in the corresponding vectors
        positionBoxes.push_back(box);
        classIds.push_back((int)result[s + 5]); // Class ID is stored at position s + 5 in the 'result' array
        confidences.push_back((float)result[s + 4]); // Confidence score is stored at position s + 4 in the 'result' array
    }
}

// Vector to store the final detection results
std::vector<DetResult> re;

// Convert the extracted detection information into DetResult objects and store them in the 're' vector
for (int i = 0; i < positionBoxes.size(); i++) {
    DetResult det(positionBoxes[i], confidences[i], classIds[i]);
    re.push_back(det);
}

// Return the vector of DetResult objects
return re;
}

void YOLOv10::drawBbox(cv::Mat& img, std::vector<DetResult>& res) {
   // Iterate through each result in the 'res' vector
for (size_t j = 0; j < res.size(); j++) {
    // Draw a rectangle around the detected object using the bounding box (bbox)
    cv::rectangle(img, res[j].bbox, cv::Scalar(255, 0, 255), 2);
    // Add text label and confidence score near the top-left corner of the bounding box
    cv::putText(
        img,  // The image on which to draw
        std::to_string(res[j].label) + "-" + std::to_string(res[j].conf), // Text to display: label and confidence score
        cv::Point(res[j].bbox.x, res[j].bbox.y - 1), // Position of the text (slightly above the top-left corner of the bounding box)
        cv::FONT_HERSHEY_PLAIN, // Font type
        1.2, // Font size
        cv::Scalar(0, 0, 255), // Text color (red)
        2 // Thickness of the text
    );
}
}

std::shared_ptr<nvinfer1::IExecutionContext> YOLOv10::createExecutionContext(const std::string& modelPath) {
      // Open the model file in binary mode
    std::ifstream filePtr(modelPath, std::ios::binary);
    
    // Check if the file was opened successfully
    if (!filePtr.good()) {
        std::cerr << "File cannot be opened, please check the file!" << std::endl;
        return nullptr;  // Return nullptr if the file cannot be opened
    }

    // Determine the size of the file
    size_t size = 0;
    filePtr.seekg(0, filePtr.end);  // Move to the end of the file
    size = filePtr.tellg();         // Get the current position, which is the size of the file
    filePtr.seekg(0, filePtr.beg);  // Move back to the beginning of the file

    // Allocate memory to hold the file contents
    char* modelStream = new char[size];
    filePtr.read(modelStream, size);  // Read the entire file into the allocated memory
    filePtr.close();  // Close the file after reading

    // Create an instance of nvinfer1::IRuntime
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "Failed to create runtime" << std::endl;
        delete[] modelStream;  // Free the allocated memory
        return nullptr;  // Return nullptr if the runtime creation fails
    }

    // Deserialize the model and create an ICudaEngine
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream, size);
    delete[] modelStream;  // Free the allocated memory
    if (!engine) {
        std::cerr << "Failed to create engine" << std::endl;
        runtime->destroy();  // Clean up runtime
        return nullptr;  // Return nullptr if the engine creation fails
    }

    // Create an execution context from the engine
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        engine->destroy();  // Clean up engine
        return nullptr;  // Return nullptr if the execution context creation fails
    }

    // Return a shared pointer to the execution context with a custom deleter
    return std::shared_ptr<nvinfer1::IExecutionContext>(context, [](nvinfer1::IExecutionContext* ctx) {
        ctx->destroy();  // Clean up the execution context
    });
}

void YOLOv10::inferVideo(const std::string& videoPath, const std::string& enginePath) {
    // Create a shared pointer to the execution context using the provided engine path
std::shared_ptr<nvinfer1::IExecutionContext> context = createExecutionContext(enginePath);

// Open the video file using OpenCV's VideoCapture
cv::VideoCapture capture(videoPath);

// Check if the video file was opened successfully
if (!capture.isOpened()) {
    std::cerr << "ERROR: Could not open video file." << std::endl;
    return;  // Exit if the video file cannot be opened
}

// Create a CUDA stream for asynchronous operations
cudaStream_t stream;
cudaStreamCreate(&stream);

// Allocate device memory for input and output data
void* inputSrcDevice;
void* outputSrcDevice;
cudaMalloc(&inputSrcDevice, 3 * 640 * 640 * sizeof(float));  // For storing input data (3 channels, 640x640 size)
cudaMalloc(&outputSrcDevice, 1 * 300 * 6 * sizeof(float));   // For storing output data (300 detections, each with 6 values)

// Create vectors to hold input and output data on the host (CPU)
std::vector<float> output_data(300 * 6);  // Buffer to hold output data from the device
std::vector<float> inputData(640 * 640 * 3);  // Buffer to hold input data (640x640 image with 3 channels)

// Process video frames in a loop
while (true) {
    cv::Mat frame;
    
    // Read the next frame from the video file
    if (!capture.read(frame)) {
        break;  // Exit the loop if no more frames are available
    }

    float factor = 0;
    // Preprocess the frame (resize, normalize, etc.) and store the result in inputData
    preProcess(&frame, 640, &factor, inputData);

    // Copy the preprocessed input data from host to device memory
    cudaMemcpyAsync(inputSrcDevice, inputData.data(), 3 * 640 * 640 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Set up bindings for TensorRT inference (input and output tensors)
    void* bindings[] = { inputSrcDevice, outputSrcDevice };
    
    // Perform inference using the TensorRT execution context
    context->enqueueV2(bindings, stream, nullptr);

    // Copy the output data from device to host memory
    cudaMemcpyAsync(output_data.data(), outputSrcDevice, 300 * 6 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    
    // Wait for the CUDA stream operations to complete
    cudaStreamSynchronize(stream);

    // Post-process the output data to extract detection results
    std::vector<DetResult> result = postProcess(output_data.data(), factor, 300);

    // Draw bounding boxes and annotations on the frame
    drawBbox(frame, result);

    // Display the annotated frame in a window
    cv::imshow("RESULT", frame);
    
    // Wait for 10 milliseconds or until a key is pressed
    cv::waitKey(10);
}

// Destroy all OpenCV windows after processing is complete
cv::destroyAllWindows();

// Free the allocated CUDA memory and destroy the CUDA stream
cudaFree(inputSrcDevice);
cudaFree(outputSrcDevice);
cudaStreamDestroy(stream);
}

void YOLOv10::inferImage(const std::string& imagePath, const std::string& enginePath) {
    // Create an execution context from the TensorRT engine file
    std::shared_ptr<nvinfer1::IExecutionContext> context = createExecutionContext(enginePath);

    // Load the image from the specified path using OpenCV
    cv::Mat img = cv::imread(imagePath);

    // Check if the image was loaded successfully
    if (img.empty()) {
        std::cerr << "ERROR: Could not open or find the image." << std::endl;
        return;
    }

    // Preprocess the image: resize, normalize, etc.
    // `inputData` is the buffer where the preprocessed image data will be stored
    float factor = 0; // Factor for scaling the image, if needed
    std::vector<float> inputData(640 * 640 * 3); // Buffer for the input data
    preProcess(&img, 640, &factor, inputData);

    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate device memory for input and output data
    void* inputSrcDevice;
    void* outputSrcDevice;
    cudaMalloc(&inputSrcDevice, 3 * 640 * 640 * sizeof(float)); // Input buffer
    cudaMalloc(&outputSrcDevice, 1 * 300 * 6 * sizeof(float));  // Output buffer

    // Copy the preprocessed input data from host to device
    cudaMemcpyAsync(inputSrcDevice, inputData.data(), 3 * 640 * 640 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Set up the input and output bindings for the TensorRT execution context
    void* bindings[] = { inputSrcDevice, outputSrcDevice };
    
    // Execute the TensorRT inference
    context->enqueueV2(bindings, stream, nullptr);

    // Allocate buffer to store the output data
    std::vector<float> output_data(300 * 6);
    cudaMemcpyAsync(output_data.data(), outputSrcDevice, 300 * 6 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    
    // Synchronize the stream to ensure all operations are complete
    cudaStreamSynchronize(stream);

    // Post-process the output data to extract detection results
    std::vector<DetResult> result = postProcess(output_data.data(), factor, 300);
    
    // Draw bounding boxes on the image
    drawBbox(img, result);

    // Display the image with results using OpenCV
    cv::imshow("RESULT", img);
    cv::waitKey(0); // Wait indefinitely for a key press

    // Free device memory
    cudaFree(inputSrcDevice);
    cudaFree(outputSrcDevice);

    // Destroy the CUDA stream
    cudaStreamDestroy(stream);
}

void YOLOv10::convertOnnxToEngine(const std::string& onnxFile, int memorySize) {
   // Check if the ONNX file exists
    std::ifstream ifile(onnxFile);
    if (!ifile) {
        // Print an error message if the file cannot be opened
        std::cerr << "Error: Could not open file " << onnxFile << std::endl;
        return;
    }

    // Extract the directory path and file name from the given ONNX file path
    std::string path(onnxFile);
    std::string::size_type iPos = (path.find_last_of('\\') + 1) == 0 ? path.find_last_of('/') + 1 : path.find_last_of('\\') + 1;
    std::string modelPath = path.substr(0, iPos); // Directory path
    std::string modelName = path.substr(iPos, path.length() - iPos); // File name with extension
    std::string modelName_ = modelName.substr(0, modelName.rfind(".")); // File name without extension
    std::string engineFile = modelPath + modelName_ + ".engine"; // Output file path for TensorRT engine

    // Create a TensorRT builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    // Define the network with explicit batch size
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // Create an ONNX parser for the network
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    // Parse the ONNX file into the network definition
    if (!parser->parseFromFile(onnxFile.c_str(), 2)) {
        // Print errors if parsing fails
        std::cerr << "Error: Failed to parse ONNX model from file: " << onnxFile << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "Parser error: " << parser->getError(i)->desc() << std::endl;
        }
        return;
    }
    std::cout << "TensorRT loaded ONNX model successfully." << std::endl;

    // Create a builder configuration
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // Set the maximum workspace size for TensorRT engine (in bytes)
    config->setMaxWorkspaceSize(1024 * 1024 * memorySize);
    // Set the builder flag to enable FP16 precision
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // Build the TensorRT engine with the network and configuration
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // Try to save the built engine to a file
    std::cout << "Trying to save engine file now..." << std::endl;
    std::ofstream filePtr(engineFile, std::ios::binary);
    if (!filePtr) {
        // Print an error message if the engine file cannot be opened
        std::cerr << "Error: Could not open plan output file: " << engineFile << std::endl;
        return;
    }

    // Serialize the engine to a memory stream and write it to the file
    nvinfer1::IHostMemory* modelStream = engine->serialize();
    filePtr.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

    // Clean up resources
    modelStream->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();

    // Print a success message
    std::cout << "Converted ONNX model to TensorRT engine model successfully!" << std::endl;
}