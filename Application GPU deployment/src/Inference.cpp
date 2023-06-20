#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <array>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main()
{
    // ----- ONNX Model Variables Initialization ----- //
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);
    //Define Image dimension (W,H,C).
    constexpr int64_t width = 256;
    constexpr int64_t height = 256;
    constexpr int64_t num_channels = 3;
    //Define Output dimension (W,H,C=classes).
    constexpr int64_t num_classes = 3;
    //Define input and output flatten size.
    constexpr int64_t num_out_classes = num_classes * height * width;
    constexpr int64_t num_in_classes = num_channels * height * width;
    //Define path from where onnx model will be loaded.
    const wchar_t* model_directory = L"/PATH/FlareNet_simple.onnx";
    //Create ONNX Session with model.
    session = Ort::Session(env, model_directory, Ort::SessionOptions{ nullptr });
    //Define and initialize the Input and Output Tensors.
    const array<int64_t, 4> input_shape = { 1, height, width, num_channels };
    const array<int64_t, 4> output_shape = { 1, height, width, num_classes };
    float* input = new float[num_in_classes];
    float* results = new float[num_out_classes];
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto input_tensor = Ort::Value::CreateTensor<float>(mem_info, input, num_in_classes, input_shape.data(), input_shape.size());
    auto output_tensor = Ort::Value::CreateTensor<float>(mem_info, results, num_out_classes, output_shape.data(), output_shape.size());
    //Input and output model names.
    Ort::AllocatorWithDefaultOptions ort_alloc;
    char* input_name = session.GetInputName(0, ort_alloc);
    char* output_name = session.GetOutputName(0, ort_alloc);
    const array<const char*, 1> input_names = { input_name };
    const array<const char*, 1> output_names = { output_name };
    int mean_duration_inference = 0;

    // ----- Image Variables Initialization ----- //
    auto start_inference = high_resolution_clock::now();
    auto duration_inference = duration_cast<microseconds>(high_resolution_clock::now() - start_inference);
    for (int i=0; i < 1000; i++) {
        start_inference = high_resolution_clock::now();
        cv::Mat image = imread("/PATH/img_in_XXXjpg");
        //Resize image to 255x255.
        resize(image, image, Size(width, height));
        //Reshape image to 1D vector.
        image = image.reshape(1, 1);
        //Normailze image RGB values (between 0 and 1). Convert to vector<float> from cv::Mat.
        vector<float> image_vec;
        image.convertTo(image_vec, CV_32FC3, 1. / 255);
        copy(image_vec.begin(), image_vec.end(), input);
        //Run inference by calling ONNX session.
        try {
             session.Run(runOptions, input_names.data(), &input_tensor, 1, output_names.data(), &output_tensor, 1);
        }
        catch (Ort::Exception& e) {
             cout << e.what() << endl;
             return 1;
        }
        duration_inference = duration_cast<microseconds>(duration_inference + duration_cast<microseconds>(high_resolution_clock::now() - start_inference));
    }

     //Rstore RGB values back to normal scale (between 0 and 255).
     for (size_t col = 0; col < num_out_classes; col++) {
          results[col] = int(results[col] * 255);
     }
     
     //Write inference image into file.
     cv::Mat img_result = cv::Mat(256, 256, CV_32FC3, results);
     mean_duration_inference = duration_inference.count() / 1000;
     const std::string write_path = "/PATH/img_out_xxx.jpg";
     cv::imwrite(write_path, img_result);
     cout << "Average execution time (ms): " << mean_duration_inference << endl << endl;
}