#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/ocl.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"
#include <curl/curl.h>
#include <cmath>

using namespace cv;
using namespace std;

int model_width;
int model_height;
int model_channels;

std::vector<std::string> Labels;
std::unique_ptr<tflite::Interpreter> interpreter;

static bool getFileContent(std::string fileName)
{
	// Open the File
	std::ifstream in(fileName.c_str());
	// Check if object is valid
	if(!in.is_open()) return false;

	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size()>0) Labels.push_back(str);
	}
	// Close The File
	in.close();
	return true;
}

size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
}

bool downloadImage(const std::string& url, const std::string& filename) {
	CURL* curl;
	CURLcode res;
	std::string readBuffer;

	curl = curl_easy_init();
	if(curl) {
		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		res = curl_easy_perform(curl);
		curl_easy_cleanup(curl);

		if(res == CURLE_OK) {
			std::ofstream outFile(filename, std::ios::binary);
			outFile.write(readBuffer.c_str(), readBuffer.size());
			outFile.close();
			return true;
		}
	}
	return false;
}

int main(int argc,char ** argv)
{
    int f;
    int In;
    Mat frame;
    Mat image;
    chrono::steady_clock::time_point Tbegin, Tend;

    const char* model_path = "/home/evanh/TensorFlow_Lite_Classification_RPi_zero/inception_v4.tflite";

    std::cout << "Loading model..." << std::endl;
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    std::cout << "Model loaded successfully" << std::endl;

    std::cout << "----------------" << std::endl;
    std::cout << "ClassifyBot v1.0" << std::endl;
    std::cout << "----------------" << std::endl;

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    interpreter->AllocateTensors();
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      //quad core

    // Get input dimension from the input tensor metadata
    // Assuming one input only
    In = interpreter->inputs()[0];
    model_height   = interpreter->tensor(In)->dims->data[1];
    model_width    = interpreter->tensor(In)->dims->data[2];
    model_channels = interpreter->tensor(In)->dims->data[3];

    cout << "height   : "<< model_height << endl;
    cout << "width    : "<< model_width << endl;
    cout << "channels : "<< model_channels << endl;

    cout << "Image URL: ";
    std::string url;
    cin >> url;

    std::string filename = "downloaded_image.jpg";

    if(downloadImage(url, filename)) {
	    std::cout << "Image downloaded successfully." << std::endl;
    } else{
	    std::cout << "Failed to download image." << std::endl;
    }

    // Get the names
    bool result = getFileContent("/home/evanh/TensorFlow_Lite_Classification_RPi_zero/labels.txt");
    if(!result){
        cout << "loading labels failed";
        exit(-1);
    }

    frame=imread(filename);  //need to refresh frame before dnn class detection
    if (frame.empty()) {
        cerr << "Can not load picture!" << endl;
        exit(-1);
    }

    // copy image to input as input tensor
    cv::resize(frame, image, Size(model_width,model_height),INTER_NEAREST);
    memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());

    cout << "tensors size: " << interpreter->tensors_size() << "\n";
    cout << "nodes size: " << interpreter->nodes_size() << "\n";
    cout << "inputs: " << interpreter->inputs().size() << "\n";
    cout << "outputs: " << interpreter->outputs().size() << "\n";

    Tbegin = chrono::steady_clock::now();

    interpreter->Invoke();      // run your model

    Tend = chrono::steady_clock::now();

    const float threshold = 0.001f;

    std::vector<std::pair<float, int>> top_results;

    int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto output_size = output_dims->data[output_dims->size - 1];
    cout << "output_size: " << output_size <<"\n";

    switch (interpreter->tensor(output)->type) {
        case kTfLiteFloat32:
            tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                                                    5, threshold, &top_results, kTfLiteFloat32);
        break;
        case kTfLiteUInt8:
            tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), output_size,
                                                    5, threshold, &top_results, kTfLiteUInt8);
        break;
        default:
            cerr << "cannot handle output type " << interpreter->tensor(output)->type << endl;
            exit(-1);
  }

    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        cout << confidence << " : " << Labels[index] << "\n";
    }
    //calculate time
    f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
    cout << "Process time: " << f << " mSec" << endl;

    return 0;
}
