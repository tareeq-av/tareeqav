#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <time.h>

#include <opencv2/opencv.hpp>


#include "darknet.h"

using namespace std; 
using namespace std::chrono; 
using namespace cv;

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: yolo-app <image path>\n";
        return -1;
    }

    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {        
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // input image size for YOLO v3
    int input_image_size = 416;

    Darknet net("/home/sameh/Autonomous-Vehicles/tareeqav/tareeq/perception/camera/yolov3/models/yolov3.cfg", &device);

    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << endl;
    net.load_weights("/home/sameh/Autonomous-Vehicles/tareeqav/tareeq/perception/camera/yolov3/models/yolov3.weights");
    std::cout << "weight loaded ..." << endl;
    
    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;
    
    Mat origin_image, resized_image;

    // origin_image = cv::imread("../139.jpg");
    origin_image = imread(argv[1]);
    
    cvtColor(origin_image, resized_image,  COLOR_RGB2BGR);
    resize(resized_image, resized_image, Size(input_image_size, input_image_size));

    Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0/255);

    //auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, input_image_size, input_image_size, 3});
    auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
    
    img_tensor = img_tensor.permute({0,3,1,2});

    // auto img_var = torch::autograd::make_variable(img_tensor.data, false).to(device);

    auto start = std::chrono::high_resolution_clock::now();
       
    auto output = net.forward(img_tensor);
    
    // filter result by NMS 
    // class_num = 80
    // confidence = 0.6
    auto result = net.write_results(output, 80, 0.6, 0.4);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start); 

    // It should be known that it takes longer time at first time
    std::cout << "inference taken : " << duration.count() << " ms" << endl; 

    if (result.dim() == 1)
    {
        std::cout << "no object found" << endl;
    }
    else
    {
        int obj_num = result.size(0);

        std::cout << obj_num << " objects found" << endl;

        float w_scale = float(origin_image.cols) / input_image_size;
        float h_scale = float(origin_image.rows) / input_image_size;

        result.select(1,1).mul_(w_scale);
        result.select(1,2).mul_(h_scale);
        result.select(1,3).mul_(w_scale);
        result.select(1,4).mul_(h_scale);

        auto result_data = result.accessor<float, 2>();

        for (int i = 0; i < result.size(0) ; i++)
        {
            rectangle(origin_image, Point(result_data[i][1], result_data[i][2]), Point(result_data[i][3], result_data[i][4]), Scalar(0, 0, 255), 1, 1, 0);
        }

        imwrite("/home/sameh/Autonomous-Vehicles/tareeqav/tareeq/perception/camera/yolov3/tests/out-det.jpg", origin_image);
    }

    std::cout << "Done" << endl;
    
    return 0;
}
