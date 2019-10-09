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
    // origin_image = imread(argv[1]);

    // Create a VideoCapture object and use camera to capture the video
    VideoCapture cap(0); 
    
    // Check if camera opened successfully
    if(!cap.isOpened())
    {
        cout << "Error opening video stream" << endl; 
        return -1; 
    } 
    
    // Default resolution of the frame is obtained.The default resolution is system dependent. 
    int frame_width = cap.set(CAP_PROP_FRAME_WIDTH, 640); 
    int frame_height = cap.set(CAP_PROP_FRAME_HEIGHT, 480); 
    
    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file. 
    VideoWriter video("outcpp.avi",VideoWriter::fourcc('M','J','P','G'), 20, Size(640,480)); 


    while(1)
    { 
        Mat origin_image, resized_image;
        
        // Capture frame-by-frame 
        cap >> origin_image;
    
        // If the frame is empty, break immediately
        if (origin_image.empty())
            break;

        cvtColor(origin_image, resized_image,  COLOR_RGB2BGR);
        resize(resized_image, resized_image, Size(input_image_size, input_image_size));

        Mat img_float;
        resized_image.convertTo(img_float, CV_32F, 1.0/255);

        auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
        
        img_tensor = img_tensor.permute({0,3,1,2});

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

            // imwrite("/home/sameh/Autonomous-Vehicles/tareeqav/tareeq/perception/camera/yolov3/tests/out-det.jpg", origin_image);

            // Write the frame into the file 'outcpp.avi'
            video.write(origin_image);
            imshow( "Frame", origin_image );

        }


        
        
        
        // Display the resulting frame    
        
    
        // Press  ESC on keyboard to  exit
        char c = (char)waitKey(1);
        if( c == 27 ) 
            break;
    }
 
    // When everything done, release the video capture and write object
    cap.release();
    video.release();
    
    // Closes all the windows
    destroyAllWindows();
    
    std::cout << "Done" << endl;
    
    return 0;
}
