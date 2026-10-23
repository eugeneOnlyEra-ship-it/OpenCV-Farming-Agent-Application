#pragma once
#include <string>
#include <iostream>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn/all_layers.hpp>
#include<opencv2/dnn.hpp>

// A plain, ordinary C++ class - no pybind11 awareness needed here at all.
// The binding layer in pod_capture_bindings.cpp is what makes this
// callable from Python; this header itself stays completely portable
// and could be used from any other C++ code just as easily.
class NewComponent
{
public:
    std::string greet() {
        
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;

cv::VideoCapture cap(0);  // 0 = default/main camera
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open camera." << std::endl;
        return "Error: could not open camera.";
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Camera Feed", frame);

        int key = cv::waitKey(30);  // ~30 fps poll
        if (key == 13) break;       // Enter key
    }

    cap.release();
    cv::destroyAllWindows();

        return "Hello, "; }
};