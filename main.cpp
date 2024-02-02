#include <iostream>
#include <string>
#include <tuple>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "inference.h"
#include "depth_anything.h"
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

bool IsPathExist(const std::string& path) {
    return (access(path.c_str(), F_OK) == 0);
}
bool IsFile(const std::string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

// 从文件名中提取数字
int extractNumber(const std::string& s) {
    size_t start = s.find("imL_") + 4;  // 找到 "imL_" 后面的位置
    size_t end = s.find(".png", start);  // 找到 ".png" 的位置
    return std::stoi(s.substr(start, end - start));
}


/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv)
{
    const std::string engine_file_path{ argv[1] };
    const std::string path{ argv[2] };
    std::vector<std::string> imagePathList;
    bool                     isVideo{ false };
    assert(argc == 3);

    if (IsFile(path)) {
    std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png") 
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv") 
        {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsPathExist(path)) 
    {
        std::cout<<"IsPathExist."<<std::endl;
        cv::glob(path + "/imL*.png", imagePathList);
        // 按照文件名中数字的顺序排序
        std::sort(imagePathList.begin(), imagePathList.end(), [](const cv::String& a, const cv::String& b) {
            return extractNumber(a) < extractNumber(b);
        });
    }
    // Assume it's a folder, add logic to handle folders
    // init model
    DepthAnything depth_model(engine_file_path, logger);

    if (isVideo) {
        //path to video
        string VideoPath = path;
        // open cap
        cv::VideoCapture cap(VideoPath);

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // Create a VideoWriter object to save the processed video
        // std::cout<<"width is "<<width<<", height is "<<height<<std::endl;
        cv::VideoWriter output_video("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height));
        while (1)
        {
            cv::Mat frame;
            cv::Mat show_frame;
            cap >> frame;

            if (frame.empty())
                break;
            frame.copyTo(show_frame);
            cv::Mat new_frame;
            frame.copyTo(new_frame);
            auto start = std::chrono::system_clock::now();
            cv::Mat result_d = inference(frame, depth_model);
            auto end = chrono::system_clock::now();
            cout << "Time of per frame: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
            // addWeighted(show_frame, 0.7, result_d, 0.3, 0.0, show_frame);
            cv::Mat result;
            cv::hconcat(result_d, show_frame, result);
            cv::resize(result, result, cv::Size(1280, 480));
            imshow("depth_result", result);
            output_video.write(result_d);
            cv::waitKey(1);
        }

        // Release resources
        cv::destroyAllWindows();
        cap.release();
        output_video.release();
    }
    else {
        // path to folder saves images
        string imageFolderPath_out = "out_dir/";
        for (const auto& imagePath : imagePathList)
        {
            // open image
            cv::Mat frame = cv::imread(imagePath);
            // std::cout<<"width is "<<frame.cols<<", height is "<<frame.rows<<std::endl;
            if (frame.empty())
            {
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }
            cv::Mat show_frame;
            show_frame = frame.clone();
            // frame.copyTo(show_frame);

            auto start = chrono::system_clock::now();
            cv::Mat result_d = inference(frame, depth_model);
            auto end = chrono::system_clock::now();
            cout << "Time of per frame: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
            // addWeighted(show_frame, 0.7, result_d, 0.3, 0.0, show_frame);
            cv::Mat result;
            cv::hconcat(result_d, show_frame, result);
            cv::resize(result, result, cv::Size(1280, 480));
            imshow("depth_result", result);
            cv::waitKey(1);

            std::istringstream iss(imagePath);
            std::string token;
            while (std::getline(iss, token, '/'))
            {
            }
            cv::imwrite(imageFolderPath_out + token, result);
            //std::cout << imageFolderPath_out + token << std::endl;
        }
    }
    return 0;
}

