/**
 * @file ImageUtil.cpp
 * @date 2017-02-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <time.h>
#include <sys/time.h>

#include <string>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "SysLog.h"
#include "Param.h"
#include "ImageUtil.h"
#include "FileMgmt.h"
#include "IO.h"

using namespace std;

template<typename Dtype>
static cv::Mat makeImage(const Dtype* data, int nthImage, int channel, int row, int col);

template<typename Dtype>
cv::Mat makeImage(const Dtype* data, int nthImage, int channel, int row, int col) {

    SASSERT0((channel == 3) || (channel == 1));

    cv::Mat testImage(row, col, CV_32FC(channel));
    int channelElemCount = row * col;
    int imageElemCount = channelElemCount * channel;
    int baseIndex = imageElemCount * nthImage;

    // HEIM : Channel, 값 범위에 상관없이 mat을 잘 표시 할 수 있도록 수정요
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int index = i * col + j;

            if (channel == 3) {
                testImage.at<cv::Vec3f>(i, j)[0] = 
                        (int)(data[baseIndex + index + channelElemCount * 0]);
                testImage.at<cv::Vec3f>(i, j)[1] = 
                        (int)(data[baseIndex + index + channelElemCount * 1]);
                testImage.at<cv::Vec3f>(i, j)[2] = 
                        (int)(data[baseIndex + index + channelElemCount * 2]);
            } else {
                testImage.at<float>(i, j) = 
                   (int)(data[baseIndex + index]); 
            }
        }
    }


		// Dtype* matData;
		// if (channel == 3) {
		// 	cv::Mat src_split[3];
		// 	src_split[0] = cv::Mat(row, col, CV_32FC1);
		// 	Dtype* data_0 = (Dtype*)src_split[0] .data;
		// 	std::memcpy(data_0, data, 1 * row* col * sizeof(Dtype));

		// 	src_split[1] = cv::Mat(row, col, CV_32FC1);
		// 	Dtype* data_1 = (Dtype*)src_split[1].data;
		// 	std::memcpy(data_1, data + 1 * row* col, 1 * row* col * sizeof(Dtype));

		// 	src_split[2] = cv::Mat(row, col, CV_32FC1);
		// 	Dtype* data_2 = (Dtype*)src_split[2].data;
		// 	std::memcpy(data_2, data + 2 * row* col, 1 * row* col * sizeof(Dtype));

		// 	merge(src_split, 3, testImage);
		// 	matData = (Dtype*)testImage.data;

		// } else {
		// 	testImage = cv::Mat(row, col, CV_32FC1);
		// 	matData = (Dtype*)testImage.data;
		// 	std::memcpy(matData, data, channel * row* col * sizeof(Dtype));
		// }

    cv::normalize(testImage, testImage, 1.0, 0, cv::NORM_MINMAX);
    return testImage;
}

template<typename Dtype>
void ImageUtil<Dtype>::showImage(const Dtype* data, int nthImage, int channel, int row,
    int col) {
    cv::Mat testImage = makeImage(data, nthImage, channel, row, col);
    
    cv::resize(testImage, testImage, cv::Size(), 2.0, 2.0);
    cv::imshow("test image", testImage);
    //cv::imwrite("/home/monhoney/yoyo.jpg", testImage);
    cv::waitKey(0);
}

template<typename Dtype>
void ImageUtil<Dtype>::saveImage(const Dtype* data, int imageCount, int channel, int row,
    int col, string folderName) {

    struct timeval val;
    struct tm* tmPtr;

    gettimeofday(&val, NULL);
    tmPtr = localtime(&val.tv_sec);

    string folderPath;
    if (strcmp(folderName.c_str(),  "") == 0) {
        char timeStr[1024];
        sprintf(timeStr, "%04d%02d%02d_%02d%02d%02d_%06ld",
            tmPtr->tm_year + 1900, tmPtr->tm_mon + 1, tmPtr->tm_mday, tmPtr->tm_hour,
            tmPtr->tm_min, tmPtr->tm_sec, val.tv_usec);

        folderPath = string(SPARAM(IMAGEUTIL_SAVE_DIR)) + "/" + string(timeStr);
    } else {
        folderPath = string(SPARAM(IMAGEUTIL_SAVE_DIR)) + "/" + folderName;
    }
    
    FileMgmt::checkDir(folderPath.c_str());

    for (int i = 0; i < imageCount; i++) {
        char imageName[1024];
        sprintf(imageName, "%d.jpg", i);

        string filePath = folderPath + "/" + string(imageName);
        cv::Mat newImage = makeImage(data, i, channel, row, col);
        cv::imwrite(filePath, newImage);
    }
}

template <typename Dtype>
void ImageUtil<Dtype>::dispDatum(const Datum* datum, const string& windowName) {
	cv::Mat cv_temp = DecodeDatumToCVMat(datum, true, true);
	dispCVMat(cv_temp, windowName);
}

template <typename Dtype>
void ImageUtil<Dtype>::dispCVMat(const cv::Mat& cv_img, const string& windowName) {
	cv::imshow(windowName, cv_img);
	cv::waitKey(0);
	cv::destroyWindow(windowName);
}

template <typename Dtype>
void ImageUtil<Dtype>::nms(std::vector<std::vector<float>>& proposals, 
        std::vector<float>& scores, const float thresh, std::vector<uint32_t>& keep) {

    vector<pair<int, float>> vec;

    for (int i = 0; i < scores.size(); i++) {
        vec.push_back(make_pair(i, scores[i]));
    }

    struct scoreCompareStruct {
        bool operator()(const pair<int, float> &left, const pair<int, float> &right) {
            return left.second > right.second;
        }
    };

    sort(vec.begin(), vec.end(), scoreCompareStruct());

    int maxScoreIndex = vec[0].first;

    bool live[vec.size()];
    for (int i = 0; i < vec.size(); i++)
        live[i] = true;

    for (int i = 0; i < vec.size() - 1; i++) {
        if (live[i] == false)
            continue;

        float x1 = proposals[vec[i].first][0];
        float y1 = proposals[vec[i].first][1];
        float x2 = proposals[vec[i].first][2];
        float y2 = proposals[vec[i].first][3];
        float area = (x2 - x1) * (y2 - y1);
        if (area == 0.0f) {
            live[i] = false;
            continue;
        }

        for (int j = i + 1; j < vec.size(); j++) {
            float tx1 = proposals[vec[j].first][0];
            float ty1 = proposals[vec[j].first][1];
            float tx2 = proposals[vec[j].first][2];
            float ty2 = proposals[vec[j].first][3];
            float tarea = (tx2 - tx1) * (ty2 - ty1);
            if (tarea == 0.0f) {
                live[j] = false;
                continue;
            }
            
            float xx1 = max(x1, tx1);
            float yy1 = max(y1, ty1);
            float xx2 = min(x2, tx2);
            float yy2 = min(y2, ty2);

            float w = max(xx2 - xx1, 0.0f);
            float h = max(yy2 - yy1, 0.0f);
            float inter = w * h;
            float iou = inter / (area + tarea - inter);

            if (iou > thresh)
                live[j] = false;
        }
    }

    for (int i = 0; i < vec.size(); i++) {
        if (live[i])
            keep.push_back(vec[i].first);
    }
}

template class ImageUtil<float>;
