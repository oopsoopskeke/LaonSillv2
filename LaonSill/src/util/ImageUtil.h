/**
 * @file ImageUtil.h
 * @date 2017-02-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef IMAGEUTIL_H
#define IMAGEUTIL_H 

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "common.h"
#include "Datum.h"


template<typename Dtype>
class ImageUtil {
public: 
    ImageUtil() {}
    virtual ~ImageUtil() {}

    static void showImage(const Dtype* data, int nthImage, int channel, int row, int col);
    static void saveImage(const Dtype* data, int imageCount, int channel, int row, int col,
        std::string folderName);


    static void dispDatum(const Datum* datum, const std::string& windowName);
    static void dispCVMat(const cv::Mat& cv_img, const std::string& windowName);

    static void nms(std::vector<std::vector<float>>& proposals, std::vector<float>& scores,
		const float thresh, std::vector<uint32_t>& keep);
};

#endif /* IMAGEUTIL_H */
