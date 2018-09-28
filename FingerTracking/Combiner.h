#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
class Combiner {
public:
	virtual void onCombine(map<string, cv::Mat> imageFrame, map<string, int> signatures) = 0;
	virtual cv::Mat getImageFrame() = 0;
	virtual int getSignature() = 0;
};