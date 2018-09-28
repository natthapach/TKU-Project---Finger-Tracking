#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
class Combiner {
public:
	virtual void onCombine(map<int, cv::Mat> imageFrame) = 0;
	virtual cv::Mat getMaskFrame() = 0;
	virtual int getSignature() = 0;
	virtual string getName() = 0;
};