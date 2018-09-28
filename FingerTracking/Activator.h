#pragma once
#include <opencv2/opencv.hpp>

class Activator {
public:
	virtual void onInitial() = 0;
	virtual void onPrepare() = 0;		// = 0 mean pure virtual function
	virtual void onReadFrame() = 0;
	virtual void onModifyFrame() = 0;
	virtual void onDraw(std::string name, cv::Mat canvas) = 0;
	virtual void onPerformKeyboardEvent(int key) = 0;
	virtual void onDie() = 0;
	virtual cv::Mat getImageFrame() = 0;
	virtual std::string getName() = 0;
	virtual int getSignature() = 0;
	std::string name;
};