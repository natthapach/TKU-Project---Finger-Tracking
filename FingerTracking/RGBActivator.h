#pragma once
#include "Activator.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <OpenNI.h>

class RGBActivator : virtual public Activator {
public:
	RGBActivator();
	void onInitial();
	void onPrepare();
	void onReadFrame();
	void onModifyFrame();
	void onMask(int signature, cv::Mat mask);
	void onDraw(int signature, cv::Mat canvas);
	void onPerformKeyboardEvent(int key);
	void onDie();

	cv::Mat getImageFrame();
	std::string getName();
	int getSignature();
protected:
	openni::Device device;
	openni::VideoStream sensor;
	uchar img[480][640][3];
	cv::Mat imageFrame;
	cv::Mat skinMask;
	void toggleIsShowMask();
private:
	bool isShowMask = true;
};
