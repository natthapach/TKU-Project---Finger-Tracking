#pragma once
#include "Activator.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <OpenNI.h>
#include <NiTE.h>

using namespace std;

class RawDepthActivator : virtual public Activator {
public:
	RawDepthActivator();
	void onInitial();
	void onPrepare();
	void onReadFrame();
	void onModifyFrame();
	void onDraw(std::string name, cv::Mat canvas);
	void onPerformKeyboardEvent(int key);
	void onDie();

	cv::Mat getImageFrame();
	std::string getName();
	int getSignature();
protected:
	openni::Device device;
	openni::VideoStream sensor;
	openni::VideoStream rgbSensor;
	uchar img[480][640][3];
	uint16_t depthRaw[480][640];
	int depthHistogram[65536];

	cv::Mat imageFrame;
private:
	// Flag
	bool enableHandTracking = true;
	bool enableHandThreshold = false;
	bool enableDrawHandPoint = false;
	bool markMode = true;

	void calDepthHistogram(openni::VideoFrameRef depthFrame, int* numberOfPoints, int* numberOfHandPoints);
	void modifyImage(openni::VideoFrameRef depthFrame, openni::VideoFrameRef rgbFrame, int numberOfPoints, int numberOfHandPoints);
};