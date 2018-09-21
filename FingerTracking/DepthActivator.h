#pragma once
#include "Activator.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <OpenNI.h>
#include <NiTE.h>

class DepthActivator : virtual public Activator {
public:
	DepthActivator();
	void onInitial();
	void onPrepare();
	void onReadFrame();
	void onModifyFrame();
	void onDraw(std::string name, cv::Mat canvas);
	void onPerformKeyboardEvent(int key);
	void onDie();
	cv::Mat getImageFrame();
	std::string getName();
	void setEnableHandTracking(bool flag);
	void setEnableHandThreshold(bool flag);
	void setEnableDrawHandPoint(bool flag);
	void toggleEnableHandTracking();
	void toggleEnableHandThreshold();
	void toggleEnableDrawHandPoint();
protected:
	uchar img[480][640][3];
	nite::HandTracker handTracker;
	int handDepth = 0;
	int numberOfHands = 0;
	int depthHistogram[65536];
	const int RANGE = 50;
	nite::HandTrackerFrameRef handsFrame;
private:
	
	bool enableHandTracking = true;
	bool enableHandThreshold = false;
	bool enableDrawHandPoint = false;
	void calDepthHistogram(openni::VideoFrameRef depthFrame, int* numberOfPoints, int* numberOfHandPoints);
	void modifyImage(openni::VideoFrameRef depthFrame, int numberOfPoints, int numberOfHandPoints);
	void settingHandValue();
};