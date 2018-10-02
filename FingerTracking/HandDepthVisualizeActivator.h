#pragma once
#include "Activator.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <OpenNI.h>
#include <NiTE.h>

using namespace std;

class HandDepthVisualizeActivator : virtual public Activator {
public:
	HandDepthVisualizeActivator();
	void onInitial();
	void onPrepare();
	void onReadFrame();
	void onModifyFrame();
	void onMask(std::map<int, cv::Mat> masks);
	void onDraw(int signature, cv::Mat canvas);
	void onPerformKeyboardEvent(int key);
	void onDie();

	cv::Mat getImageFrame();
	std::string getName();
	int getSignature();
protected:
	const int RANGE = 100;
	const int DISTANCE_THRESHOLD = 10;
	const int IGNORE_LAYER1_THRESHOLD = 10;

	nite::HandTracker handTracker;
	openni::VideoStream videoStream;
	nite::HandTrackerFrameRef handsFrame;

	int handDepth = 0;
	int numberOfHands = 0;
	float handPosX = 0;
	float handPosY = 0;

	uchar img[480][640][3];
	uchar maskImg[480][640];
	uchar m1[480][640];
	uchar m2[480][640];
	uchar m3[480][640];
	uint16_t depthRaw[480][640];
	int depthHistogram[65536];

	cv::Mat imageFrame;
	cv::Mat maskFrame;
	cv::Mat maskL1;
	cv::Mat maskL2;
	cv::Mat maskL3;
	cv::Mat mask;
private:
	// Flag
	bool enableHandTracking = true;
	bool enableHandThreshold = false;
	bool enableDrawHandPoint = false;
	bool markMode = true;

	void calDepthHistogram(openni::VideoFrameRef depthFrame, int* numberOfPoints, int* numberOfHandPoints);
	void modifyImage(openni::VideoFrameRef depthFrame, int numberOfPoints, int numberOfHandPoints);
	void settingHandValue();

	void floodFillHand(cv::Mat& in);
};