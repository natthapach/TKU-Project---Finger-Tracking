#pragma once
#include "Activator.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <OpenNI.h>
#include <NiTE.h>

using namespace std;

class DepthActivator : virtual public Activator {
public:
	DepthActivator();
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

	void setEnableHandTracking(bool flag);
	void setEnableHandThreshold(bool flag);
	void setEnableDrawHandPoint(bool flag);
	void toggleEnableHandTracking();
	void toggleEnableHandThreshold();
	void toggleEnableDrawHandPoint();
	void toggleEnableSkinMask();
protected:
	const int RANGE = 100;
	const int DISTANCE_THRESHOLD = 10;
	const double fx_d = 5.9421434211923247e+02;
	const double fy_d = 5.9104053696870778e+02;
	const double cx_d = 3.3930780975300314e+02;
	const double cy_d = 2.4273913761751615e+02;
	
	uchar img[480][640][3];
	uchar mask[480][640];
	uint16_t depthRaw[480][640];
	cv::Mat imageFrame;
	cv::Mat maskFrame;

	vector<vector<cv::Point>> contours;
	vector<cv::Point> largestContour;
	vector<cv::Point> largestHull;
	
	openni::VideoStream videoStream;
	nite::HandTrackerFrameRef handsFrame;
	nite::HandTracker handTracker;
	int handDepth = 0;
	int numberOfHands = 0;
	float handPosX = 0;
	float handPosY = 0;
	int depthHistogram[65536];

	void calculate3DCoordinate(int px, int py, uint16_t depth, float* cx, float* cy, float* cz);
private:
	// Flag
	bool enableHandTracking = true;
	bool enableHandThreshold = false;
	bool enableDrawHandPoint = false;
	bool enableRGBSkinMask = true;
	bool markMode = true;

	void calDepthHistogram(openni::VideoFrameRef depthFrame, int* numberOfPoints, int* numberOfHandPoints);
	void modifyImage(openni::VideoFrameRef depthFrame, int numberOfPoints, int numberOfHandPoints);
	void settingHandValue();

	void floodFillHand(cv::Mat& in);
	void findHandContours(cv::Mat& in, vector<vector<cv::Point>>& contours, vector<cv::Point>& largestContour);
	void findHandConvexHull(vector<cv::Point>& largestContour, vector<cv::Point>& largestHull, vector<int>& largestHull_I, vector<cv::Vec4i>& defects);
	void findConvexPoint(vector<cv::Vec4i>& defects, vector<cv::Point>& convexPoint, vector<cv::Point>& largestContour);
	void clusterPoint(vector<cv::Point>& clusterInput, vector<cv::Point>& cluseterOutput);
};