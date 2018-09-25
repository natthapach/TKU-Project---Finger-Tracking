#include "pch.h"
#include "DepthActivator.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <OpenNI.h>
#include <NiTE.h>
using namespace std;

DepthActivator::DepthActivator()
{
	name = "Depth Activator";
}

void DepthActivator::onInitial()
{
	nite::Status status = nite::STATUS_OK;

	status = nite::NiTE::initialize();
	if (status != nite::STATUS_OK)
		return;

	status = handTracker.create();
	if (status != nite::STATUS_OK)
		return;

	status = handTracker.startGestureDetection(nite::GESTURE_HAND_RAISE);
	if (status != nite::STATUS_OK)
		return;
}

void DepthActivator::onPrepare()
{
}

void DepthActivator::onReadFrame()
{
	if (!handTracker.isValid())
		return;

	nite::Status status = nite::STATUS_OK;
	//nite::HandTrackerFrameRef handsFrame;

	if (enableHandTracking) {
		status = handTracker.readFrame(&handsFrame);
		if (status != nite::STATUS_OK || !handsFrame.isValid())
			return;
	}
	

	const nite::Array<nite::GestureData>& gestures = handsFrame.getGestures();
	for (int i = 0; i < gestures.getSize(); ++i) {
		if (gestures[i].isComplete()) {
			nite::HandId handId;
			handTracker.startHandTracking(gestures[i].getCurrentPosition(), &handId);
		}
	}

	openni::VideoFrameRef depthFrame = handsFrame.getDepthFrame();

	int numberOfPoints = 0;
	int numberOfHandPoints = 0;
	calDepthHistogram(depthFrame, &numberOfPoints, &numberOfHandPoints);
	modifyImage(depthFrame, numberOfPoints, numberOfHandPoints);
	
	settingHandValue();
}

void DepthActivator::onModifyFrame()
{
	imageFrame = cv::Mat(480, 640, CV_8UC3, &img);
	if (numberOfHands <= 0)
		return;

	cv::Mat gray;
	cv::cvtColor(imageFrame, gray, cv::COLOR_BGR2GRAY);

	// make sure Seed Point (hand point) belong to hand region
	int smallKernel = 3;
	for (int y = handPosY-smallKernel; y < handPosY+smallKernel; y++) {
		for (int x = handPosX-smallKernel; x < handPosX+smallKernel; x++) {
			gray.at<uchar>(y, x) = 128;
		}
	}
	cv::floodFill(gray, cv::Point((int)handPosX, (int)handPosY), cv::Scalar(255));
	cv::threshold(gray, gray, 129, 255, cv::THRESH_BINARY);
	
	// find contours
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	cv::findContours(gray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	// find largest contour
	double maxArea = 0;
	int maxIndex = 0;
	vector<cv::Point> largestContour;
	for (int i = 0; i < contours.size(); i++) {
		double a = cv::contourArea(contours[i], false);
		if (a > maxArea) {
			maxArea = a;
			maxIndex = i;
		}
	}
	largestContour = contours[maxIndex];

	// find convex hull of largest contour
	vector<vector<cv::Point>> largestHull(1);
	vector<vector<int>> largestHullI(1);
	cv::convexHull(cv::Mat(largestContour), largestHull[0]);
	cv::convexHull(cv::Mat(largestContour), largestHullI[0]);

	vector<vector<cv::Vec4i>> defects(1);
	cv::convexityDefects(largestContour, largestHullI[0], defects[0]);

	cv::Mat drawing = cv::Mat::zeros(gray.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++) {
		cv::drawContours(drawing, contours, i, cv::Scalar(0, 255, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	}
	cv::drawContours(drawing, largestHull, 0, cv::Scalar(0, 255, 255), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	for (int j = 0; j < largestHull[0].size(); j++) {
		//cv::circle(drawing, largestHull[0][j], 2, cv::Scalar(0, 0, 255), -1, 8);
	}
	cout << "finded " << largestHull[0].size() << "hull point" << endl;

	vector<vector<double>> distMat(largestHull[0].size());
	for (int i = 0; i < distMat.size(); i++) {
		cv::Point pi = largestHull[0][i];
		distMat[i] = vector<double>(largestHull[0].size());

		for (int j = i+1; j < distMat[i].size(); j++) {
			cv::Point pj = largestHull[0][j];
			distMat[i][j] = sqrt(pow(pi.x - pj.x, 2) + pow(pi.y - pj.y, 2));
		}
	}
	double thresholdDist = 10;
	vector<int> cluster(distMat.size(), 0);
	int clusterCount = 1;
	for (int i = 0; i < distMat.size(); i++) {
		for (int j = i + 1; j < distMat[i].size(); j++) {
			if (distMat[i][j] < thresholdDist) {
				if (cluster[i] == 0 && cluster[j] == 0) {
					cluster[i] = clusterCount;
					cluster[j] = clusterCount;
					clusterCount++;
				}
				else if (cluster[i] == 0) {
					cluster[i] = cluster[j];
				}
				else {
					cluster[j] = cluster[i];
				}
			}
		}
	}
	for (int i = 0; i < cluster.size(); i++) {
		if (cluster[i] == 0) {
			cluster[i] = clusterCount;
			clusterCount++;
		}
	}
	int m = cluster.size();
	for (int i = 1; i < m - 1; i++) {
		double x_sum = 0;
		double y_sum = 0;
		int x_count = 0;
		int y_count = 0;
		for (int j = 0; j < largestHull[0].size(); j++) {
			if (cluster[j] == i) {
				x_sum += largestHull[0][j].x;
				y_sum += largestHull[0][j].y;
				y_count++;
				x_count++;
			}
		}
		cv::circle(drawing, cv::Point((int)(x_sum/x_count), (int)(y_sum/y_count)), 4, cv::Scalar(255, 255, 255), 2, 8);
	}
	//if (clusterCount > 0) {
		
	//}

	//for (int i = 0; i < defects[0].size(); i++) {
	//	const cv::Vec4i& v = defects[0][i];
	//	float depth = v[3];		// depth value
	//	if (depth < 500) {
	//		int startIndex = v[0];
	//		int endIndex = v[1];
	//		int farIndex = v[2];
	//		cv::Point ptStart(largestContour[startIndex]);
	//		cv::Point ptEnd(largestContour[endIndex]);
	//		cv::Point ptFar(largestContour[farIndex]);

	//		cv::line(drawing, ptStart, ptEnd, cv::Scalar(255, 0, 0), 1);
	//		cv::line(drawing, ptStart, ptFar, cv::Scalar(255, 0, 0), 1);
	//		cv::line(drawing, ptEnd, ptFar, cv::Scalar(255, 0, 0), 1);
	//		cv::circle(drawing, ptFar, 4, cv::Scalar(255, 0, 0), 2);
	//		cout << "convex depth " << depth << endl;
	//	}
	//}
	imageFrame = drawing;
}

void DepthActivator::onDraw(string name, cv::Mat canvas)
{
	//cv::rectangle(canvas, cv::Rect(55, 60, 10, 10), cv::Scalar(0, 0, 255), -1, cv::LINE_8);
	//cv::rectangle(canvas, cv::Rect(600, 470, 10, 10), cv::Scalar(0, 0, 255), -1, cv::LINE_8);
	//cv::rectangle(canvas, cv::Rect(55, 60, 565, 410), cv::Scalar(0, 255, 255), 5, cv::LINE_8);

	if (!enableDrawHandPoint)
		return;

	const nite::Array<nite::HandData>& hands = handsFrame.getHands();

	for (int i = 0; i < hands.getSize(); i++) {
		nite::HandData hand = hands[i];

		if (hand.isTracking()) {
			float posX, posY;
			handTracker.convertHandCoordinatesToDepth(
				hand.getPosition().x,
				hand.getPosition().y,
				hand.getPosition().z,
				&posX, &posY
			);
			char buffer[50];
			sprintf_s(buffer, "(%.2f,%.2f,%.2f)",
				hand.getPosition().x,
				hand.getPosition().y,
				hand.getPosition().z);
			printf("Hand Detect #%d (%.2f, %.2f, %.2f)\n", hand.getId(), hand.getPosition().x,
				hand.getPosition().y,
				hand.getPosition().z);
			cv::circle(canvas, cv::Point((int)posX, (int)posY), 5, cv::Scalar(255, 0, 0), -1, cv::LINE_8);
			cv::putText(canvas, buffer, cv::Point((int)posX, (int)posY + 5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
		}
	}
}

void DepthActivator::onPerformKeyboardEvent(int key)
{
	if (key == 't' || key == 'T') {
		toggleEnableHandThreshold();
	}
	else if (key == 'k' || key == 'K') {
		toggleEnableDrawHandPoint();
		std::cout << "K " << enableDrawHandPoint << std::endl;
	}
}

void DepthActivator::onDie()
{
}

cv::Mat DepthActivator::getImageFrame()
{
	return imageFrame;
}

std::string DepthActivator::getName()
{
	return "Depth Activator";
}

void DepthActivator::setEnableHandTracking(bool flag)
{
	enableHandTracking = flag;
}

void DepthActivator::setEnableHandThreshold(bool flag)
{
	enableHandThreshold = flag;
}

void DepthActivator::setEnableDrawHandPoint(bool flag)
{
	enableDrawHandPoint = flag;
}

void DepthActivator::toggleEnableHandTracking()
{
	enableHandTracking = !enableHandTracking;
}

void DepthActivator::toggleEnableHandThreshold()
{
	enableHandThreshold = !enableHandThreshold;
}

void DepthActivator::toggleEnableDrawHandPoint()
{
	enableDrawHandPoint = !enableDrawHandPoint;
}

void DepthActivator::calDepthHistogram(openni::VideoFrameRef depthFrame, int * numberOfPoints, int * numberOfHandPoints)
{
	*numberOfPoints = 0;
	*numberOfHandPoints = 0;

	memset(depthHistogram, 0, sizeof(depthHistogram));
	for (int y = 0; y < depthFrame.getHeight(); ++y)
	{
		openni::DepthPixel* depthCell = (openni::DepthPixel*)
			(
			(char*)depthFrame.getData() +
				(y * depthFrame.getStrideInBytes())
				);
		for (int x = 0; x < depthFrame.getWidth(); ++x, ++depthCell)
		{
			if (*depthCell != 0)
			{
				depthHistogram[*depthCell]++;
				(*numberOfPoints)++;

				if (handDepth > 0 && numberOfHands > 0) {
					if (handDepth - RANGE <= *depthCell && *depthCell <= handDepth + RANGE)
						(*numberOfHandPoints)++;
				}
			}
		}
	}
	for (int nIndex = 1; nIndex < sizeof(depthHistogram) / sizeof(int); nIndex++)
	{
		depthHistogram[nIndex] += depthHistogram[nIndex - 1];
	}
}

void DepthActivator::modifyImage(openni::VideoFrameRef depthFrame, int numberOfPoints, int numberOfHandPoints)
{
	for (unsigned int y = 0; y < 480; y++) {
		for (unsigned int x = 0; x < 640; x++) {
			openni::DepthPixel* depthPixel = (openni::DepthPixel*)
				((char*)depthFrame.getData() + (y*depthFrame.getStrideInBytes())) + x;

			if (handDepth != 0 && numberOfHands > 0 && enableHandThreshold) {
				if (depthPixel != 0 && (handDepth-RANGE <= *depthPixel && *depthPixel <= handDepth + RANGE)) {
					if (markMode) {
						img[y][x][0] = 128;
						img[y][x][1] = 128;
						img[y][x][2] = 128;
					}
					else {
						uchar depthValue = (uchar)(((float)depthHistogram[*depthPixel] / numberOfHandPoints) * 255);
						img[y][x][0] = 255 - depthValue;
						img[y][x][1] = 255 - depthValue;
						img[y][x][2] = 255 - depthValue;
					}
				}
				else {
					if (markMode) {
						img[y][x][0] = 0;
						img[y][x][1] = 0;
						img[y][x][2] = 0;
					}
					else {
						img[y][x][0] = 0;
						img[y][x][1] = 255;
						img[y][x][2] = 0;
					}
				}
			}
			else {
				if (*depthPixel != 0) {
					uchar depthValue = (uchar)(((float)depthHistogram[*depthPixel] / numberOfPoints) * 255);
					img[y][x][0] = 255 - depthValue;
					img[y][x][1] = 255 - depthValue;
					img[y][x][2] = 255 - depthValue;
				}
				else {
					img[y][x][0] = 0;
					img[y][x][1] = 0;
					img[y][x][2] = 0;
				}
			}
		}
	}
}

void DepthActivator::settingHandValue()
{
	const nite::Array<nite::HandData>& hands = handsFrame.getHands();

	for (int i = 0; i < hands.getSize(); i++) {
		nite::HandData hand = hands[i];

		if (hand.isTracking()) {
			nite::Point3f position = hand.getPosition();
			float x, y;
			handTracker.convertHandCoordinatesToDepth(
				hand.getPosition().x,
				hand.getPosition().y,
				hand.getPosition().z,
				&x, &y
			);
			handPosX = x;
			handPosY = y;
			openni::VideoFrameRef depthFrame = handsFrame.getDepthFrame();
			openni::DepthPixel* depthPixel = (openni::DepthPixel*) ((char*)depthFrame.getData() + ((int)y * depthFrame.getStrideInBytes())) + (int)x;
			handDepth = *depthPixel;
		}

		if (hand.isLost())
			numberOfHands--;
		if (hand.isNew())
			numberOfHands++;
	}
}


