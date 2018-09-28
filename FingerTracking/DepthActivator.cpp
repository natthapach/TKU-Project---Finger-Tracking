#include "pch.h"
#include "DepthActivator.h"
#include "Constants.h"

#include <stdio.h>
#include <math.h>
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

	openni::Status statusNi = openni::STATUS_OK;

	statusNi = openni::OpenNI::initialize();
	if (statusNi != openni::STATUS_OK)
		return;

	openni::Device device;
	statusNi = device.open(openni::ANY_DEVICE);
	if (statusNi != openni::STATUS_OK)
		return;

	statusNi = videoStream.create(device, openni::SENSOR_DEPTH);
	if (statusNi != openni::STATUS_OK)
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
	
	imageFrame = cv::Mat(480, 640, CV_8UC3, &img);
	maskFrame = cv::Mat(480, 640, CV_8UC1, &mask);
	settingHandValue();
}

void DepthActivator::onModifyFrame()
{
	//imageFrame = cv::Mat(480, 640, CV_8UC3, &img);
	if (numberOfHands <= 0)
		return;

	/*cv::Mat gray;
	cv::cvtColor(imageFrame, gray, cv::COLOR_BGR2GRAY);*/

	cv::Mat gray = maskFrame;
	// make sure Seed Point (hand point) belong to hand region

	int smallKernel = 3;
	for (int y = handPosY-smallKernel; y < handPosY+smallKernel; y++) {
		for (int x = handPosX-smallKernel; x < handPosX+smallKernel; x++) {
			gray.at<uchar>(y, x) = 128;
		}
	}
	cv::floodFill(gray, cv::Point((int)handPosX, (int)handPosY), cv::Scalar(255));
	cv::threshold(gray, gray, 129, 255, cv::THRESH_BINARY);

	imageFrame = gray;
	return;
	
	// find contours
	vector<cv::Vec4i> hierarchy;
	cv::findContours(gray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	// find largest contour
	double maxArea = 0;
	int maxIndex = 0;
	for (int i = 0; i < contours.size(); i++) {
		double a = cv::contourArea(contours[i], false);
		if (a > maxArea) {
			maxArea = a;
			maxIndex = i;
		}
	}
	largestContour = contours[maxIndex];

	// find convex hull of largest contour
	cv::convexHull(cv::Mat(largestContour), largestHull);
	
	// find convexity defect
	vector<int> largestHull_I;
	cv::convexHull(cv::Mat(largestContour), largestHull_I);
	vector<cv::Vec4i> defects;
	cv::convexityDefects(largestContour, largestHull_I, defects);

	cv::Mat drawing = cv::Mat::zeros(gray.size(), CV_8UC3);

	vector<cv::Point> convexPoint;

	for (int i = 0; i < defects.size(); i++) {
		cv::Vec4i& v = defects[i];
		float depth = v[3];
		if (depth > 10) {
			int startIndex = v[0];
			int endIndex = v[1];
			int farIndex = v[2];

			cv::Point startPoint(largestContour[startIndex]);
			cv::Point endPoint(largestContour[endIndex]);
			cv::Point farPoint(largestContour[farIndex]);

			double ms = ((double)(startPoint.y - farPoint.y)) / (startPoint.x - farPoint.x);
			double me = ((double)(endPoint.y - farPoint.y)) / (endPoint.x - farPoint.x);
			double angle = atan((me - ms) / (1 + (ms * me))) * (180/3.14159265);

			cv::line(drawing, startPoint, endPoint, cv::Scalar(255, 255, 255), 1);
			cv::line(drawing, startPoint, farPoint, cv::Scalar(255, 255, 255), 1);
			cv::line(drawing, endPoint, farPoint, cv::Scalar(255, 255, 255), 1);

			if (angle < 0) {
				cv::circle(drawing, farPoint, 4, cv::Scalar(0, 0, 255), 2);
				//cv::circle(drawing, startPoint, 4, cv::Scalar(0, 255, 0), 2);
				//cv::circle(drawing, endPoint, 4, cv::Scalar(0, 255, 0), 2);
				convexPoint.push_back(startPoint);
				convexPoint.push_back(endPoint);
			}
			/*else 
				cv::circle(drawing, farPoint, 4, cv::Scalar(255, 255, 255), 2);*/
		}
	}


	
	cv::drawContours(drawing, vector<vector<cv::Point>>{ largestContour }, 0, cv::Scalar(0, 255, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());

	vector<cv::Point> clusterInput = convexPoint;
	vector<vector<double>> distMat(clusterInput.size());
	for (int i = 0; i < distMat.size(); i++) {
		cv::Point pi = clusterInput[i];
		distMat[i] = vector<double>(clusterInput.size());

		for (int j = i+1; j < distMat[i].size(); j++) {
			cv::Point pj = clusterInput[j];
			distMat[i][j] = sqrt(pow(pi.x - pj.x, 2) + pow(pi.y - pj.y, 2));
		}
	}
	vector<int> cluster(distMat.size(), 0);
	int clusterCount = 1;
	for (int i = 0; i < distMat.size(); i++) {
		for (int j = i + 1; j < distMat[i].size(); j++) {
			if (distMat[i][j] < DISTANCE_THRESHOLD) {
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
		for (int j = 0; j < clusterInput.size(); j++) {
			if (cluster[j] == i) {
				x_sum += clusterInput[j].x;
				y_sum += clusterInput[j].y;
				y_count++;
				x_count++;
			}
		}
		if (x_count == 0) {
			continue;
		}
		int x = (int)(x_sum / x_count);
		int y = (int)(y_sum / y_count);
		uint16_t depth = depthRaw[y][x];
		
		cv::circle(drawing, cv::Point(x, y), 4, cv::Scalar(255, 255, 255), -1, 8);
		
		float cx, cy, cz;
		calculate3DCoordinate(x, y, depth, &cx, &cy, &cz);
		char buffer[50];
		sprintf_s(buffer, "(%.2f,%.2f,%.2f)", cx, cy, cz);
		cv::putText(drawing, buffer, cv::Point(x, y + 5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

	}
	imageFrame = drawing;
}

void DepthActivator::onMask(int signature, cv::Mat mask)
{
}

void DepthActivator::onDraw(int signature, cv::Mat canvas)
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
	return WINDOW_NAME_ACTIVATOR_DEPTH;
}

int DepthActivator::getSignature()
{
	return ACTIVATOR_DEPTH;
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

void DepthActivator::calculate3DCoordinate(int px, int py, uint16_t depth, float * cx, float* cy, float* cz)
{
	/*(*cx) = (px - cx_d) * depth / fx_d;
	(*cy) = (py - cy_d) * depth / fy_d;
	(*cz) = depth;*/
	/*float bx, by;
	handTracker.convertDepthCoordinatesToHand(px, py, (int)depth, &bx, &by);
	*cx = bx;
	*cy = by;
	*cz = depth;*/
	openni::CoordinateConverter::convertDepthToWorld(videoStream, (float) px,(float) py, (float) depth, cx, cy, cz);

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
			depthRaw[y][x] = *depthPixel;
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

			if (handDepth != 0 && numberOfHands > 0) {
				if (*depthPixel != 0 && (handDepth - RANGE <= *depthPixel && *depthPixel <= handDepth + RANGE)) {
					mask[y][x] = 128;
				}
				else {
					mask[y][x] = 0;
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

			float calX;
			float calY;
			double calZ;
			handTracker.convertDepthCoordinatesToHand(x, y, (int)*depthPixel, &calX, &calY);
			//calculate3DCoordinate(x, y, *depthPixel, &calX, &calY, &calZ);
			//cout << "hand #" << hand.getId() << " NITE : (" << hand.getPosition().x << ", " << hand.getPosition().y << ", " << hand.getPosition().z << ")" << endl;
			//cout << "hand #" << hand.getId() << " CAL  : (" << calX << ", " << calY << ", " << calZ << ")" << endl;
			int a = 1;
		}

		if (hand.isLost())
			numberOfHands--;
		if (hand.isNew())
			numberOfHands++;
	}
}


