#include "pch.h"
#include "DepthActivator.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <OpenNI.h>
#include <NiTE.h>

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
}

void DepthActivator::onDraw(cv::Mat canvas)
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
	return cv::Mat(480, 640, CV_8UC3, &img);
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
					uchar depthValue = (uchar)(((float)depthHistogram[*depthPixel] / numberOfHandPoints) * 255);
					img[y][x][0] = 255 - depthValue;
					img[y][x][1] = 255 - depthValue;
					img[y][x][2] = 255 - depthValue;
				}
				else {
					img[y][x][0] = 0;
					img[y][x][1] = 255;
					img[y][x][2] = 0;
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


