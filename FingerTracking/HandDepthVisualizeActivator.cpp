#include "pch.h"
#include "HandDepthVisualizeActivator.h"
#include "Constants.h"

using namespace std;

HandDepthVisualizeActivator::HandDepthVisualizeActivator()
{
}

void HandDepthVisualizeActivator::onInitial()
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

void HandDepthVisualizeActivator::onPrepare()
{
}

void HandDepthVisualizeActivator::onReadFrame()
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
	maskFrame = cv::Mat(480, 640, CV_8UC1, &maskImg);
	maskL1 = cv::Mat(480, 640, CV_8UC1, &m1);
	maskL2 = cv::Mat(480, 640, CV_8UC1, &m2);
	maskL3 = cv::Mat(480, 640, CV_8UC1, &m3);
	cv::Mat ms[] = { maskL1, maskL2, maskL3 };
	settingHandValue();
}

void HandDepthVisualizeActivator::onModifyFrame()
{
	if (numberOfHands <= 0)
		return;

	cv::Mat drawing = cv::Mat::zeros(imageFrame.size(), CV_8UC3);

	floodFillHand(maskFrame);
	cv::bitwise_and(maskL1, maskFrame, maskL1);
	cv::bitwise_and(maskL2, maskFrame, maskL2);
	cv::bitwise_and(maskL3, maskFrame, maskL3);

	vector<cv::Vec4i> hierarchy1;
	vector<cv::Vec4i> hierarchy2;
	vector<cv::Vec4i> hierarchy3;
	vector<vector<cv::Point>> contours1;
	vector<vector<cv::Point>> contours2;
	vector<vector<cv::Point>> contours3;
	cv::findContours(maskL1, contours1, hierarchy1, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::findContours(maskL2, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::findContours(maskL3, contours3, hierarchy3, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	cv::cvtColor(maskL1, maskL1, cv::COLOR_GRAY2BGR);
	cv::cvtColor(maskL2, maskL2, cv::COLOR_GRAY2BGR);
	cv::cvtColor(maskL3, maskL3, cv::COLOR_GRAY2BGR);

	for (int i = 0; i < contours1.size(); i++) {
		cv::drawContours(maskL1, contours1, i, cv::Scalar(0, 0, 255), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
		cv::drawContours(maskL2, contours1, i, cv::Scalar(0, 0, 255), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
		cv::drawContours(maskL3, contours1, i, cv::Scalar(0, 0, 255), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
		cv::drawContours(drawing, contours1, i, cv::Scalar(0, 0, 255), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	}
	for (int i = 0; i < contours2.size(); i++) {
		cv::drawContours(maskL2, contours2, i, cv::Scalar(0, 255, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
		cv::drawContours(maskL3, contours2, i, cv::Scalar(0, 255, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
		cv::drawContours(drawing, contours2, i, cv::Scalar(0, 255, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	}
	for (int i = 0; i < contours3.size(); i++) {
		cv::drawContours(maskL3, contours3, i, cv::Scalar(255, 0, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
		cv::drawContours(drawing, contours3, i, cv::Scalar(255, 0, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	}

	cv::imshow("mask1", maskL1);
	cv::imshow("mask2", maskL2);
	cv::imshow("mask3", maskL3);
	cv::imshow("drawing", drawing);
}

void HandDepthVisualizeActivator::onMask(std::map<int, cv::Mat> masks)
{
}

void HandDepthVisualizeActivator::onDraw(int signature, cv::Mat canvas)
{
}

void HandDepthVisualizeActivator::onPerformKeyboardEvent(int key)
{
}

void HandDepthVisualizeActivator::onDie()
{
}

cv::Mat HandDepthVisualizeActivator::getImageFrame()
{
	if (numberOfHands > 0)
		return maskFrame;
	return imageFrame;
}

std::string HandDepthVisualizeActivator::getName()
{
	return WINDOW_NAME_ACTIVATOR_HAND_DEPTH_VISUALIZE;
}

int HandDepthVisualizeActivator::getSignature()
{
	return ACTIVATOR_HAND_DEPTH_VISUALIZE;
}

void HandDepthVisualizeActivator::calDepthHistogram(openni::VideoFrameRef depthFrame, int * numberOfPoints, int * numberOfHandPoints)
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

void HandDepthVisualizeActivator::modifyImage(openni::VideoFrameRef depthFrame, int numberOfPoints, int numberOfHandPoints)
{
	for (unsigned int y = 0; y < 480; y++) {
		for (unsigned int x = 0; x < 640; x++) {
			openni::DepthPixel* depthPixel = (openni::DepthPixel*) ((char*)depthFrame.getData() + (y*depthFrame.getStrideInBytes())) + x;
			depthRaw[y][x] = *depthPixel;

			if (numberOfHands > 0) {
				if (*depthPixel != 0) {
					uchar depthValue = (uchar)(((float)depthHistogram[*depthPixel] / numberOfHandPoints) * 255);
					if (handDepth - RANGE <= *depthPixel && *depthPixel <= handDepth + RANGE) {
						/*img[y][x][0] = 255 - depthValue;
						img[y][x][1] = 255 - depthValue;
						img[y][x][2] = 255 - depthValue;*/
						maskImg[y][x] = 128;

						if (handDepth - RANGE <= *depthPixel && *depthPixel < handDepth - 20) {
							img[y][x][0] = 0;
							img[y][x][1] = 0;
							img[y][x][2] = 255 - depthValue;

							m1[y][x] = 255;
							m2[y][x] = 0;
							m3[y][x] = 0;
						}
						else if (handDepth - 20 <= *depthPixel && *depthPixel < handDepth + 30) {
							img[y][x][0] = 0;
							img[y][x][1] = 255 - depthValue;
							img[y][x][2] = 0;

							m1[y][x] = 0;
							m2[y][x] = 255;
							m3[y][x] = 0;
						}
						else {
							img[y][x][0] = 255 - depthValue;
							img[y][x][1] = 0;
							img[y][x][2] = 0;

							m1[y][x] = 0;
							m2[y][x] = 0;
							m3[y][x] = 255;
						}
					}
					else {
						img[y][x][0] = 0;
						img[y][x][1] = 0;
						img[y][x][2] = 0;

						maskImg[y][x] = 0;

						m1[y][x] = 0;
						m2[y][x] = 0;
						m3[y][x] = 0;
					}
				}
				else {
					img[y][x][0] = 0;
					img[y][x][1] = 0;
					img[y][x][2] = 0;

					maskImg[y][x] = 0;

					m1[y][x] = 0;
					m2[y][x] = 0;
					m3[y][x] = 0;
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

void HandDepthVisualizeActivator::settingHandValue()
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

void HandDepthVisualizeActivator::floodFillHand(cv::Mat & in)
{
	int smallKernel = 3;
	for (int y = handPosY - smallKernel; y < handPosY + smallKernel; y++) {
		for (int x = handPosX - smallKernel; x < handPosX + smallKernel; x++) {
			in.at<uchar>(y, x) = 128;
		}
	}
	cv::floodFill(in, cv::Point((int)handPosX, (int)handPosY), cv::Scalar(255));
	cv::threshold(in, in, 129, 255, cv::THRESH_BINARY);
}
