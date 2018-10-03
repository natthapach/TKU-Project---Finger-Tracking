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

void HandDepthVisualizeActivator::onPrepare()
{
	cv::namedWindow("mask1", cv::WINDOW_NORMAL);
	cv::namedWindow("mask2", cv::WINDOW_NORMAL);
	cv::namedWindow("mask3", cv::WINDOW_NORMAL);
	cv::namedWindow("drawing", cv::WINDOW_NORMAL);
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

	this->depthFrame = cv::Mat(480, 640, CV_8UC3, &img);
	maskFrame = cv::Mat(480, 640, CV_8UC1, &maskImg);
	maskL1 = cv::Mat(480, 640, CV_8UC1, &m1);
	maskL2 = cv::Mat(480, 640, CV_8UC1, &m2);
	maskL3 = cv::Mat(480, 640, CV_8UC1, &m3);
	cv::Mat ms[] = { maskL1, maskL2, maskL3 };
	settingHandValue();
}

void HandDepthVisualizeActivator::onModifyFrame()
{
	if (numberOfHands <= 0) {
		imageFrame = depthFrame;
	}
	else {
		floodFillHand(maskFrame);
		imageFrame = maskFrame;
	}
}

void HandDepthVisualizeActivator::onMask(std::map<int, cv::Mat> masks)
{
	if (numberOfHands <= 0)
		return;

	cv::Mat drawing = cv::Mat::zeros(imageFrame.size(), CV_8UC3);
	cv::Mat maskL1Corner;
	/*floodFillHand(maskFrame);*/

	if (masks.count(COMBINER_RGB_DEPTH) > 0) {
		cv::bitwise_and(maskFrame, masks[COMBINER_RGB_DEPTH], maskFrame);
	}
	cv::cvtColor(maskFrame, drawing, cv::COLOR_GRAY2BGR);
	cv::bitwise_and(maskL1, maskFrame, maskL1);
	cv::bitwise_and(maskL2, maskFrame, maskL2);
	cv::bitwise_and(maskL3, maskFrame, maskL3);
	cv::Mat maskL1x;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::dilate(maskL1, maskL1, kernel);
	cv::GaussianBlur(maskL1, maskL1x, cv::Size(3, 3), 1);
	//cv::addWeighted(maskL1, 2.0, maskL1x, -0.5, 0, maskL1x);
	cv::cornerHarris(maskL1, maskL1Corner, 10, 5, 0.04, cv::BORDER_DEFAULT);
	cv::normalize(maskL1Corner, maskL1Corner, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());


	map<int, vector<cv::Point>> mapCornerContourL1;	// { contoursIndex: [cornerPoint, ...] }
	vector<cv::Point> fingerL1;
	vector<cv::Point> fingerL2, fingerL2Cluster;
	vector<cv::Vec4i> hierarchy1;
	vector<cv::Vec4i> hierarchy2;
	vector<cv::Vec4i> hierarchy3;
	vector<vector<cv::Point>> contours1;
	vector<vector<cv::Point>> contours2;
	vector<vector<cv::Point>> contours3;
	cv::findContours(maskL1, contours1, hierarchy1, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::findContours(maskL2, contours2, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::findContours(maskL3, contours3, hierarchy3, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	vector<vector<cv::Point>> convexHull1(contours1.size()), convexHull2(contours2.size()), convexHull3(contours3.size());
	vector<vector<int>> convexHull1_I(contours1.size()), convexHull2_I(contours2.size()), convexHull3_I(contours3.size());
	vector<vector<cv::Vec4i>> defects1(contours1.size()), defects2(contours2.size());
	vector<cv::Point> largestContoursL2;
	vector<cv::Point> largestHullL2;
	vector<int> largestHullL2_I;
	vector<cv::Vec4i> defect2;
	int largestIndexL2 = 0;

	for (int i = 0; i < contours1.size(); i++) {
		cv::convexHull(contours1[i], convexHull1[i]);
		cv::convexHull(contours1[i], convexHull1_I[i]);
	}
	for (int i = 0; i < contours2.size(); i++) {
		cv::convexHull(contours2[i], convexHull2[i]);
		cv::convexHull(contours2[i], convexHull2_I[i]);

		mapCornerContourL1[i] = vector<cv::Point>();
	}
	for (int i = 0; i < contours3.size(); i++) {
		cv::convexHull(contours3[i], convexHull3[i]);
	}

	double maxArea = 0;
	for (int i = 0; i < contours2.size(); i++) {
		double a = cv::contourArea(contours2[i], false);
		if (a > maxArea) {
			largestIndexL2 = i;
			maxArea = a;
		}
	}

	if (maxArea != 0) {
		largestContoursL2 = contours2[largestIndexL2];
		largestHullL2 = convexHull2[largestIndexL2];
		largestHullL2_I = convexHull2_I[largestIndexL2];

		cv::convexityDefects(largestContoursL2, largestHullL2_I, defect2);
	}

	cv::cvtColor(maskL1, maskL1, cv::COLOR_GRAY2BGR);
	cv::cvtColor(maskL2, maskL2, cv::COLOR_GRAY2BGR);
	cv::cvtColor(maskL3, maskL3, cv::COLOR_GRAY2BGR);

	for (int j = 0; j < maskL1Corner.rows; j++) {
		for (int i = 0; i < maskL1Corner.cols; i++) {
			if ((int)maskL1Corner.at<float>(j, i) > 200) {
				cv::Point p(i, j);
				cv::circle(maskL1, p, 2, cv::Scalar(0, 0, 255), -1);
				//cv::circle(drawing, p, 2, cv::Scalar(0, 0, 255), -1);
				fingerL1.push_back(p);

				// find contour contian this corner
				for (int k = 0; k < contours1.size(); k++) {
					double c = cv::pointPolygonTest(contours1[k], p, false);
					if (c >= 0) {
						mapCornerContourL1[k].push_back(p);
					}
				}
			}
		}
	}

	if (maxArea != 0) {
		cv::drawContours(maskL2, convexHull2, largestIndexL2, cv::Scalar(0, 255, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	}
	for (int i = 0; i < defect2.size(); i++) {
		cv::Vec4i& v = defect2[i];
		float depth = v[3];
		if (depth > 10) {
			int startIndex = v[0];
			int endIndex = v[1];
			int farIndex = v[2];

			cv::Point startPoint(largestContoursL2[startIndex]);
			cv::Point endPoint(largestContoursL2[endIndex]);
			cv::Point farPoint(largestContoursL2[farIndex]);

			double ms = ((double)(startPoint.y - farPoint.y)) / (startPoint.x - farPoint.x);
			double me = ((double)(endPoint.y - farPoint.y)) / (endPoint.x - farPoint.x);
			double angle = atan((me - ms) / (1 + (ms * me))) * (180 / 3.14159265);

			if (angle < 0) {
				cv::circle(maskL2, startPoint, 2, cv::Scalar(0, 255, 0), -1);
				cv::circle(maskL2, endPoint, 2, cv::Scalar(0, 255, 0), -1);

				//cv::circle(drawing, startPoint, 2, cv::Scalar(0, 255, 0), -1);
				//cv::circle(drawing, endPoint, 2, cv::Scalar(0, 255, 0), -1);

				fingerL2.push_back(startPoint);
				fingerL2.push_back(endPoint);
			}
		}
	}

	//for (int i = 0; i < contours2.size(); i++) {
	//	cv::drawContours(maskL2, contours2, i, cv::Scalar(0, 255, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	//	cv::drawContours(maskL2, convexHull2, i, cv::Scalar(0, 255, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	//	//cv::drawContours(drawing, convexHull2, i, cv::Scalar(0, 255, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	//}
	for (int i = 0; i < contours3.size(); i++) {
		cv::drawContours(maskL3, contours3, i, cv::Scalar(255, 0, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
		cv::drawContours(maskL3, convexHull3, i, cv::Scalar(255, 0, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
		//cv::drawContours(drawing, convexHull3, i, cv::Scalar(255, 0, 0), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
	}

	// clustering
	// calculate distance matrix
	vector<vector<double>> distMat(fingerL2.size());
	for (int i = 0; i < distMat.size(); i++) {
		cv::Point pi = fingerL2[i];
		distMat[i] = vector<double>(fingerL2.size());

		for (int j = i + 1; j < distMat[i].size(); j++) {
			cv::Point pj = fingerL2[j];
			distMat[i][j] = sqrt(pow(pi.x - pj.x, 2) + pow(pi.y - pj.y, 2));
		}
	}

	// cluster near point
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
	// assign cluster to single point
	for (int i = 0; i < cluster.size(); i++) {
		if (cluster[i] == 0) {
			cluster[i] = clusterCount;
			clusterCount++;
		}
	}

	// find centroid of each cluster
	int m = cluster.size();
	for (int i = 1; i < m - 1; i++) {
		double x_sum = 0;
		double y_sum = 0;
		int x_count = 0;
		int y_count = 0;
		for (int j = 0; j < fingerL2.size(); j++) {
			if (cluster[j] == i) {
				x_sum += fingerL2[j].x;
				y_sum += fingerL2[j].y;
				y_count++;
				x_count++;
			}
		}
		if (x_count == 0) {
			continue;
		}
		int x = (int)(x_sum / x_count);
		int y = (int)(y_sum / y_count);
		cv::Point p(x, y);
		fingerL2Cluster.push_back(p);
	}

	vector<cv::Point> fingerL1Abs;
	for (map<int, vector<cv::Point>>::iterator it = mapCornerContourL1.begin(); it != mapCornerContourL1.end(); it++) {
		vector<cv::Point> points = it->second;
		int minI = 0;
		double minV = 800;
		bool isIgnore = false;
		for (int i = 0; i < points.size(); i++) {
			cv::Point p1 = points[i];
			for (int j = 0; j < fingerL2Cluster.size(); j++) {
				cv::Point p2 = fingerL2Cluster[j];
				double d = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));

				if (d < IGNORE_LAYER1_THRESHOLD + 5) {
					isIgnore = true;
					break;
				}
			}
			if (isIgnore) {
				break;
			}
			double dc = depthRaw[p1.y][p1.x];
			//double dc = sqrt(pow(p1.x-handPosX, 2) + pow(p1.y-handPosY, 2));
			if (dc < minV) {
				minV = dc;
				minI = i;
			}
		}
		if (isIgnore) {
			break;
		}
		if (points.size() > 0) {
			fingerL1Abs.push_back(points[minI]);
		}
	}

	for (int i = 0; i < fingerL2Cluster.size(); i++) {
		cv::circle(drawing, fingerL2Cluster[i], 4, cv::Scalar(0, 255, 0), -1);
	}
	for (int i = 0; i < fingerL1Abs.size(); i++) {
		cv::circle(drawing, fingerL1Abs[i], 4, cv::Scalar(0, 0, 255), -1);
	}

	float wx, wy, wz;
	int px, py;
	openni::DepthPixel pz;
	openni::CoordinateConverter::convertDepthToWorld(videoStream, handPosX, handPosY, handDepth, &wx, &wy, &wz);
	openni::CoordinateConverter::convertWorldToDepth(videoStream, wx + 50, wy, wz, &px, &py, &pz);
	int r = abs(handPosX - px);
	if (r > 0) {
		cv::Scalar rediusColor;
		if (fingerL2Cluster.size() + fingerL1Abs.size() == 5)
			rediusColor = cv::Scalar(0, 255, 255);
		else
			rediusColor = cv::Scalar(255, 255, 0);
		cv::circle(maskL1, cv::Point(handPosX, handPosY), r, rediusColor, 2);
		cv::circle(maskL2, cv::Point(handPosX, handPosY), r, rediusColor, 2);
		cv::circle(maskL3, cv::Point(handPosX, handPosY), r, rediusColor, 2);
		cv::circle(drawing, cv::Point(handPosX, handPosY), r, rediusColor, 2);
	}

	cv::circle(maskL1, cv::Point(handPosX, handPosY), 4, cv::Scalar(255, 0, 255), 2);
	cv::circle(maskL2, cv::Point(handPosX, handPosY), 4, cv::Scalar(255, 0, 255), 2);
	cv::circle(maskL3, cv::Point(handPosX, handPosY), 4, cv::Scalar(255, 0, 255), 2);
	cv::circle(drawing, cv::Point(handPosX, handPosY), 4, cv::Scalar(255, 0, 255), 2);

	imageFrame = depthFrame;

	//cv::imshow("mask1", maskL1);
	//cv::imshow("mask2", maskL2);
	//cv::imshow("mask3", maskL3);
	cv::imshow("drawing", drawing);
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
	/*if (numberOfHands > 0)
		return maskFrame;*/
	cv::bitwise_and(imageFrame, imageFrame, imageFrame, maskFrame);
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
							m2[y][x] = 255;
							m3[y][x] = 255;
						}
						else if (handDepth - 20 <= *depthPixel && *depthPixel < handDepth + 30) {
							img[y][x][0] = 0;
							img[y][x][1] = 255 - depthValue;
							img[y][x][2] = 0;

							m1[y][x] = 0;
							m2[y][x] = 255;
							m3[y][x] = 255;
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
