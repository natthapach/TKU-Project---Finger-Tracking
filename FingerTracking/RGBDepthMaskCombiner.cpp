#include "pch.h"
#include "RGBDepthMaskCombiner.h"
#include "Constants.h"

void RGBDepthMaskCombiner::onCombine(map<string, cv::Mat> imageFrames, map<string, int> signatures)
{
	string depthKey;
	string rgbKey;
	for (map<string, int>::iterator it = signatures.begin(); it != signatures.end(); it++) {
		if (it->second == ACTIVATOR_DEPTH) {
			depthKey = it->first;
		}
		else if (it->second == ACTIVATOR_RGB) {
			rgbKey = it->first;
		}
	}
	
	cv::Mat depthFrame = imageFrames[depthKey];
	cv::Mat rgbFrame = imageFrames[rgbKey];
	cv::Mat res;

	if (depthFrame.type() == CV_8UC1 && rgbFrame.type() == CV_8UC1) {
		cv::bitwise_and(depthFrame, rgbFrame, imageFrame);
		cv::Mat mask = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		// find contours
		vector<cv::Vec4i> hierarchy;
		cv::findContours(imageFrame, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		if (contours.size() == 0)
			return;

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
		cv::cvtColor(imageFrame, imageFrame, cv::COLOR_GRAY2BGR);
		cv::drawContours(mask, vector<vector<cv::Point>>{ largestContour }, 0, cv::Scalar(255, 255, 255), -1, 8, vector<cv::Vec4i>(), 0, cv::Point());
		cv::convexHull(cv::Mat(largestContour), largestHull);
		cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
		cv::drawContours(mask, vector<vector<cv::Point>>{ largestHull }, 0, cv::Scalar(0, 255, 0), 2, 8, vector<cv::Vec4i>(), 0, cv::Point());
		//cv::dilate(mask, mask, cv::Mat());
		//cv::GaussianBlur(mask, mask, cv::Size(3, 3), 0);
		imageFrame = mask;
	}
	else {
		imageFrame = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	}
}

cv::Mat RGBDepthMaskCombiner::getImageFrame()
{
	return imageFrame;
}

int RGBDepthMaskCombiner::getSignature()
{
	return COMBINER_RGB_DEPTH;
}
