#include "pch.h"
#include "RGBDepthMaskCombiner.h"
#include "Constants.h"
#include <stdio.h>

void RGBDepthMaskCombiner::onCombine(map<int, cv::Mat> imageFrames)
{
	cv::Mat depthFrame = imageFrames[ACTIVATOR_DEPTH];
	cv::Mat rgbFrame = imageFrames[ACTIVATOR_RGB];
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
		cv::dilate(mask, mask, cv::Mat());
		imageFrame = mask;
	}
	else {
		imageFrame = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	}
}

cv::Mat RGBDepthMaskCombiner::getMaskFrame()
{
	return imageFrame;
}

int RGBDepthMaskCombiner::getSignature()
{
	return COMBINER_RGB_DEPTH;
}

string RGBDepthMaskCombiner::getName()
{
	return WINDOW_NAME_COMBINER_RGB_DEPTH;
}
