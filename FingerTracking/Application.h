#pragma once
#include "Activator.h"
#include "Combiner.h"
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;

class Application {
public:
	Application(vector<shared_ptr<Activator>> activators, vector<shared_ptr<Combiner>> combiners);	// window_name: activator
	void onInitial();
	int start();
	void setOnKeyboardCallback(int (*callback)(int key));
	void startWriteVideo();
	void stopWriteVideo();
	void captureImage();
private:
	
	map<int, cv::Mat> imageFrames;
	map<int, cv::Mat> maskFrames;
	map<int, shared_ptr<Activator>> activators;		// { signature: Activator }
	map<int, shared_ptr<Combiner>> combiners;
	map<int, string> windowNames;
	

	cv::VideoWriter outVideo;
	map<int, cv::VideoWriter> videoWriters;
	time_t startTimestamp = 0;
	double estimateFPS = 0;
	long int frameCount = 0;
	bool isWriteVideo = false;
	bool isShowMask = true;

	int (*onKeybordCallback)(int key);
	void onDie();
};