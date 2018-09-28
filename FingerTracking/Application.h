#pragma once
#include "Activator.h"
#include "Combiner.h"
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;

class Application {
public:
	Application(map<string, shared_ptr<Activator>> mainFrameActivators);	// window_name: activator
	void onInitial();
	int start();
	void registerActivator(shared_ptr<Activator> activator);
	void registerCombiner(string windowName, shared_ptr<Combiner> combiner);
	void setOnKeyboardCallback(int (*callback)(int key));
	void startWriteVideo();
	void stopWriteVideo();
	void captureImage();
private:
	vector<shared_ptr<Activator>> activators;
	map<string, cv::Mat> imageFrames;
	map<string, shared_ptr<Activator>> mainFrameActivators;		// { window_name : Activator }
	map<string, int> signatures;
	map<string, shared_ptr<Combiner>> combiners;

	cv::VideoWriter outVideo;
	map<string, cv::VideoWriter> videoWriters;
	time_t startTimestamp = 0;
	double estimateFPS = 0;
	long int frameCount = 0;
	bool isWriteVideo = false;

	int (*onKeybordCallback)(int key);
	void onDie();
};