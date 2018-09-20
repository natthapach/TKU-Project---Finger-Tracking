#pragma once
#include "Activator.h"
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>

class Application {
public:
	Application(std::shared_ptr<Activator> activator);
	void onInitial(std::string windowName);
	int start();
	void registerActivator(std::shared_ptr<Activator> activator);
	void setMainFrameActivator(std::shared_ptr<Activator> activator);
	void setOnKeyboardCallback(int (*callback)(int key));
private:
	std::vector<std::shared_ptr<Activator>> activators;
	std::shared_ptr<Activator> mainFrameActivator;
	cv::Mat imageFrame;
	std::string windowName;
	std::vector<std::string> windowNames;

	int (*onKeybordCallback)(int key);
	void onDie();
};