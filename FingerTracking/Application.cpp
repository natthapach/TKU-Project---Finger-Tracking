#include "pch.h"
#include "Application.h"
#include "Activator.h"
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;

Application::Application(vector<shared_ptr<Activator>> activators, vector<shared_ptr<Combiner>> combiners)
{
	for (vector<shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
		this->activators[(*it)->getSignature()] = (*it);
	}
	for (vector<shared_ptr<Combiner>>::iterator it = combiners.begin(); it != combiners.end(); it++) {
		this->combiners[(*it)->getSignature()] = (*it);
	}
}

void Application::onInitial()
{
	
	for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
		cv::namedWindow(it->second->getName(), cv::WINDOW_NORMAL);
	}
	if (isShowMask) {
		for (map<int, shared_ptr<Combiner>>::iterator it = combiners.begin(); it != combiners.end(); it++) {
			cv::namedWindow(it->second->getName(), cv::WINDOW_NORMAL);
		}
	}
	
	for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
		it->second->onInitial();
	}
}

int Application::start()
{
	// Life Cycle Loop
	while (true) {
		// onPrepare
		for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
			it->second->onPrepare();
		}
		// onReadFrame
		for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
			it->second->onReadFrame();
		}

		// onModifyFrame
		for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
			it->second->onModifyFrame();
		}

		// Read Activator Frame
		for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
			imageFrames[it->first] = it->second->getImageFrame();
		}

		// onCombine
		for (map<int, shared_ptr<Combiner>>::iterator it = combiners.begin(); it != combiners.end(); it++) {
			it->second->onCombine(imageFrames);
		}

		// Read Combiner Mask
		for (map<int, shared_ptr<Combiner>>::iterator it = combiners.begin(); it != combiners.end(); it++) {
			maskFrames[it->first] = it->second->getMaskFrame();
		}
		// onMask
		for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
			for (map<int, cv::Mat>::iterator it2 = maskFrames.begin(); it2 != maskFrames.end(); it2++) {
				it->second->onMask(it2->first, it2->second);
			}
		}

		// onDraw
		for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
			for (map<int, cv::Mat>::iterator it2 = imageFrames.begin(); it2 != imageFrames.end(); it2++) {
				it->second->onDraw(it2->first, it2->second);
			}
		}

		// Show Image
		for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
			cv::imshow(it->second->getName(), imageFrames[it->first]);
			/*if (isWriteVideo)
				videoWriters[]*/
		}
		if (isShowMask) {
			for (map<int, shared_ptr<Combiner>>::iterator it = combiners.begin(); it != combiners.end(); it++) {
				cv::imshow(it->second->getName(), maskFrames[it->first]);
			}
		}
		

		if (frameCount == 0) {
			startTimestamp = time(nullptr);
		}
		else {
			time_t timeElapsed = time(nullptr) - startTimestamp;
			if (timeElapsed != 0)
				estimateFPS = ((double) frameCount) / timeElapsed;
		}
		frameCount++;

		int key = cv::waitKey(5);
		if ((*onKeybordCallback)(key) != 0 ){
			break;
		}

		// onPerformKeyboardEvent
		for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); it++) {
			it->second->onPerformKeyboardEvent(key);
		}
	}


	// Die
	onDie();
	return 0;
}


void Application::setOnKeyboardCallback(int(*callback)(int key))
{
	onKeybordCallback = callback;
}

void Application::startWriteVideo()
{
	if (estimateFPS == 0) {
		cout << "Processing FPS estimate, please try again later" << endl;
		return;
	}
	if (isWriteVideo) {
		cout << "Video is already writing now";
		return;
	}
	isWriteVideo = true;
	time_t ts = time(nullptr);
	for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
		char buffer[80];
		sprintf_s(buffer, "%d - %s.avi", ts, it->second->getName().c_str());
		cv::VideoWriter outVideo;
		outVideo.open(buffer, cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), estimateFPS, cv::Size(640, 480), true);
		videoWriters[it->first] = outVideo;
		cout << "Start writing " << buffer << " FPS : " << estimateFPS << endl;
	}
	
}

void Application::stopWriteVideo()
{
	isWriteVideo = false;
	for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
		videoWriters[it->first].release();
	}
	cout << "Stop writing video" << endl;
}

void Application::captureImage()
{
	time_t ts = time(nullptr);
	for (map<int, shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
		char buffer[80];
		sprintf_s(buffer, "%d - %s.jpg", ts, it->second->getName().c_str());
		cv::imwrite(buffer, imageFrames[it->first]);
	}
}

void Application::onDie()
{
	cv::destroyAllWindows();
}
