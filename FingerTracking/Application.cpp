#include "pch.h"
#include "Application.h"
#include "Activator.h"
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;

Application::Application(map<string, shared_ptr<Activator>> mainFrameActivators) : mainFrameActivators(mainFrameActivators)
{

}

void Application::onInitial()
{
	
	for (map<string, shared_ptr<Activator>>::iterator it = mainFrameActivators.begin(); it != mainFrameActivators.end(); ++it) {
		cv::namedWindow(it->first, cv::WINDOW_NORMAL);
	}
	
	for (vector<shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
		(*it)->onInitial();
	}

	/*outVideo.open("test1.avi", cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), 15, cv::Size(640, 480), true);
	cout << "output open " << outVideo.isOpened() << endl;*/
}

int Application::start()
{
	// Life Cycle Loop
	while (true) {
		// onPrepare
		for (vector<shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			(*it)->onPrepare();
		}

		// onReadFrame
		for (vector<shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			(*it)->onReadFrame();
		}

		// onModifyFrame
		for (vector<shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			(*it)->onModifyFrame();
		}

		for (map<string, shared_ptr<Activator>>::iterator it = mainFrameActivators.begin(); it != mainFrameActivators.end(); ++it) {
			imageFrames[it->first] = it->second->getImageFrame();
		}

		// onDraw
		for (vector<shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			for (map<string, shared_ptr<Activator>>::iterator it2 = mainFrameActivators.begin(); it2 != mainFrameActivators.end(); ++it2) {
				(*it)->onDraw(it2->first, imageFrames[it2->first]);
			}
		}

		for (map<string, shared_ptr<Activator>>::iterator it = mainFrameActivators.begin(); it != mainFrameActivators.end(); ++it) {
			cv::imshow(it->first, imageFrames[it->first]);
			if (isWriteVideo)
				videoWriters[it->first] << imageFrames[it->first];
		}
		//outVideo << imageFrames["RGB"];

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

		for (vector<shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			(*it)->onPerformKeyboardEvent(key);
		}
	}


	// Die
	onDie();
	return 0;
}

void Application::registerActivator(shared_ptr<Activator> activator)
{
	activators.push_back(activator);
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
	for (map<string, shared_ptr<Activator>>::iterator it = mainFrameActivators.begin(); it != mainFrameActivators.end(); ++it) {
		char buffer[40];
		sprintf_s(buffer, "%d - %s.avi", ts, it->first.c_str());
		cv::VideoWriter outVideo;
		outVideo.open(buffer, cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), estimateFPS, cv::Size(640, 480), true);
		videoWriters[it->first] = outVideo;
		cout << "Start writing " << buffer << " FPS : " << estimateFPS << endl;
	}
	
}

void Application::stopWriteVideo()
{
	isWriteVideo = false;
	cout << "Stop writing video" << endl;
}

void Application::onDie()
{
	cv::destroyAllWindows();
}
