#include "pch.h"
#include "Application.h"
#include "Activator.h"
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;

Application::Application(std::shared_ptr<Activator> activator)	: mainFrameActivator(activator)
{

}

void Application::onInitial(std::string window_name)
{
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	windowName = window_name;

	for (std::vector<std::shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
		(*it)->onInitial();
	}
}

int Application::start()
{
	// Life Cycle Loop
	while (true) {
		// onPrepare
		for (std::vector<std::shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			(*it)->onPrepare();
		}

		// onReadFrame
		for (std::vector<std::shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			(*it)->onReadFrame();
		}

		// onModifyFrame
		for (std::vector<std::shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			(*it)->onModifyFrame();
		}

		imageFrame = mainFrameActivator->getImageFrame();


		// onDraw
		for (std::vector<std::shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			(*it)->onDraw(imageFrame);
		}

		cv::imshow(windowName, imageFrame);

		int key = cv::waitKey(5);
		if ((*onKeybordCallback)(key) != 0 ){
			break;
		}

		for (std::vector<std::shared_ptr<Activator>>::iterator it = activators.begin(); it != activators.end(); ++it) {
			(*it)->onPerformKeyboardEvent(key);
		}
	}


	// Die
	onDie();
	return 0;
}

void Application::registerActivator(std::shared_ptr<Activator> activator)
{
	activators.push_back(activator);
}

void Application::setMainFrameActivator(std::shared_ptr<Activator> activator)
{
	cout << "set main frame Activator " << activator->getName() << endl; 
	mainFrameActivator = activator;
}

void Application::setOnKeyboardCallback(int(*callback)(int key))
{
	onKeybordCallback = callback;
}

void Application::onDie()
{
	cv::destroyAllWindows();
}
