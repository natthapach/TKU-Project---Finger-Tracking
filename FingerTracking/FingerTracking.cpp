// FingerTracking.cpp : This file contains the 'main' function. Program execution begins and ends there.
// test for push from Lab
//

#include "pch.h"
#include "Application.h"
#include "RGBActivator.h"
#include "DepthActivator.h"
#include "RawDepthActivator.h"
#include "RGBDepthMaskCombiner.h"
#include "HandDepthVisualizeActivator.h"

using namespace std;

std::shared_ptr<Activator> depthActivatorPtr(new DepthActivator());
std::shared_ptr<Activator> rgbActivatorPtr(new RGBActivator());
std::shared_ptr<Activator> rawDepthActivator(new RawDepthActivator());
std::shared_ptr<Activator> handDepthVisualizeActivator(new HandDepthVisualizeActivator());
std::shared_ptr<Combiner> rgbDepthMaskCombiner(new RGBDepthMaskCombiner());
//map<string, shared_ptr<Activator>> mainFrameActivators = { 
//															{"RGB", rgbActivatorPtr},
//															{"Depth", depthActivatorPtr},
//															{"Raw Depth", rawDepthActivator}
//														};
vector<shared_ptr<Activator>> activators = {
	//depthActivatorPtr,
	rgbActivatorPtr,
	//rawDepthActivator,
	handDepthVisualizeActivator
};
vector<shared_ptr<Combiner>> combiners = {
	rgbDepthMaskCombiner
};
Application mainApplication(activators, combiners);

int keyboardCallback(int key) {
	if (key == 27)
		return 1;
	if (key == 'v' || key == 'V') {
		cout << "Perfrom Key V" << endl;
		mainApplication.startWriteVideo();
	}
	else if (key == 'b' || key == 'B') {
		cout << "Perform key B" << endl;
		mainApplication.stopWriteVideo();
	}
	else if (key == 'c' || key == 'C') {
		mainApplication.captureImage();
	}
	return 0;
}

int main()
{
	cout << "sig depth " << depthActivatorPtr->getSignature() << endl;
	cout << "sig rgb " << depthActivatorPtr->getSignature() << endl;
	mainApplication.setOnKeyboardCallback(keyboardCallback);

	mainApplication.onInitial();
	mainApplication.start();
	return 0;
}