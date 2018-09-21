// FingerTracking.cpp : This file contains the 'main' function. Program execution begins and ends there.
// test for push from Lab
//

#include "pch.h"
#include "Application.h"
#include "RGBActivator.h"
#include "DepthActivator.h"

using namespace std;

std::shared_ptr<Activator> depthActivatorPtr(new DepthActivator());
std::shared_ptr<Activator> rgbActivatorPtr(new RGBActivator());
map<string, shared_ptr<Activator>> mainFrameActivators = { {"RGB", rgbActivatorPtr}, {"Depth", depthActivatorPtr} };
Application mainApplication(mainFrameActivators);

int keyboardCallback(int key) {
	if (key == 27)
		return 1;

	return 0;
}

int main()
{
	mainApplication.registerActivator(depthActivatorPtr);
	mainApplication.registerActivator(rgbActivatorPtr);
	mainApplication.setOnKeyboardCallback(keyboardCallback);

	mainApplication.onInitial();
	mainApplication.start();
	return 0;
}