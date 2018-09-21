#include "pch.h"
#include "RGBActivator.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;

RGBActivator::RGBActivator()
{
	name = "RGB Activator";
}

void RGBActivator::onInitial()
{
	printf("RGB initial\n");
	openni::Status status = openni::STATUS_OK;
	status = openni::OpenNI::initialize();
	if (status != openni::STATUS_OK)
		return;

	status = device.open(openni::ANY_DEVICE);
	if (status != openni::STATUS_OK)
		return;
	
	status = sensor.create(device, openni::SENSOR_COLOR);
	if (status != openni::STATUS_OK)
		return;

	openni::VideoMode vmod;
	vmod.setFps(30);
	vmod.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
	vmod.setResolution(640, 480);
	status = sensor.setVideoMode(vmod);
	if (status != openni::STATUS_OK)
		return;

	status = sensor.start();
	if (status != openni::STATUS_OK)
		return;

}

void RGBActivator::onPrepare()
{
}

void RGBActivator::onReadFrame()
{
	openni::Status status = openni::STATUS_OK;
	openni::VideoStream* streamPointer = &sensor;
	int streamReadyIndex;
	status = openni::OpenNI::waitForAnyStream(&streamPointer, 1, &streamReadyIndex, 100);

	if (status != openni::STATUS_OK && streamReadyIndex != 0)
		return;

	openni::VideoFrameRef newFrame;
	status = sensor.readFrame(&newFrame);
	if (status != openni::STATUS_OK && !newFrame.isValid())
		return;

	for (unsigned int y = 0; y < 480; y++) {
		for (unsigned int x = 0; x < 640; x++) {
			OniRGB888Pixel* streamPixel = (OniRGB888Pixel*)((char*)newFrame.getData() + (y * newFrame.getStrideInBytes())) + x;
			img[y][x][0] = streamPixel->b;
			img[y][x][1] = streamPixel->g;
			img[y][x][2] = streamPixel->r;
		}
	}
}

void RGBActivator::onModifyFrame()
{
}

void RGBActivator::onDraw(string name, cv::Mat canvas)
{
}

void RGBActivator::onPerformKeyboardEvent(int key)
{
}

void RGBActivator::onDie()
{
}

cv::Mat RGBActivator::getImageFrame()
{
	return cv::Mat(480, 640, CV_8UC3, &img);
}

std::string RGBActivator::getName()
{
	return "RGB Activator";
}

