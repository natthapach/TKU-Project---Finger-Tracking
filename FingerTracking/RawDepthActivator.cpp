#include "pch.h"
#include "RawDepthActivator.h"
#include "Constants.h"

RawDepthActivator::RawDepthActivator()
{
	name = "Depth Activator";
}

void RawDepthActivator::onInitial()
{
	openni::Status status = openni::STATUS_OK;

	status = openni::OpenNI::initialize();
	if (status != openni::STATUS_OK)
		return;

	status = device.open(openni::ANY_DEVICE);
	if (status != openni::STATUS_OK)
		return;

	status = sensor.create(device, openni::SENSOR_DEPTH);
	if (status != openni::STATUS_OK)
		return;

	status = rgbSensor.create(device, openni::SENSOR_COLOR);
	if (status != openni::STATUS_OK)
		return;

	status = sensor.start();
	if (status != openni::STATUS_OK)
		return;

	status = rgbSensor.start();
	if (status != openni::STATUS_OK)
		return;
}

void RawDepthActivator::onPrepare()
{
}

void RawDepthActivator::onReadFrame()
{
	openni::Status status = openni::STATUS_OK;
	openni::VideoStream* streamPointer = &sensor;
	int streamReadyIndex;
	status = openni::OpenNI::waitForAnyStream(&streamPointer, 1, &streamReadyIndex, 100);

	if (status != openni::STATUS_OK && streamReadyIndex != 0)
		return;

	openni::VideoFrameRef depthFrame;
	openni::VideoFrameRef rgbFrame;
	status = sensor.readFrame(&depthFrame);
	if (status != openni::STATUS_OK && !depthFrame.isValid())
		return;
	//status = rgbSensor.readFrame(&rgbFrame);
	//if (status != openni::STATUS_OK && !rgbFrame.isValid())
	//	return;

	int numberOfPoints = 0;
	int numberOfHandPoints = 0;
	calDepthHistogram(depthFrame, &numberOfPoints, &numberOfHandPoints);
	modifyImage(depthFrame, rgbFrame, numberOfPoints, numberOfHandPoints);
}

void RawDepthActivator::onModifyFrame()
{
	imageFrame = cv::Mat(480, 640, CV_8UC3, &img);
}

void RawDepthActivator::onDraw(std::string name, cv::Mat canvas)
{
}

void RawDepthActivator::onPerformKeyboardEvent(int key)
{
}

void RawDepthActivator::onDie()
{
}

cv::Mat RawDepthActivator::getImageFrame()
{
	return imageFrame;
}

std::string RawDepthActivator::getName()
{
	return name;
}

int RawDepthActivator::getSignature()
{
	return ACTIVATOR_RAW_DEPTH;
}

void RawDepthActivator::calDepthHistogram(openni::VideoFrameRef depthFrame, int * numberOfPoints, int * numberOfHandPoints)
{
	*numberOfPoints = 0;
	*numberOfHandPoints = 0;

	memset(depthHistogram, 0, sizeof(depthHistogram));
	for (int y = 0; y < depthFrame.getHeight(); ++y)
	{
		openni::DepthPixel* depthCell = (openni::DepthPixel*)
			(
			(char*)depthFrame.getData() +
				(y * depthFrame.getStrideInBytes())
				);
		for (int x = 0; x < depthFrame.getWidth(); ++x, ++depthCell)
		{
			if (*depthCell != 0)
			{
				depthHistogram[*depthCell]++;
				(*numberOfPoints)++;

				/*if (handDepth > 0 && numberOfHands > 0) {
					if (handDepth - RANGE <= *depthCell && *depthCell <= handDepth + RANGE)
						(*numberOfHandPoints)++;
				}*/
			}
		}
	}
	for (int nIndex = 1; nIndex < sizeof(depthHistogram) / sizeof(int); nIndex++)
	{
		depthHistogram[nIndex] += depthHistogram[nIndex - 1];
	}
}

void RawDepthActivator::modifyImage(openni::VideoFrameRef depthFrame, openni::VideoFrameRef rgbFrame, int numberOfPoints, int numberOfHandPoints)
{
	for (unsigned int y = 0; y < 480; y++) {
		for (unsigned int x = 0; x < 640; x++) {
			openni::DepthPixel* depthPixel = (openni::DepthPixel*)
				((char*)depthFrame.getData() + (y*depthFrame.getStrideInBytes())) + x;
			depthRaw[y][x] = *depthPixel;
			if (*depthPixel != 0) {
				uchar depthValue = (uchar)(((float)depthHistogram[*depthPixel] / numberOfPoints) * 255);
				img[y][x][0] = 255 - depthValue;
				img[y][x][1] = 255 - depthValue;
				img[y][x][2] = 255 - depthValue;
			}
			else {
				img[y][x][0] = 0;
				img[y][x][1] = 0;
				img[y][x][2] = 0;
			}
		}
	}
}

