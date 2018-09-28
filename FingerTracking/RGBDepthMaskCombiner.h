#include "pch.h"
#include "Combiner.h"

using namespace std;

class RGBDepthMaskCombiner : virtual public Combiner {
public:
	void onCombine(map<int, cv::Mat> imageFrame);
	cv::Mat getMaskFrame();
	int getSignature();
	string getName();
protected:
	cv::Mat imageFrame;

	vector<vector<cv::Point>> contours;
	vector<cv::Point> largestContour;
	vector<cv::Point> largestHull;
};