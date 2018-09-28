#include "pch.h"
#include "Combiner.h"

using namespace std;

class RGBDepthMaskCombiner : virtual public Combiner {
public:
	void onCombine(map<string, cv::Mat> imageFrame, map<string, int> signatures);
	cv::Mat getImageFrame();
	int getSignature();
protected:
	cv::Mat imageFrame;

	vector<vector<cv::Point>> contours;
	vector<cv::Point> largestContour;
	vector<cv::Point> largestHull;
};