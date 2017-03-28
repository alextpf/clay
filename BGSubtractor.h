#if !defined BGSub
#define BGSub

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include "videoprocessor.h"

class BGSubtractor : public FrameProcessor 
{
public:

	// ctor
	BGSubtractor() : 
		m_learningRate(0.1) 
	{}

	void setLearningRate(float v)
	{
		m_learningRate = v;
	}

	//overwritten function
	virtual void process(cv::Mat &frame, cv::Mat &output)
	{
		mog(frame, output, m_learningRate);
	}

private:

	cv::BackgroundSubtractorMOG mog;
	float m_learningRate;
};


#endif
