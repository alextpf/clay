#if !defined BGSubOptFlow
#define BGSubOptFlow

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include "videoprocessor.h"

class BGSubtractorOptFlow : public FrameProcessor
{
public:

    // ctor
    BGSubtractorOptFlow() :
        m_learningRate(0.1f)
    {}

    void setLearningRate(float v)
    {
        m_learningRate = v;
    }

    //overwritten function
    virtual void process(cv::Mat &frame, cv::Mat &output);

private:
    bool AcceptTrackedPoint(int i);
    void HandleTrackedPoints(cv::Mat &frame);

    //----------------------------------------
    void ShapeConstraint(cv::Mat & maskBak);

    //----------------------------------------
	void Segmentation(cv::Mat &frame, cv::Mat& maskBak, cv::Mat & frameBak);
    void CannyEdge(cv::Mat& newImg);
	void WaterShedEdge(cv::Mat & newImg, cv::Mat & output, int i, cv::Rect roi);
    void GrabCutEdge( cv::Mat & newImg, cv::Point center, int leftEdge, int topEdge, int width, int height);

    //----------------------------------------
    void OpticalFlow(cv::Mat & frameBak);

    //----------------------------------------
    void ReOrganizeContours();
    void DrawContour(cv::Mat & frame);
	void DrawTrackedObjs(cv::Mat & output);

    std::vector< std::vector < cv::Point > > m_Features;
    cv::Mat m_MarkerFG;
    cv::Mat m_MarkerBG;

    std::vector< cv::Vec4i > m_Hierarchy;
    std::vector< std::vector < cv::Point > > m_Contours;
    std::vector<int> m_RemovalIdx;
    std::vector<int> m_InFavorIdx;

    std::vector<uchar> m_Status; // status of tracked features
    cv::BackgroundSubtractorMOG mog;
    float m_learningRate;

    cv::Mat m_gray_curr;    	// current gray-level image
    cv::Mat m_gray_prev;		// previous gray-level image
    std::vector<cv::Point2f> m_OptFeatures[2]; // tracked features from 0->1

    //for debug
    std::string m_TypeName;
};


#endif
