#include <iostream>
#include <algorithm>
#include <string>
#include <opencv2/video/tracking.hpp>
#include "BGSubtractorOptFlow.h"

#define NUM_ITER 1
#define MOVE_THRESH 2 //this should be able to be calculated by some assumption
#define SCORE_PERCENTAGE 0.7f
#define DEVIATION_THRESH 2
#define AREA_RATIO_THRESH 0.3f
#define CLAY_MAX_RATIO_SQR 4.0f * 4.0f // let's say the clays's largest dimension is at most 4 times its smallest dimension

#define DEBUG
#define DEBUG_SEG
#define DEBUG_MOG_MORPH
//=======================================================================
// helper function to show type
std::string type2str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

//=======================================================================
void ContourMinMaxSize(
    cv::Point center,
    const std::vector < cv::Point > & contour,
    float & minSqr,
    float & maxSqr)
{
    // init
    minSqr = 10000000000.0f;
    maxSqr = 0.0f;

    const int size = contour.size();
    float lenSqr;

    for (int i = 0; i < size; i++)
    {
        const cv::Point halfDiff ( abs( contour[i].x - center.x ), abs( contour[i].y - center.y ) );
        const cv::Point diff1 = halfDiff * 2;
        const cv::Point diff = diff1 + cv::Point(1,1); // adding 1 is for inclusive length

        lenSqr = diff.x * diff.x + diff.y * diff.y;

        if (lenSqr > maxSqr)
        {
            maxSqr = lenSqr;
        }

        if (lenSqr < minSqr)
        {
            minSqr = lenSqr;
        }
    }
} // ContourMinMaxSize

//=======================================================================
cv::Point ContourCenter(const std::vector < cv::Point > & contour)
{
    cv::Point sum(0, 0);
    const int size = contour.size();

    for (int i = 0; i < size; i++)
    {
        sum += contour[i];
    }
    const cv::Point center = sum * (1.0 / size);

    return center;
}

//=======================================================================
cv::Point GetContourStatistics (
    const std::vector < cv::Point > & contour,
    float & maxX,
    float & maxY,
    float & minX,
    float & minY,
    float & width,
    float & height )
{
    cv::Point sum(0, 0);
    const int size = contour.size();

    maxX = -1.0f;
    maxY = -1.0f;
    minX = 1000000000000.0f;
    minY = 1000000000000.0f;

    for (int i = 0; i < size; i++)
    {
        const cv::Point pt = contour[i];
        sum += pt;

        if (pt.x > maxX)
        {
            maxX = pt.x;
        }

        if (pt.x < minX)
        {
            minX = pt.x;
        }

        if (pt.y > maxY)
        {
            maxY = pt.y;
        }

        if (pt.y < minY)
        {
            minY = pt.y;
        }
    }

    width = maxX - minX + 1;
	height = maxY - minY + 1;

    const cv::Point center = sum * (1.0 / size);

    return center;
}// GetContourStatistics


void BGSubtractorOptFlow::OpticalFlow(cv::Mat & frameBak)
{
    // Save the feature points for the next round
    m_Features = m_Contours;

    bool cleanUp = false;

    // track features
    std::vector<float> err;

    if (m_OptFeatures[0].size() > 0)
    {
        //debug: show
        /*cv::imshow("prev", m_gray_prev);
        cv::imshow("curr", m_gray_curr);*/
        ////////////////

        cv::calcOpticalFlowPyrLK(m_gray_prev, m_gray_curr, // 2 consecutive images
            m_OptFeatures[0], // input point position in first image
            m_OptFeatures[1], // output point postion in the second image
            m_Status,    // tracking success
            err);      // tracking error

        HandleTrackedPoints(frameBak);
    }
    else
    {
        cleanUp = true;
    }// if (m_OptFeatures[0].size() > 0)

    // fill up the features for tracking in the next frame
    m_OptFeatures[0].clear();

    // put the remaining contours into a feature point array "m_OptFeatures[0]"
    const int numContours = m_Features.size();

    for (int i = 0; i < numContours; i++)
    {
        m_OptFeatures[0].insert(m_OptFeatures[0].end(), m_Features[i].begin(), m_Features[i].end());
    }// for numContours

#ifdef DEBUG
    cv::imshow("optFlow", frameBak);
#endif

	//debug:
    // init with zeros
	
    cv::Mat matrix = cv::Mat::zeros(frameBak.rows, frameBak.cols, frameBak.type());
    if (!cleanUp)
    {
		DrawContour(matrix);
    }

#ifdef DEBUG
	cv::imshow("optFlow result", matrix);
#endif
	///////////////

} // OpticalFlow

//=======================================================================
void BGSubtractorOptFlow::process(cv::Mat &frame, cv::Mat &output)
{
    // init output
    output = cv::Mat::zeros(frame.rows, frame.cols, frame.type());

    // copy the current frame; convert to gray-level image
    //cv::cvtColor(frame, m_gray_curr, CV_BGR2GRAY);

    //debug - a copy of the raw img
    cv::Mat frameBak = frame.clone();
    /////
	static bool doMorph = true;
    static bool doOptFlow = true;
    static bool doSegmentation = true;

	////////////////////
	// morph options:
	////////////////////
	int enlargeObjLevel = 1; //1: no enlargement

    // Copy current frame to m_gray_curr and convert it to single channel
    //m_gray_curr = frame.clone();
    cv::cvtColor(frame, m_gray_curr, CV_BGR2GRAY); // CV_8U

	//////////////////////////////////////////////////
    // 1. mixture of gaussian background subtractor
	//////////////////////////////////////////////////
    mog(frame, output, m_learningRate);

	//////////////////////////////////////////////////
    // 2. do noise clean up by Morph op
	//////////////////////////////////////////////////

    /*cv::morphologyEx(output, output, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 1);
    cv::morphologyEx(output, output, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);*/
	if (doMorph)
	{
		dilate(output, output, cv::Mat(), cv::Point(-1, -1), NUM_ITER);
		erode(output, output, cv::Mat(), cv::Point(-1, -1), NUM_ITER * 2);

		// save a copy for watershed use
		m_MarkerFG = output.clone();// CV_U8C1

		//cv::Mat marker;
		//marker.convertTo(marker, CV_32S); // foreground. convert to 32 bit single channel

		//// background
		//cv::rectangle(marker, cv::Point(5, 5), cv::Point(marker.cols - 5, marker.rows - 5), cv::Scalar(128), 3);

		//cv::Mat tmp;
		//marker.convertTo(tmp, CV_8U);

		////debug
		//cv::imshow("marker img", marker);
		//cv::imshow("marker img", tmp);
		//cv::watershed(frameBak, marker);
		////convert back to U8
		//marker.convertTo(tmp, CV_8U);
		//cv::imshow("watershed img", tmp);
		////

		dilate(output, output, cv::Mat(), cv::Point(-1, -1), enlargeObjLevel * NUM_ITER);

		//// for watersheding
		//dilate(output, m_MarkerBG, cv::Mat(), cv::Point(-1, -1), 4 * NUM_ITER);
#ifdef DEBUG
#ifdef DEBUG_MOG_MORPH
		//debug
		cv::imshow("mog+morph results", output);
#endif
#endif

	}//doMorph
	else
	{
#ifdef DEBUG
		////debug
		char name[50];
		sprintf_s(name, "mog results");
		cv::imshow(name, output);
#endif


		m_MarkerFG = output.clone();// CV_U8C1

		dilate(m_MarkerFG, m_MarkerFG, cv::Mat(), cv::Point(-1, -1), NUM_ITER);
		erode(m_MarkerFG, m_MarkerFG, cv::Mat(), cv::Point(-1, -1), NUM_ITER * 2);
	} // Do Morph
    ////////////////

    // 3. find the contours of the result

    //cv::Mat mask( output.rows, output.cols, CV_8UC1 );
    //cv::cvtColor( output, mask, CV_BGR2GRAY);
    cv::Mat mask;

    cv::threshold(output, mask, 128, 255, cv::THRESH_BINARY);// 255: inside, 0: outside, i.e. object is represented by white region

    // NOTICE: cv::findContours() alters the source, so MAKE A COPY BEFORE YOU DO IT
    cv::Mat maskBak = mask.clone();

    cv::findContours(mask, m_Contours, m_Hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	/////////////////////////////
	// Do shape constraint
	/////////////////////////////
    ShapeConstraint(maskBak);

#ifdef DEBUG
	//debug
	cv::Mat tmpp = cv::Mat::zeros(frame.rows, frame.cols, frame.type());
	DrawContour(tmpp);
	cv::imshow("mog+morph+shape constraint", tmpp );
#endif

	// ini m_InFavorIdx
	int const numContour = m_Contours.size();
	m_InFavorIdx.clear();
	for (int i = 0; i < numContour; i++)
	{
		m_InFavorIdx.push_back(0);
	}

	/////////////////////////////
	// Do segmentation
	/////////////////////////////
    if (doSegmentation)
    {
		Segmentation(frame, maskBak, frameBak);
    }

    ///////////////////////////////////////////////////////////////////
    // Do optical flow
    ///////////////////////////////////////////////////////////////////

    if (doOptFlow)
    {
        OpticalFlow(frameBak);
    }// if (doOptFlow)

	// swap current and prev frames
	cv::swap(m_gray_curr, m_gray_prev);

	/////////////////////////////////////////
	// draw rectangle on the output image
	/////////////////////////////////////////
	output = frame;
	DrawTrackedObjs(output);

}// process

//=======================================================================
void BGSubtractorOptFlow::DrawTrackedObjs(cv::Mat & output)
{
	int const numContour = m_Contours.size();
	
	cv::Scalar green(0, 255, 0);
	cv::Scalar red(0, 0, 255);
	cv::Scalar color;

	for (int i = 0; i < numContour; i++)
	{
		if (m_InFavorIdx[i] != 0)
		{
			color = m_InFavorIdx[i] == 2 ? red : green;

			float maxX, maxY, minX, minY;
			float width, height;

			cv::Point center = GetContourStatistics(m_Contours[i], maxX, maxY, minX, minY, width, height);

			//create a rect that's X times larger than the bounding box of the contour
			static float scale = 2.0f;

			int leftEdge = minX - scale * width;
			leftEdge = leftEdge < 0.0f ? 0.0f : leftEdge;

			int rightEdge = maxX + scale * width;
			rightEdge = rightEdge > output.cols - 1 ? output.cols - 1 : rightEdge;

			int topEdge = minY - scale * height;
			topEdge = topEdge < 0.0f ? 0.0f : topEdge;

			int bottomEdge = minY + scale * height;
			bottomEdge = bottomEdge > output.rows - 1 ? output.rows - 1 : bottomEdge;

			cv::rectangle(output, cv::Point(leftEdge, topEdge), cv::Point(rightEdge, bottomEdge), color);
		}		
	}// for i

}//DrawTrackedObjs

//=======================================================================
void BGSubtractorOptFlow::DrawContour(cv::Mat & frame)
{
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int contourThickness = 1;
    const int numContours = m_Contours.size();
    for (int idx = 0; idx < numContours; idx++)
    {
        const int contourSize = m_Contours[idx].size();

        for (int m = 0; m < contourSize - 1; m++)
        {
            cv::line(frame, m_Contours[idx][m], m_Contours[idx][m + 1], cv::Scalar(255, 255, 255));
        }

        // last point wrap around
        cv::line(frame, m_Contours[idx][contourSize-1], m_Contours[idx][0], cv::Scalar(255, 255, 255));

    }// for idx
}
//=======================================================================
void BGSubtractorOptFlow::ReOrganizeContours()
{
    int numContours = m_Contours.size();
    int rmIdx = 0;
    int removalSize = m_RemovalIdx.size();
    int k = 0;
    for (int i = 0; i < numContours; i++)
    {
        if (rmIdx < removalSize && i == m_RemovalIdx[rmIdx])
        {
            rmIdx++;
            continue;
        }
        m_Contours[k++] = m_Contours[i];
    }

    m_Contours.resize(k);
    m_RemovalIdx.clear();
}//ReOrganizeContours

//=======================================================================
std::vector< cv::Point > FindNonZeroLoc(const cv::Mat & seedFG)
{
	std::vector< cv::Point > pts;

	int value;
	for (int r = 0; r < seedFG.rows; r++)
	{
		for (int c = 0; c < seedFG.cols; c++)
		{
			value = (int)seedFG.at<uchar>(r, c);
			if (value > 128)
			{
				pts.push_back(cv::Point(r, c));
			}
		}// for c
	}// for r

	return pts;
}// FindNonZeroLoc

//=======================================================================
std::vector<cv::Point> RegionGrow( cv::Point loc, cv::Mat & mask)
{
	std::vector<cv::Point> pts;

	if (loc.x >= mask.rows || loc.x < 0 || loc.y >= mask.cols || loc.y < 0)
	{
		std::cout << "error region growing" << std::endl;
		return pts;
	}

	return pts;

}// RegionGrow

//=======================================================================
void BGSubtractorOptFlow::ShapeConstraint( cv::Mat & maskBak)
{
	///////////////////////
	// Do decluttering
	///////////////////////
	int scale = 10;
	cv::Size size(maskBak.rows / scale, maskBak.cols / scale);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, size, cv::Point(-1, -1));
	cv::Mat dilatedMask;
	cv::dilate(maskBak, dilatedMask, kernel);
	
	//debug
	cv::imshow("dilated mask", dilatedMask);
	///////
	
	cv::Mat erodedMask;
	cv::erode(maskBak, erodedMask, kernel);

	//debug
	cv::imshow("eroded mask", erodedMask);
	///////

	cv::Mat xorMask;
	xorMask = erodedMask ^ dilatedMask;

	//debug
	cv::imshow("xor mask", xorMask);
	///////

	std::vector< cv::Point > nonZeros = FindNonZeroLoc(xorMask);

	cv::Mat newMask = maskBak.clone();

	int numNonZeros = nonZeros.size();

	for (int i = 0; i < numNonZeros; i++)
	{
		std::vector<cv::Point> region = RegionGrow(nonZeros[i], newMask);
	}

    const int numContours = m_Contours.size();
    
    for (int i = 0; i < numContours; i++)
    {
        cv::Point2i center = ContourCenter(m_Contours[i]);

		/////////////////////////////////////////////////////////
        // 1: if the center is not inside the mass, remove it. 
		// i.e. the shape has to be convex
		/////////////////////////////////////////////////////////
        const int centerPt = (int)maskBak.at<uchar>(center.y, center.x);

        //// look at 4 connected component
        //const int top = center.y - 1 >= 0 ? center.y - 1 : center.y;
        //const int bot = center.y + 1 <= maskBak.rows ? center.y + 1 : center.y;
        //const int right = center.x + 1 <= maskBak.cols ? center.x + 1 : center.x;
        //const int left = center.x - 1 >= 0 ? center.x - 1 : center.x;

        //const int topPt = (int)maskBak.at<uchar>(top, center.x);
        //const int botPt = (int)maskBak.at<uchar>(bot, center.x);
        //const int leftPt = (int)maskBak.at<uchar>(center.y, left);
        //const int rightPt = (int)maskBak.at<uchar>(center.y, right);

        //if ( centerPt < 128 && topPt < 128 && botPt < 128 && leftPt < 128 && rightPt < 128)
        if (centerPt < 128)
        {
            m_RemovalIdx.push_back(i);
            continue;
        }
		
        // criteria 2: use the size
        float minSqr, maxSqr;

        ContourMinMaxSize(center, m_Contours[i], minSqr, maxSqr);
        const float ratio = maxSqr / minSqr;

        if (ratio > CLAY_MAX_RATIO_SQR)
        {
            m_RemovalIdx.push_back(i);
            continue;
        }

    }//for (int i = 0; i < numContours; i++)

    ReOrganizeContours();
}// ShapeConstraint

//=======================================================================
void BGSubtractorOptFlow::CannyEdge(cv::Mat& newImg)
{
    cv::Mat canny_output;
    int low = 125;
    int high = 350;
    cv::Canny(newImg, canny_output, low, high);

    //do a closing
    dilate(canny_output, canny_output, cv::Mat(), cv::Point(-1, -1), NUM_ITER);
    erode(canny_output, canny_output, cv::Mat(), cv::Point(-1, -1), NUM_ITER);

    //debug
    char name[50];
    sprintf_s(name, "canny");
    cv::imshow(name, canny_output);
    //

    /// Find contours
    std::vector< cv::Vec4i > cannyHierarchy;
    std::vector< std::vector < cv::Point > > cannyContours;

    cv::findContours(canny_output, cannyContours, cannyHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    cv::Mat tmpMask(newImg.size(), CV_8U, cv::Scalar(0));
    cv::drawContours(tmpMask, cannyContours, -1, cv::Scalar(255), CV_FILLED);

    //debug
    sprintf_s(name, "canny filled");
    cv::imshow(name, tmpMask);
    //

    //int contourThickness = 1;

    //for (int idx = 0; idx >= 0; idx = cannyHierarchy[idx][0])
    //{
    //	// if the index is not in the removal list, draw it
    //	//cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
    //	cv::Scalar color(0, 0, 0);
    //	cv::drawContours(newImg, cannyContours, idx, color, contourThickness, 8, cannyHierarchy);
    //}

    //debug
    sprintf_s(name, "newImg");
    cv::imshow(name, newImg);
}//CannyEdge

//=======================================================================
cv::Point FindFirstNonZeroLoc( const cv::Mat & seedFG )
{
    cv::Point seed;

    int value;
    for (int r = 0 ; r < seedFG.rows ; r++)
    {
        for (int c = 0 ; c < seedFG.cols ; c++)
        {
            value = (int)seedFG.at<uchar>(r, c);
            if ( value > 128 )
            {
                seed.x = c;
                seed.y = r;

                break;
            }
        }// for c

        if ( value > 128 )
        {
            break;
        }
    }// for r

    return seed;
}// FindFirstNonZeroLoc

//=======================================================================
void BGSubtractorOptFlow::WaterShedEdge(cv::Mat & newImg, cv::Mat & output, int i, cv::Rect roi)
{
    ///////////////////////////////////////////////////////////////////
    // criteria 3: do segmentation with the contour as the initial seed,
    // and compare the results
    ///////////////////////////////////////////////////////////////////

    //cv::Mat seedBG = m_MarkerBG(roi).clone(); //CV_8U
    //cv::Mat seedFG = m_MarkerFG(roi).clone(); //CV_8U
    //cv::threshold(seedBG, seedBG, 1, 128, cv::THRESH_BINARY_INV);

    //cv::Mat seedMask(seedBG.size(), CV_8U, cv::Scalar(0));
    //seedMask = seedBG + seedFG;

    /*
    cv::rectangle(seedMask,
    cv::Point(seedMask.cols / 2 - 2, seedMask.rows / 2 - 2),
    cv::Point(seedMask.cols / 2 + 2, seedMask.rows / 2 + 2),
    cv::Scalar(255), 2);

    cv::rectangle(seedMask,
    cv::Point(0, 0),
    cv::Point(seedMask.cols - 1, seedMask.rows - 1),
    cv::Scalar(128), 2);
    */
#ifdef DEBUG
#ifdef DEBUG_SEG
    //debug
    char name[50];
    sprintf_s(name, "newImg%i",i);
    cv::imshow(name, newImg);
#endif
#endif
    ///*cv::Mat tmp;
    //seedMask.convertTo(tmp,CV_)*/
    //sprintf_s(name, "seedMask");
    //cv::imshow(name, seedMask);
    //m_TypeName = type2str(newImg.type());
    //printf("%s\n", m_TypeName.c_str());
    //m_TypeName = type2str(seedMask.type());
    //printf("%s\n", m_TypeName.c_str());
    ///////////////////////////
    cv::Mat dst = newImg.clone();
    cv::Point seed(0,0); // top-left corner
    int	loDiff = 20;
    int	upDiff = 20;

    int	flags = 4 + cv::FLOODFILL_FIXED_RANGE + cv::FLOODFILL_MASK_ONLY + (255 << 8);

    ////////////////////////
    // background seed
    ////////////////////////
    cv::Mat outerMask(newImg.size(), CV_8UC1);
    outerMask.setTo(0);

    cv::copyMakeBorder(outerMask, outerMask, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    cv::Scalar newVal;
    cv::Rect ccomp;

    cv::floodFill(dst, outerMask, seed, newVal, &ccomp, cv::Scalar(loDiff, loDiff, loDiff), cv::Scalar(upDiff, upDiff, upDiff), flags);

    ////////////////////////
    // foreground seed
    ////////////////////////
    cv::Mat innerMask = outerMask.clone();
    innerMask.setTo(0);

    cv::Mat seedFG = m_MarkerFG(roi).clone(); //CV_8U
    seed = FindFirstNonZeroLoc(seedFG);

    //sanity check
    if (seed.x >= dst.cols || seed.y >= dst.rows )
    {
        std::cout << "size error" << std::endl;
        return;
    }

    cv::floodFill(dst, innerMask, seed, newVal, &ccomp, cv::Scalar(loDiff, loDiff, loDiff), cv::Scalar(upDiff, upDiff, upDiff), flags);

    cv::Mat Mask = outerMask.clone();
    outerMask = outerMask / 2;
    Mask = outerMask | innerMask;

#ifdef DEBUG
#ifdef DEBUG_SEG
    //debug
    sprintf_s(name, "Seed regions%i",i);
    //cv::imshow(name, tmp);
    cv::imshow(name, Mask);
    ////////////////
#endif
#endif

    ///////////////////////////////
    //	Perform	watershed
    ///////////////////////////////

    cv::Mat labelImage(newImg.size(), CV_32SC1);
    labelImage = Mask(cv::Rect(1, 1, newImg.cols, newImg.rows));
    labelImage.convertTo(labelImage, CV_32SC1);

    cv::watershed(newImg, labelImage);
    labelImage.convertTo(output, CV_8U);

#ifdef DEBUG
#ifdef DEBUG_SEG
    //debug
    sprintf_s(name, "Watershed%i", i);
	cv::imshow(name, output);
#endif
#endif
    //// do watershed
    //seedMask.convertTo(seedMask, CV_32S);
    //cv::watershed(newImg, seedMask);

    ////debug
    //cv::Mat tmp;
    //seedMask.convertTo(tmp, CV_8U);
    //sprintf_s(name, "watershed");
    //cv::imshow(name, tmp);
    /////
}//WaterShedEdge

void BGSubtractorOptFlow::GrabCutEdge(
    cv::Mat & newImg,
    cv::Point center,
    int leftEdge,
    int topEdge,
    int width,
    int height)
{

    //cv::Mat seedMask = marker(roi).clone; //CV_8U
    //for (int r = 0; r < seedMask.rows; r++)
    //{
    //	for (int c = 0; c < seedMask.cols; c++)
    //	{
    //		if ((int)seedMask.at<uchar>(r, c) == 255)
    //		{
    //			//forground
    //			seedMask.at<uchar>(r, c) = cv::GC_FGD;
    //			//debug
    //			int t;
    //			t = (int)seedMask.at<uchar>(r, c);
    //		}
    //		else
    //		{
    //			seedMask.at<uchar>(r, c) = cv::GC_PR_BGD;

    //			//debug
    //			int t;
    //			t = (int)seedMask.at<uchar>(r, c);
    //		}//if
    //	}//for c
    //}//for r
    cv::Mat result;
    cv::Mat bgModel, fgModel; // the models (internally used)
    cv::Rect rectangle(center.x - leftEdge, center.y - topEdge, width / 2 + 1, height / 2 + 1);
    int numIte = cv::min(rectangle.size().width, rectangle.size().height);
    numIte = cv::min(numIte, 5);

    cv::grabCut(newImg,
        result,
        rectangle,
        bgModel, fgModel, // models
        numIte, // number of iterations
        cv::GC_INIT_WITH_RECT); // use rectangle

    //debug:show results
    // checking first bit with bitwise-and
    result = result & 1; // will be 1 if FG

    //cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ); // Generate output image

    cv::Mat foreground(newImg.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat ones(newImg.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    ones.copyTo(foreground, result);

    char name[50];
    sprintf_s(name, "newImg");
    cv::imshow(name, newImg);

    sprintf_s(name, "grabResult");
    cv::imshow(name, foreground);
    //////////////////
}//GrabCutEdge

//=======================================================================
int GetAreaInMask(cv::Mat & mask)
{
	int area = 0;

	for (int r = 0; r < mask.rows; r++)
	{
		for (int c = 0; c < mask.cols; c++)
		{
			int value = (int)mask.at<uchar>(r, c);
			if (value == 255)
			{
				area++;
			}
		}// for c
	}// for r

	return area;
}//GetAreaInMask

//=======================================================================
void BGSubtractorOptFlow::Segmentation(cv::Mat &frame, cv::Mat& maskBak, cv::Mat & frameBak)
{
    ////////////////////////////
    // segmentation section
    ////////////////////////////

    static bool doCanny = false; // doesn't perform as I expected
    static bool doWaterSheding = true; // doesn't perform as I expected
    static bool doGrabCut = false;

    const int numContours = m_Contours.size();

	if (m_InFavorIdx.size() != numContours)
	{
		std::cout << "error" << std::endl;
		return;
	}

    for (int i = 0; i < numContours; i++)
    {
        float maxX, maxY, minX, minY;
        float width, height;

        cv::Point center = GetContourStatistics(m_Contours[i], maxX, maxY, minX, minY, width, height);

        //create a rect that's X times larger than the bounding box of the contour
        static float scale = 2.0f;

        int leftEdge = minX - scale * width;
        leftEdge = leftEdge < 0.0f ? 0.0f : leftEdge;

        int rightEdge = maxX + scale * width;
        rightEdge = rightEdge > frame.cols - 1 ? frame.cols - 1 : rightEdge;

        int topEdge = minY - scale * height;
        topEdge = topEdge < 0.0f ? 0.0f : topEdge;

        int bottomEdge = minY + scale * height;
        bottomEdge = bottomEdge > frame.rows - 1 ? frame.rows - 1 : bottomEdge;

        //debug: draw rectangle on the raw image
        cv::rectangle(frameBak, cv::Point(leftEdge, topEdge), cv::Point(rightEdge, bottomEdge), cv::Scalar(0, 255, 0));
        ////////////

        const int rectWidth = rightEdge - leftEdge;
        const int rectHeight = bottomEdge - topEdge;
        cv::Rect roi(leftEdge, topEdge, rectWidth, rectHeight);

        cv::Mat newImg = frame(roi).clone();

        if (doCanny)
        {
            CannyEdge(newImg);
        }

        if (doWaterSheding)
        {
			cv::Mat output;
            WaterShedEdge(newImg, output, i, roi);

			// count area of the interior:
			int interiorArea = GetAreaInMask(output);
			//count the area of counter
			int contourArea = GetAreaInMask(maskBak(roi));

			float ratio = (float)std::abs(interiorArea - contourArea) / (float)contourArea;
			
			if (ratio < AREA_RATIO_THRESH)
			{
				m_InFavorIdx[i]++;
			}

        }// doWaterSheding

        if (doGrabCut)
        {
            GrabCutEdge(newImg, center, leftEdge, topEdge, width, height);
        }//if (doGrabCut)
    }//for (int i = 0; i < numContours; i++)
}//Segmentation

//=======================================================================
bool BGSubtractorOptFlow::AcceptTrackedPoint(int i)
{
    bool ok = false;

    if (m_Status[i])
    {
        // if point has moved
        // city-block distance
        const int cityBlockDist = abs(m_OptFeatures[0][i].x - m_OptFeatures[1][i].x) +
                                  abs(m_OptFeatures[0][i].y - m_OptFeatures[1][i].y);

        ok = cityBlockDist > MOVE_THRESH;
    }

    return ok;
}//AcceptTrackedPoint

//=======================================================================
void BGSubtractorOptFlow::HandleTrackedPoints(cv::Mat &frame)
{
    // 1. remove those features that don't "move"
    int numFeatures = 0;
    for (int id = 0; id < m_OptFeatures[1].size(); id++)
    {
        // do we keep this point?
        if (AcceptTrackedPoint(id))
        {
            // keep this point in vector
            m_OptFeatures[0][numFeatures] = m_OptFeatures[0][id];
            m_OptFeatures[1][numFeatures] = m_OptFeatures[1][id];
            numFeatures++;
        }
    }

    m_OptFeatures[0].resize(numFeatures);
    m_OptFeatures[1].resize(numFeatures);

    //////////////
    //debug: draw the features on the image
    //////////////
    //for all tracked points
    for (int m = 0; m < m_OptFeatures[0].size(); m++)
    {
        // draw line and circle
        cv::circle(frame, m_OptFeatures[0][m], 3, cv::Scalar(255, 255, 0), -1);
        cv::line(frame, m_OptFeatures[0][m], m_OptFeatures[1][m], cv::Scalar(0, 0, 0));
        cv::circle(frame, m_OptFeatures[1][m], 3, cv::Scalar(0, 0, 255),-1);
    }//for m
    ////////////////////////////////////////////////////

    // Features To Survie

    const bool measureDist = true;

    int minScore;
    float signedDist;
    float dist;

    // 2. loop through each features, and check if the contour is near the feature.
    // if a contour is not near any of the feature, delete the contour
    int numContours = m_Contours.size();

    if (m_InFavorIdx.size() != numContours)
    {
        std::cout << "error" << std::endl;
        return;
    }

    for (int i = 0; i < numContours; i++)
    {
        int score = 0; // if the score is higher than threshold, keep the contour, otherwise delete it

        for (int m = 0; m < numFeatures; m++)
        {
            signedDist = cv::pointPolygonTest(m_Contours[i], m_OptFeatures[1][m], measureDist);
            dist = fabs(signedDist);

            if (dist < DEVIATION_THRESH)
            {
                score++;
            }
        }// for m

        const int contourSize = m_Contours[i].size();
        minScore = std::min( contourSize, numFeatures ) * SCORE_PERCENTAGE;
        if (score > minScore)
        {
            // add the contour into want-to-keep list
            m_InFavorIdx[i]++;
        }
    }// for numContours

}//HandleTrackedPoints
