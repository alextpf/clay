/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 10 of the cookbook:
   Computer Vision Programming using the OpenCV Library.
   by Robert Laganiere, Packt Publishing, 2011.

   This program is free software; permission is hereby granted to use, copy, modify,
   and distribute this source code, or portions thereof, for any purpose, without fee,
   subject to the restriction that the copyright notice may not be removed
   or altered from any source or altered source distribution.
   The software is released on an as-is basis and without any warranties of any kind.
   In particular, the software is not guaranteed to be fault-tolerant or free from failure.
   The author disclaims all warranties with regard to this software, any use,
   and any consequent failure, is purely the responsibility of the user.

   Copyright (C) 2010-2011 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include "videoprocessor.h"
#include "BGFGSegmentor.h"
#include "BGSubtractor.h"
#include "BGSubtractorOptFlow.h"

//=======================================================================
void Tracking()
{
	// Create video procesor instance
	VideoProcessor processor;

	// Create background/foreground segmentor
	BGSubtractorOptFlow segmentor;
	segmentor.setLearningRate(0.1f);

	//////////////////
	// variables
	//////////////////

	//char path[] = "../../foreground/BGSubtractorOptFlow/results/";
	char path[] = "results/";
	//char path[] = "";
	//char filename[] = "frisbee-new";//frisbee,clay2, trap1, trap2
	//char filename[] = "alex2";
	//char filename[] = "clay2";
	//char filename[] = "trap1";
	//char filename[] = "trap2";
	//char filename[] = "thrower1";
	//char filename[] = "thrower2";
	char filename[] = "baseball";
	
	char inputMode[] = "raw";//raw, mog, mog+morph for the input
	char outputMode[] = "final";//raw, mog, mog+morph, contours for the input
	//int num = 165; // frame number. frisbee: 165, clay2: 178, trap1: 270, trap2: 620
	//int num = 1211;//alex2
	//int num = 178;//clay2
	//int num = 270;//trap1
	//int num = 670;//trap2
	//int num = 1095;//thrower1
	//int num = 830;//thrower2
	int num = 120;//baseball
	
	int startFrame = 0;// frame number we want to start at

	/////////////////////
	// Input & Output
	/////////////////////
	// 0: imgs, 1: video, 2: webcam
	static int inputType = 0;

	// 0: imgs, 1: video, 2: no output written
	static int outputType = 0;

	/////////////////
	// Frame Rate:
	/////////////////
	float fps = 60.f;

	//float delay = 1000. / processor.getFrameRate();
	float delay = -1.0f;
	//float delay = 1000. / fps;

	/////////////////////////////////////////////////////
	// Input
	/////////////////////////////////////////////////////
	switch (inputType)
	{
	case 0:
	{
		/////////////////////////
		// input: images
		/////////////////////////
		std::vector<std::string> imgs;

		for (int i = 0; i < num; i++)
		{
			char buffer[100];
			sprintf_s(buffer, "%s%s/%s/%s_%03i.jpg", path, filename, inputMode, filename, i);

			std::string name = buffer;
			imgs.push_back(name);
		}

		processor.setInput(imgs);
		////////////////
	}
	break;
	case 1:
	{
		/////////////////////////
		// input: video
		/////////////////////////
		char buffer[100];
		sprintf_s(buffer, "%s%s.mp4", path, filename);

		std::string name = buffer;
		processor.setInput(name);
	}
	break;

	case 2:
	{
		/////////////////////////
		// input: webcam
		/////////////////////////
		processor.setInput(0);//webcam
	}
	break;

	default:
		break;
	}//switch (inputType)

	/////////////////////////////////////////////////////
	// Output
	/////////////////////////////////////////////////////
	switch (outputType)
	{
	case 0:
	{
		/////////////////////////
		// output: images
		/////////////////////////
		char buffer[100];
		sprintf_s(buffer, "%s%s/%s/%s_", path, filename, outputMode, filename);

		processor.setOutput(buffer, ".jpg");
	}
	break;

	case 1:
	{
		/////////////////////////
		// output: video
		/////////////////////////
		char buffer[100];
		sprintf_s(buffer, "%s%s/%s/%s.mp4", path, filename, outputMode, filename);

		processor.setOutput(buffer);
	}
	break;

	// case 2:// no output

	default:
		break;
	}//switch (outputType)

	// set frame processor
	processor.setFrameProcessor(&segmentor);

	// Declare a window to display the video
	processor.displayOutput("Extracted Foreground");

	// Declare a window to display the input
	processor.displayInput("Input");

	processor.setFrameNumber(startFrame);
	processor.setDelay(delay);

	// Start the process
	processor.run();

	cv::waitKey();
}// Tracking

//=======================================================================
void Processing()
{
	// Create instance
	VideoProcessor processor;

	//////////////////
	// book keeping
	//////////////////

	//char path[] = "../../foreground/BGSubtractorOptFlow/results/";
	char path[] = "results/";
	//char filename[] = "alex2";//frisbee,clay2, trap1, trap2
	//char filename[] = "frisbee-new";
	//char filename[] = "clay2";
	//char filename[] = "trap1";
	//char filename[] = "trap2";
	//char filename[] = "thrower1";
	//char filename[] = "thrower2";
	char filename[] = "baseball1";
	char inputMode[] = "raw";//raw, mog, mog+morph for the input
	char outputMode[] = "raw";//raw, mog, mog+morph, contours for the input
	//int num = 165; // frame number. frisbee: 165, clay2: 178, trap1: 270, trap2: 620
	//int num = 1211;//alex2
	//int num = 178;//clay2
	//int num = 270;//trap1
	int num = 670;//trap2
	//int startFrame = 0;// frame number we want to start at
	int startTime = 68 * 1000;//ms
	int stopFrame = 680;

	////////////////////////
	// Input & Output
	////////////////////////
	// 0: imgs, 1: video, 2: webcam
	static int inputType = 1;
	// 0: imgs, 1: video, 2: no output written
	static int outputType = 0;

	////////////////////////
	// FPS
	////////////////////////
	float fps = 60.f;
	//float delay = 1000. / processor.getFrameRate();
	//float delay = -1.0f;
	float delay = 1000. / fps;

	/////////////////////////////////////////////////////
	// Input
	/////////////////////////////////////////////////////
	switch (inputType)
	{
	case 0:
	{
		/////////////////////////
		// input: images
		/////////////////////////
		std::vector<std::string> imgs;

		for (int i = 0; i < num; i++)
		{
			char buffer[100];
			sprintf_s(buffer, "%s%s/%s/%s_%03i.jpg", path, filename, inputMode, filename, i);

			std::string name = buffer;
			imgs.push_back(name);
		}

		processor.setInput(imgs);
		////////////////
	}
	break;
	case 1:
	{
		/////////////////////////
		// input: video
		/////////////////////////
		char buffer[100];
		sprintf_s(buffer, "%s%s.mp4", path, filename);

		std::string name = buffer;
		processor.setInput(name);
	}
	break;

	case 2:
	{
		/////////////////////////
		// input: webcam
		/////////////////////////
		processor.setInput(0);//webcam
	}
	break;

	default:
		break;
	}

	/////////////////////////////////////////////////////
	// Output
	/////////////////////////////////////////////////////
	switch (outputType)
	{
	case 0:
	{
		/////////////////////////
		// output: images
		/////////////////////////
		char buffer[100];
		sprintf_s(buffer, "%s%s/%s/%s_", path, filename, outputMode, filename);

		processor.setOutput(buffer, ".jpg");
	}
	break;

	case 1:
	{
		/////////////////////////
		// output: video
		/////////////////////////
		char buffer[100];
		sprintf_s(buffer, "%s%s-new.avi", path, filename);

		std::string name = buffer;
		char codec[4];
		processor.setOutput(name, processor.getCodec(codec), 20.);

	}
	break;

	// case 2:// no output

	default:
		break;
	}

	// Get basic info about video file
	cv::Size size = processor.getFrameSize();
	std::cout << size.width << " " << size.height << std::endl;
	std::cout << processor.getFrameRate() << std::endl;
	std::cout << processor.getTotalFrameCount() << std::endl;
	std::cout << processor.getFrameNumber() << std::endl;
	std::cout << processor.getPositionMS() << std::endl;

	// No processing
	processor.dontCallProcess();

	/*char codec[4];
	processor.setOutput("../../clay2-.avi", processor.getCodec(codec), processor.getFrameRate());
	std::cout << "Codec: " << codec[0] << codec[1] << codec[2] << codec[3] << std::endl;
	*/
	// Position the stream at frame 300
	/*processor.setFrameNumber(300);
	processor.stopAtFrameNo(120);*/
	//
	
	//int endTime = 640 * 1000;//ms

	processor.setPositionMS(startTime);
	processor.stopAtFrameNo(stopFrame);

	// Declare a window to display the video
	processor.displayInput("Current Frame");
	processor.displayOutput("Output Frame");

	// Play the video at the original frame rate
	//processor.setDelay(1000. / processor.getFrameRate());
	processor.setDelay(delay);

	// Start the process
	processor.run();

	/*std::cout << processor.getFrameNumber() << std::endl;
	std::cout << processor.getPositionMS() << std::endl;
	*/
	cv::waitKey();
}//Processing

int main()
{
	// 1: MOG+Morph+Optical Flow + Segmentation
	// 2: video / image processing
    int method = 1;
    switch (method)
    {    
    case 1:
    {
		Tracking();        
    }
    break;
    case 2:
	{
		Processing();
    }
    break;
    }
}
