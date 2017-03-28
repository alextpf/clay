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

#if !defined VPROCESSOR
#define VPROCESSOR

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <SFML/Audio.hpp>

// The frame processor interface
class FrameProcessor {

  public:
    // processing method
    virtual void process(cv:: Mat &input, cv:: Mat &output)= 0;
};

class VideoProcessor {

public:

	// Constructor setting the default values
	VideoProcessor();

	// set the name of the video file
	bool setInput(std::string filename);

	// set the camera ID
	bool setInput(int id);

	// set the vector of input images
	bool setInput(const std::vector<std::string>& imgs);

	// set the output video file
	// by default the same parameters than input video will be used
	bool setOutput(
		const std::string &filename,
		int codec = 0,
		double framerate = 0.0,
		bool isColor = true);

	// set the output as a series of image files
	// extension must be ".jpg", ".bmp" ...
	bool setOutput(
		const std::string &filename, // filename prefix
		const std::string &ext, // image file extension
		int numberOfDigits = 3,   // number of digits
		int startIndex = 0);

	// set the callback function that will be called for each frame
	void setFrameProcessor(void(*frameProcessingCallback)(cv::Mat&, cv::Mat&));

	// set the instance of the class that implements the FrameProcessor interface
	void setFrameProcessor(FrameProcessor* frameProcessorPtr);

	// stop streaming at this frame number
	void stopAtFrameNo(long frame) {

		frameToStop = frame;
	}

	// process callback to be called
	void callProcess() {

		callIt = true;
	}

	// do not call process callback
	void dontCallProcess() {

		callIt = false;
	}

	// to display the processed frames
	void displayInput(std::string wn) {

		windowNameInput = wn;
		cv::namedWindow(windowNameInput);
	}

	// to display the processed frames
	void displayOutput(std::string wn) {

		windowNameOutput = wn;
		cv::namedWindow(windowNameOutput);
	}

	// do not display the processed frames
	void dontDisplay() {

		cv::destroyWindow(windowNameInput);
		cv::destroyWindow(windowNameOutput);
		windowNameInput.clear();
		windowNameOutput.clear();
	}

	// set a delay between each frame
	// 0 means wait at each frame
	// negative means no delay
	void setDelay(int d) {

		delay = d;
	}

	// a count is kept of the processed frames
	long getNumberOfProcessedFrames() {

		return fnumber;
	}

	// return the size of the video frame
	cv::Size getFrameSize();

	// return the frame number of the next frame
	long getFrameNumber();

	// return the position in ms
	double getPositionMS();

	// return the frame rate
	double getFrameRate();

	// return the number of frames in video
	long getTotalFrameCount();

	// get the codec of input video
	int getCodec(char codec[4]);

	// go to this frame number
	bool setFrameNumber(long pos);

	// go to this position
	bool setPositionMS(double pos);

	// go to this position expressed in fraction of total film length
	bool setRelativePosition(double pos);

	// Stop the processing
	void stopIt() {

		stop = true;
	}

	// Is the process stopped?
	bool isStopped() {

		return stop;
	}

	// Is a capture device opened?
	bool isOpened() {

		return capture.isOpened() || !images.empty();
	}

	void setInitPosX(int x)
	{
		initPosX = x;
	}

	void setInitPosY(int y)
	{
		initPosY = y;
	}

	void setOffsetX(int x)
	{
		offsetX = x;
	}

	void setOffsetY(int y)
	{
		offsetY = y;
	}

	// to grab (and process) the frames of the sequence
	void run();

  private:

      // the OpenCV video capture object
      cv::VideoCapture capture;
      // the callback function to be called
      // for the processing of each frame
      void (*process)(cv::Mat&, cv::Mat&);
      // the pointer to the class implementing
      // the FrameProcessor interface
      FrameProcessor *frameProcessor;
      // a bool to determine if the
      // process callback will be called
      bool callIt;
      // Input display window name
      std::string windowNameInput;
      // Output display window name
      std::string windowNameOutput;
      // delay between each frame processing
      int delay;
      // number of processed frames
      long fnumber;

      long totalFrame;

      // stop at this frame number
      long frameToStop;
      // to stop the processing
      bool stop;

      // vector of image filename to be used as input
      std::vector<std::string> images;
      // image vector iterator
      std::vector<std::string>::const_iterator itImg;

      // the OpenCV video writer object
      cv::VideoWriter writer;
      // output filename
      std::string outputFile;

      // current index for output images
      int currentIndex;
      // number of digits in output image filename
      int digits;
      // extension of output images
      std::string extension;

      int initPosX;
      int initPosY;
      int offsetX; //
      int offsetY; //
      cv::Mat tmpFrame; // tmp frame

      // to get the next frame
      // could be: video file; camera; vector of images
	  bool readNextFrame(cv::Mat& frame);

      // to write the output frame
      // could be: video file or images
	  void writeNextFrame(cv::Mat& frame);

};

#endif
