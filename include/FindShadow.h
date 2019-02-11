/**
 * @file FindShadow.h
 * This header file is included in FindShadow.cpp and Main.cpp. Further details can be found in those files
 *
 * @author Martini Davide
 * @version 1.0
 * @since 1.0
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <thread>
#include <chrono>
#include <ctime>
#include <mutex>

using namespace cv;
using namespace std;

#ifndef FS__H
#define FS__H

void findShadow(Mat imgL, Mat imgA, Mat imgB, tuple<int, int, int> labValues,
    vector<Point> labPixels, int lStep, int aStep, int bStep, vector<Point>& shadowPoints);
#endif
