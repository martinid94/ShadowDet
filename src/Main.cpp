/**
 * @file Main.cpp
 *  The goal of the code in this file is to provide an effective strategy to segment
 *  a picture into shadow and non-shadow areas. In particular, it relies on OpenCV library
 *  to convert the input RGB image into the CIE L*a*b* (or Lab) color space and to retrieve
 *  connected components.
 *  The results are printed in an external file in order to easily analyze the output.
 *
 * @author Martini Davide
 * @version 1.0
 * @since 1.0
 *
 */

#include "FindShadow.h"

int main(int argc, char** argv){

  if (argc != 5){
    cout << endl;
    cout << "Run this executable by invoking it like this: " << endl;
    cout << "   ./ShadowDet ../data/flickr-4159721472_c55deb37d6_b.jpg 20 50 50" << endl;
    cout << endl;
    cout << "The first argument is the input image path." << endl;
    cout << "The second argument is the lStep parameter. It must be positive." << endl;
    cout << "The third argument is the aStep parameter. It must be positive." << endl;
    cout << "The fourth argument is the bStep parameter. It must be positive." << endl;
    cout << endl;
    return 1;
  }

  const string srcPath = argv[1];
  const int lStep = atoi(argv[2]);
  const int aStep = atoi(argv[3]);
  const int bStep = atoi(argv[4]);

  if(aStep <= 0 || bStep <=0 || lStep <= 0){
    cout << endl;
    cout << "Wrong argument! lStep, aStep and bStep must be positive." << endl;
    cout << "Run this executable by invoking it like this: " << endl;
    cout << "   ./ShadowDet ../data/flickr-4159721472_c55deb37d6_b.jpg 20 50 50" << endl;
    cout << endl;
    return 1;
  }

  chrono::time_point<chrono::system_clock> start, end;
  start = chrono::system_clock::now();

  Mat imgRGB;
  imgRGB = imread(srcPath);

  if(imgRGB.empty()){
    cout << "Wrong argument! Can not open input image. Check for errors in the provided path" << endl;
    cout << "Run this executable by invoking it like this: " << endl;
    cout << "   ./ShadowDet ../data/flickr-4159721472_c55deb37d6_b.jpg 50 50 50" << endl;
    cout << endl;
    return 1;
  }

  // in this process we use CIE LAB color space
  Mat imgLAB;
  cvtColor(imgRGB, imgLAB, COLOR_RGB2Lab);

  Mat channelLAB[3];
  split(imgLAB, channelLAB);
  Mat imgL = channelLAB[0];
  Mat imgA = channelLAB[1];
  Mat imgB = channelLAB[2];

  //bilateral filter to reduce noise but preserve edges
  Mat lFiltered;
  bilateralFilter(imgL, lFiltered, 5, 80, 80);
  imgL = lFiltered;
  lFiltered.release();

  Mat aFiltered;
  bilateralFilter(imgA, aFiltered, 5, 80, 80);
  imgA = aFiltered;
  aFiltered.release();

  Mat bFiltered;
  bilateralFilter(imgB, bFiltered, 5, 80, 80);
  imgB = bFiltered;
  bFiltered.release();

  // compute the mean and the standard deviation of the luminance component
  // these values are considered as the "background light" so they allow to
  // distinguish "probably shadow pixels" (PSP) from surely "not shadow pixels" (NSP)
  Mat imgLD, meanL, stdDevL;
  bool useSTD = true;
  imgL.convertTo(imgLD, CV_64FC1);
  meanStdDev(imgLD, meanL, stdDevL);
  if(stdDevL.at<double> (0, 0) <  (double) 255 / 6){
    useSTD = false;
  }

  cout << "Mean lightness value: " << meanL.at<double> (0, 0) << ", standard deviation: " << stdDevL.at<double> (0, 0)
       << " useSTD: " << useSTD << endl;

  // create the mask
  Mat maskAvgL = Mat_<uchar>::zeros(imgL.size());

  // depending on the standard deviation on imgL, each pixel with lightness component less than meanL - stdDevL/3
  // or simply meanL is a PSP, otherwise it is a NSP.
  // in the resulting masks, each PSP value is set to the one assumed in the luminance image plus 1
  // while each NSP remains set to 0
  int maskPixels = 0;
  for(int i = 0; i < maskAvgL.rows; i++){
    for(int j = 0; j < maskAvgL.cols; j++){
      if(useSTD){
        if(imgLD.at<double> (i, j) < (meanL.at<double> (0, 0) - (stdDevL.at<double> (0, 0) / 3))){
          maskAvgL.at<uchar> (i, j) = 1 + imgL.at<uchar> (i, j);
          maskPixels++;
        }
        else{
          maskAvgL.at<uchar> (i, j) = 0;
        }
      }
      else{
        if(imgLD.at<double> (i, j) < meanL.at<double> (0, 0)){
          maskAvgL.at<uchar> (i, j) = 1 + imgL.at<uchar> (i, j);
          maskPixels++;
        }
        else{
          maskAvgL.at<uchar> (i, j) = 0;
        }
      }
    }
  }

  // write the files to see results
  imwrite("../results/mask_step_one.jpg", maskAvgL);

  // now further computation to detect the shadow pixels (SP) from the PSP
  map<tuple<int, int, int> , vector<Point> > labMap;
  map<tuple<int, int, int>, vector<Point> >::iterator labIt;

  // fill  labMap:
  // key = (l*, a*, b*) pixel component. It represents a color bin
  // value = points of the mask in the bin
  for (int i = 0; i < maskAvgL.rows; i++){
    for (int j = 0; j < maskAvgL.cols; j++){
      if (maskAvgL.at<uchar> (i, j) != 0){
        Point p(i, j);
        int theL = ceil(imgL.at<uchar> (i, j) / lStep);
        int theA = ceil(imgA.at<uchar> (i, j) / aStep);
        int theB = ceil(imgB.at<uchar> (i, j) / bStep);
        tuple<int, int, int> key = make_tuple(theL, theA, theB);

        labIt = labMap.find(key);
        if(labIt == labMap.end()){
          vector<Point> v;
          v.push_back(p);
          labMap.insert(pair<tuple<int, int, int>, vector<Point> >(key, v));
        }
        else{
          labMap[key].push_back(p);
        }
      }
    }
  }

  int labMapPixels = 0;
  for (labIt = labMap.begin(); labIt != labMap.end(); labIt++){
    vector<Point> labPixels = labIt->second;
    labMapPixels = labMapPixels + labPixels.size();
  }

  if(maskPixels == labMapPixels){
    cout << "labMap succesfully created. Entries in labMap: " << labMap.size() << endl;
  }

  // use thread pool to analyze each (key, value) pair in labMap
  const int poolSize = thread::hardware_concurrency();
  cout << "Max threads concurrent: " << poolSize << endl;
  vector<thread> threads;
  vector<Point> shadowPoints;
  labIt = labMap.begin();

  while(labIt != labMap.end()){
    if(threads.size() < poolSize){
      // lauch thread form the pool. See FindShadow.cpp for more information
      threads.push_back(thread(findShadow, imgL, imgA, imgB, labIt->first, labIt->second, lStep, aStep, bStep, ref(shadowPoints)));
      labIt++; // move to the next (key, value) pair in labMap
    }
    else{
      // wait for threads to finish their computation
      for(int t = 0; t < threads.size(); t++){
        threads[t].join();
      }
      // free the pool
      threads.erase(threads.begin(), threads.end());
    }
  }

  // wait for active threads remaining
  for(int t = 0; t < threads.size(); t++){
    threads[t].join();
  }
  // free the pool
  threads.erase(threads.begin(), threads.end());

  // write the final result
  Mat maskFinal = Mat_<uchar>::zeros(maskAvgL.size());

  for (int i = 0; i < shadowPoints.size(); i++){
    Point p = shadowPoints[i];
    maskFinal.at<uchar> (p.x, p.y) = 255;
  }

  stringstream sstm;
  sstm << "../results/mask_step_two_lStep" << lStep << "_aStep" << aStep << "_bStep" << bStep << ".jpg";
  string s = sstm.str();
  imwrite(s, maskFinal);

  // provide information to the user
  end = chrono::system_clock::now();
  int elapsed_seconds = chrono::duration_cast<std::chrono::milliseconds> (end-start).count();
  time_t end_time = chrono::system_clock::to_time_t(end);

  cout << "Finished computation at " << ctime(&end_time) << " Elapsed time: " << elapsed_seconds << " ms" << endl;

  return 0;
}
