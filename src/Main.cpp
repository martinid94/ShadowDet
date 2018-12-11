
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  Mat imgSrc, imgLAB;

  if (argc != 5){
    cout << endl;
    cout << "Run this executable by invoking it like this: " << endl;
    cout << "   ./ShadowDet ../data/flickr-4159721472_c55deb37d6_b.jpg 10 10 10" << endl;
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
    cout << "   ./ShadowDet ../data/flickr-4159721472_c55deb37d6_b.jpg 10 10 10" << endl;
    cout << endl;
    return 1;
  }

  imgSrc = imread(srcPath);

  if(imgSrc.empty()){
    cout << "Wrong argument! Can not open input image. Check for errors in the provided path" << endl;
    cout << "Run this executable by invoking it like this: " << endl;
    cout << "   ./ShadowDet ../data/flickr-4159721472_c55deb37d6_b.jpg 10 10 10" << endl;
    cout << endl;
    return 1;
  }

  // in this process we use CIE LAB color space
  cvtColor(imgSrc, imgLAB, COLOR_RGB2Lab);

  Mat channelLAB[3];
  split(imgLAB, channelLAB);
  Mat imgL = channelLAB[0];
  Mat imgA = channelLAB[1];
  Mat imgB = channelLAB[2];

  // write the files to see results
  imwrite("LAB.jpg", imgLAB);
  imwrite("L.jpg", imgL);
  imwrite("A.jpg", imgA);
  imwrite("B.jpg", imgB);

  //bilateral filter to reduce noise but preserve edges
  Mat lFiltered;
  bilateralFilter(imgL, lFiltered, 5, 80, 80);

  imgL = lFiltered;
  imwrite("L_BFiltered.jpg", lFiltered);
  lFiltered.release();

  Mat aFiltered;
  bilateralFilter(imgA, aFiltered, 5, 80, 80);
  //GaussianBlur(imgA, temp, Size(5, 5), 10, 10);
  imgA = aFiltered;
  imwrite("A_BFiltered.jpg", aFiltered);
  aFiltered.release();

  Mat bFiltered;
  bilateralFilter(imgB, bFiltered, 5, 80, 80);
  //GaussianBlur(imgB, temp, Size(5, 5), 10, 10);
  imgB = bFiltered;
  imwrite("B_BFiltered.jpg", bFiltered);
  bFiltered.release();

  // compute the average and the median value of the luminance component
  // these values are considered as the "background light" so they allow to
  // distinguish "probably shadow pixels" (PSP) from surely "not shadow pixels" (NSP)
  double avgLValue = mean(imgL)[0];

  cout << "average L value " << avgLValue << endl;

  Mat avgL(imgL.size(), CV_16SC1, avgLValue); //16 bits SIGNED per pixel -> short data type
  Mat imgL16;
  imgL.convertTo(imgL16, CV_16SC1); //16 bits SIGNED per pixel -> short data type

  // compute the normalized images subtracting background light
  // pixels greater than average/median light are positive in norm images
  // pixels smaller than average/median light are negative in norm images
  // that's why I used CV_16SC1
  Mat normAvgL(imgL16.size(), CV_16SC1);
  normAvgL = imgL16 - avgL;

  // release memory
  avgL.release();

  // create the masks
  Mat maskAvgL = Mat_<uchar>::zeros(imgL.size());

  // each negative pixel in norm images is a PSP, otherwise it is a NSP.
  // in the resulting masks, each PSP value is set to the one assumed in the luminance image
  // while each NSP remains set to 0
  int maskPixels = 0;
  for(int i = 0; i < imgL.rows; i++){
    for(int j = 0; j < imgL.cols; j++){
      if(normAvgL.at<short> (i,j) < 0){
        maskAvgL.at<uchar> (i,j) = 1 + imgL.at<uchar> (i, j);
        maskPixels++;
      }
    }
  }

  // write the files to see results
  imwrite("mask_step_one.jpg", maskAvgL);

  // now further computation to detect the shadow pixels (SP) from the PSP
  map<tuple<int, int, int> , vector<Point> > labMap;
  map<tuple<int, int, int>, vector<Point> >::iterator labIt;

  // fill the map:
  // key = a*, b* color bin
  // value = points of the mask in the bin
  for (int i = 0; i < maskAvgL.rows; i++){
    for (int j = 0; j < maskAvgL.cols; j++){
      if (maskAvgL.at<uchar> (i, j) != 0){
        Point p(i, j);
        int theB = ceil(imgB.at<uchar> (i, j) / bStep);
        int theA = ceil(imgA.at<uchar> (i, j) / aStep);
        int theL = ceil(imgL.at<uchar> (i, j) / lStep);
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

  cout << "labMap created. labMap size: " << labMap.size() << endl;

  for (labIt = labMap.begin(); labIt != labMap.end(); labIt++){
    tuple<int, int, int> labValues = labIt->first;
    vector<Point> labPixels = labIt->second;
    Mat labTemp = Mat_<uchar>::zeros(maskAvgL.size());

    // fill labTemp
    for (int w = 0; w < labPixels.size(); w++){
      Point p = labPixels[w];
      labTemp.at<uchar> (p.x, p.y) = 255;
    }

    // find connected pixels with same color component
    Mat labLabels;
    int labComp = connectedComponents(labTemp, labLabels);
    labTemp.release();

    cout << "current bin (" << get<0>(labValues) << ", " << get<1>(labValues) << ", " << get<2>(labValues) << ") -> totPixels: " << labPixels.size()
    << ", labComponents: " << labComp << endl;

    // for each component, retrieve its pixels and split them in terms of lightness
    for (int cc = 1; cc < labComp; cc++){
      vector<Point> labCompPixels; // store pixels in the current component

      // retrieve pixels
      for(int i = 0; i < labLabels.rows; i++){
        for(int j = 0; j < labLabels.cols; j++){
          if(labLabels.at<int> (i,j) == cc){
            labCompPixels.push_back(Point(i,j));
          }
        }
      }

      vector<Point> border;
      Mat supportBorder = Mat_<uchar>::zeros(labLabels.size());

      // retrieve border pixels
      for(int w = 0; w < labCompPixels.size(); w++){
        Point p = labCompPixels[w];
        int i = p.x;
        int j = p.y;
        if(i != 0 && j != 0 && i != (labLabels.rows - 1) && j != (labLabels.cols - 1)){
          for(int x = -1; x <= 1; x++){
            for(int y = -1; y <= 1; y++){
              if(y == 0 && x == 0){
                continue;
              }
              else{
                if(labLabels.at<int> (i+x,j+y) != cc && supportBorder.at<uchar> (i+x,j+y) == 0){
                  border.push_back(Point(i+x,j+y));
                  supportBorder.at<uchar> (i+x,j+y) = 255;
                }
              }
            }
          }
        }
      }

      supportBorder.release(); // only needed not to have double pixels in border

      // look at border lightness
      // if there is a pixel with same lightness as the component, it means that the component is part of a shadow that lies on a non-uniform background
      // othrewise it is an object
      bool isShadow = false;
      for (int w = 0; w < border.size(); w++){
        Point bp = border[w];
        int bpL = ceil(imgL.at<uchar>(bp.x, bp.y) / lStep);
        int bpA = ceil(imgA.at<uchar> (bp.x, bp.y) / aStep);
        int bpB = ceil(imgB.at<uchar> (bp.x, bp.y) / bStep);

        // bp is lighter than the component (or zero in the temporary mask) but has same color of the component
        if(bpL > get<0>(labValues) && get<0>(labValues) > 0 && bpA == get<1>(labValues) && bpB == get<2>(labValues)){
              isShadow = true;
              break;
        }
        // bp has the same lightness as the component but different color
        // --> shadow on eterogeneous background
        // --> gives problems for black objects
        //if(bpL == get<0>(labValues)){
        //  isShadow = true;
        //  break;
        //}
      }

      for (int w = 0; w < labCompPixels.size(); w++){
        Point p = labCompPixels[w];
        if(isShadow){
          maskAvgL.at<uchar> (p.x, p.y) = 255;
        }
        else{
          maskAvgL.at<uchar> (p.x, p.y) = 0;
        }
      }
    }
  }

  stringstream sstm;
  sstm << "mask_step_two_lStep" << lStep << "_aStep" << aStep << "_bStep" << bStep << ".jpg";
  string s = sstm.str();
  imwrite(s, maskAvgL);
  return 0;
}
