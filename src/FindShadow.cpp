/**
 * @file FindShadow.cpp
 * The goal of the code in this file is to provide the implementation of
 * findShadow() function. Since it is called concurrently by multiple threads
 * mutex are introduced to sincronyze access to shared resources. In particular,
 * these are the standard output and the vector of points that collects the shadow
 * pixels.
 *
 * @author Martini Davide
 * @version 1.0
 * @since 1.0
 *
 */

#include "FindShadow.h"

mutex sPointsMutex; // mutex used to protect access to shadowPoints
mutex printMutex; // mutex used to protect access to standard output

/**
* This function states if a connected component composed by pixels with equal l*, a* and b* is a shadow or not.
* In order to do this, it computes the border of each component. Then it looks for a border pixels with a* and b*
* equal as those of the component, but with higher lightnes value. If such a pixel is found, then the component is
* a shadow patch.
*
* @param imgL l* component of the filtered input image
* @param imgA a* component of the filtered input image
* @param imgB b* component of the filtered input image
* @param labValues l*, a*, b* values of all pixels in the provided components
* @param labCComps vector of connected components with equal l*, a*, b* values
* @param lStep step used to group l* components. See Main.cpp for more details
* @param aStep step used to group a* components. See Main.cpp for more details
* @param bStep step used to group b* components. See Main.cpp for more details
* @param shadowPoints vector used to store shadow pixels. It is shared by threads that call this function. See Main.cpp for more details
*/
void findShadow(Mat imgL, Mat imgA, Mat imgB, tuple<int, int, int> labValues,
    vector<vector<Point> > labCComps, int lStep, int aStep, int bStep, vector<Point>& shadowPoints){

  chrono::time_point<chrono::system_clock> Tstart, Tend;
  Tstart = chrono::system_clock::now();
  int pixelCounter = 0;

  for(int labCC = 0; labCC < labCComps.size(); labCC++){
    vector<Point> labCompPixels = labCComps[labCC];
    vector<Point> border;
    Mat supportBorder = Mat_<uchar>::zeros(imgL.size()); // used to avoid duplicates in border
    pixelCounter = pixelCounter + labCompPixels.size();

    // retrieve border pixels
    for(int w = 0; w < labCompPixels.size(); w++){
      Point p = labCompPixels[w];
      int i = p.x;
      int j = p.y;
      if(i != 0 && j != 0 && i != (imgL.rows - 1) && j != (imgL.cols - 1)){
        for(int x = -1; x <= 1; x++){
          for(int y = -1; y <= 1; y++){
            if(y == 0 && x == 0){
              continue;
            }
            else{
              if(supportBorder.at<uchar> (i+x,j+y) == 0){
                border.push_back(Point(i+x,j+y));
                supportBorder.at<uchar> (i+x,j+y) = 255;
              }
            }
          }
        }
      }
    }

    supportBorder.release();

    // look at border pixels:
    // 1) if there is a pixel with higher lightness than the component but equal chromatic values, it means
    //    that the component is a shadow that lies on a uniform background
    // 2) othrewise it is an object
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
    }

    if(isShadow){
      // write shadow pixels in the common container
      sPointsMutex.lock();
      for (int w = 0; w < labCompPixels.size(); w++){
        Point p = labCompPixels[w];
        shadowPoints.push_back(p);
      }
      sPointsMutex.unlock();
    }
  }

  Tend = chrono::system_clock::now();
  int Telapsed_seconds = chrono::duration_cast<std::chrono::milliseconds> (Tend-Tstart).count();

  printMutex.lock();
  cout << "Bin (" << get<0>(labValues) << ", " << get<1>(labValues) << ", " << get<2>(labValues)
  << ") -> totPixels: " << pixelCounter << ", totCC: " << labCComps.size() << ". Done in " << Telapsed_seconds << " ms" << endl;
  printMutex.unlock();
}
