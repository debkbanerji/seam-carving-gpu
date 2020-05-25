#include <bits/stdc++.h>
#include <iostream>
#include <CImg.h>

using namespace cimg_library;


float* getWorkingArray(CImg<float> image);

float * getEnergyMap(float * workingArray, int originalWidth, int originalHeight, int currentWidth, int currentHeight);

CImg<float> getOutputImage(float* image, int originalWidth, int originalHeight, int newWidth, int newHeight); // also cleans up input array

CImg<float> getOutputEnergyMapImage(float * workingenergyMapArray, int originalWidth, int newWidth, int newHeight); // also cleans up input array

void identifyAndRemoveVerticalSeam(
    float * workingArray,
    float * workingenergyMapArray,
    int originalWidth, int originalHeight,
    int currentWidth, int currentHeight);

void identifyAndRemoveHorizontalSeam(
    float * workingArray,
    float * workingenergyMapArray,
    int originalWidth, int originalHeight,
    int currentWidth, int currentHeight);
