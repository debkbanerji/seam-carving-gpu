#include <bits/stdc++.h>
#include <iostream>
#include <CImg.h>

#include "SeamCarvingShrinker.h"

using namespace cimg_library;


CImg<float> seamCarvingShrink(CImg<float> inputImage, int newWidth, int newHeight)
{
    int inputWidth  = inputImage.width();
    int inputHeight = inputImage.height();

    float * workingArray = getWorkingArray(inputImage);
    float * workingenergyMapArray = getEnergyMap(workingArray, inputWidth, inputHeight, inputWidth, inputHeight);

    float totalWidthShrinks  = inputWidth - newWidth;
    float totalHeightShrinks = inputHeight - newHeight;

    int currentWidth  = inputWidth;
    int currentHeight = inputHeight;

    while (currentWidth > newWidth || currentHeight > newHeight) {
        if ((currentWidth - newWidth) / totalWidthShrinks >
          (currentHeight - newHeight) / totalHeightShrinks)
        {
            identifyAndRemoveVerticalSeam(
                workingArray,
                workingenergyMapArray,
                inputWidth, inputHeight,
                currentWidth, currentHeight);
            currentWidth -= 1;
        } else {
            identifyAndRemoveHorizontalSeam(
                workingArray,
                workingenergyMapArray,
                inputWidth, inputHeight,
                currentWidth, currentHeight);
            currentHeight -= 1;
        }
    }

    CImg<float> outputImage =
      getOutputImage(workingArray, inputWidth, inputHeight, newWidth, newHeight);

    // CImg<float> outputEnergyMapImage =
    //   getOutputEnergyMapImage(workingenergyMapArray, inputWidth, newWidth, newHeight);


    cudaFree(workingArray);
    cudaFree(workingenergyMapArray);

    // CImgDisplay main_disp(outputEnergyMapImage);
    // std::cin.ignore();


    return outputImage;
} // seamCarvingShrink
