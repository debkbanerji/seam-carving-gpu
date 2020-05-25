#include <bits/stdc++.h>
#include <iostream>
#include <cmath>

#include "SeamCarvingShrinker.h"

// TODO: Get based on GPU
#ifndef BLOCK_SIZE
# define BLOCK_SIZE 1024
#endif

#ifndef NUM_COLORS
# define NUM_COLORS 3
#endif

#ifndef IMG_VAL
# define IMG_VAL(arr, width, col, row, color) ((arr)[(NUM_COLORS * (width) * (row)) + ((col) * NUM_COLORS) + (color)])
#endif

#ifndef ENERGY_IMG_VAL
# define ENERGY_IMG_VAL(arr, width, col, row) ((arr)[((width) * (row)) + (col)])
#endif

using namespace cimg_library;
using namespace std;

inline float getPixelColorsEuclideanDifference(float * workingArray, int originalWidth,
  int i1, int j1, int i2, int j2)
{
    float sumSquare = 0.0;

    for (int color = 0; color < NUM_COLORS; color++) {
        float diff =
          abs(IMG_VAL(workingArray, originalWidth, i1, j1, color)
            - IMG_VAL(workingArray, originalWidth, i2, j2, color));
        sumSquare += diff;
    }

    return sqrt(sumSquare);
}

float getPixelEnergy(float * workingArray, int i, int j, int originalWidth, int originalHeight, int currentWidth,
  int currentHeight)
{
    float totalEnergy = 0.0;

    if (i > 0) {
        totalEnergy += getPixelColorsEuclideanDifference(workingArray, originalWidth, i, j, i - 1, j);
    }
    if (i < originalWidth - 1) {
        totalEnergy += getPixelColorsEuclideanDifference(workingArray, originalWidth, i, j, i + 1, j);
    }
    if (j > 0) {
        totalEnergy += getPixelColorsEuclideanDifference(workingArray, originalWidth, i, j, i, j - 1);
    }
    if (j < originalHeight - 1) {
        totalEnergy += getPixelColorsEuclideanDifference(workingArray, originalWidth,
            i, j, i, j + 1);
    }

    // forward energy stuff
    // we're going to calculate forward energy in both directions because figuring out which
    // direction to calculate forward energy for would require a large rewrite, and this
    // is an easy improvement
    if (i > 0 && i < originalWidth - 1) {
        totalEnergy += getPixelColorsEuclideanDifference(workingArray, originalWidth, i - 1, j, i + 1, j);
    }
    if (j > 0 && j < originalHeight - 1) {
        totalEnergy += getPixelColorsEuclideanDifference(workingArray, originalWidth, i, j - 1, i, j + 1);
    }

    return totalEnergy;
}

float * getWorkingArray(CImg<float> image)
{
    auto width  = image.width();
    auto height = image.height();

    // assume image depth of 1

    float * result;

    cudaMallocManaged(&result, width * height * NUM_COLORS * sizeof(float)); // RGB image

    // old fashioned copy - not using Cuda since this isn't a bottleneck
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            for (int color = 0; color < NUM_COLORS; color++) {
                IMG_VAL(result, width, i, j, color) = image(i, j, 0, color);
            }
        }
    }

    return result;
}

float * getEnergyMap(float * workingArray, int originalWidth, int originalHeight, int currentWidth,
  int currentHeight)
{
    float * result;

    cudaMallocManaged(&result, originalWidth * originalHeight * sizeof(float)); // RGB image

    // old fashioned copy - not using Cuda since this isn't a bottleneck
    for (int i = 0; i < originalWidth; i++) {
        for (int j = 0; j < originalHeight; j++) {
            ENERGY_IMG_VAL(result, originalWidth, i, j) = getPixelEnergy(
                workingArray, i, j, originalWidth, originalHeight, currentWidth, currentHeight);
        }
    }

    return result;
}

CImg<float> getOutputImage(float * workingArray, int originalWidth, int originalHeight, int newWidth, int newHeight) // also cleans up input array
{
    CImg<float> result(newWidth, newHeight, 1, NUM_COLORS);

    for (int i = 0; i < newWidth; i++) {
        for (int j = 0; j < newHeight; j++) {
            for (int color = 0; color < NUM_COLORS; color++) {
                result(i, j, 0, color) = IMG_VAL(workingArray, originalWidth, i, j, color);
            }
        }
    }

    return result;
}

CImg<float> getOutputEnergyMapImage(float * workingenergyMapArray, int originalWidth, int newWidth, int newHeight) // also cleans up input array
{
    CImg<float> result(newWidth, newHeight, 1, NUM_COLORS);

    for (int i = 0; i < newWidth; i++) {
        for (int j = 0; j < newHeight; j++) {
            result(i, j, 0, 0) = ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, i, j);
        }
    }

    return result;
}

__global__
void calculateVerticalCumulativeEnergyTableKernel(
    float * workingenergyMapArray,
    bool * hasFinishedCalculationTable,
    float * cumulativeEnergyTable,
    int originalWidth,
    int currentWidth, int currentHeight
)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int row = 0; row < currentHeight; row++) {
        // iterate over every column this kernel is responsible for
        for (int col = index; col < currentWidth; col += stride) {
            // calculate the minimum cumulative energy for this index;
            float energy = ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, col, row);
            if (row > 0) {
                float minPreviousEnergy  = ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, col, row - 1); // guaranteed to be done
                int numSideColumns       = 0;
                int sideColumnIndices[2] = { 0, 0 }; // placeholders
                if (col > 0) {
                    sideColumnIndices[numSideColumns] = col - 1;
                    numSideColumns++;
                }
                if (col < currentWidth - 1) {
                    sideColumnIndices[numSideColumns] = col + 1;
                    numSideColumns++;
                }
                bool dependenciesResolved = false;
                // keep checking to see if dependencies (adjacent columns) have been resolved
                while (!dependenciesResolved) {
                    bool newDependenciesResolved = true;
                    for (int i = 0; i < numSideColumns; i++) {
                        newDependenciesResolved = newDependenciesResolved &&
                          ENERGY_IMG_VAL(hasFinishedCalculationTable, originalWidth, sideColumnIndices[i], row - 1);
                    }

                    dependenciesResolved = newDependenciesResolved;
                }

                // dependencies have been resolved - we can calculate the value now;
                for (int i = 0; i < numSideColumns; i++) {
                    minPreviousEnergy =
                      min(minPreviousEnergy,
                        ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, sideColumnIndices[i], row - 1));
                }

                energy += minPreviousEnergy;
            }
            ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, col, row)       = energy;
            ENERGY_IMG_VAL(hasFinishedCalculationTable, originalWidth, col, row) = true;
        }
    }
} // calculateVerticalCumulativeEnergyTableKernel

__global__
void initializeHasFinishedCalculationTableKernel(
    bool * hasFinishedCalculationTable,
    int    hasFinishedCalculationTableSize
)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < hasFinishedCalculationTableSize; i += stride) hasFinishedCalculationTable[i] = false;
}

int * getVerticalSeamColumnIndices(
    float * workingenergyMapArray,
    int originalWidth, int originalHeight,
    int currentWidth, int currentHeight)
{
    float * cumulativeEnergyTable;

    // TODO: Optimize to only use currentWidth and currentHeight
    cudaMallocManaged(&cumulativeEnergyTable, originalWidth * originalHeight * sizeof(float));


    bool * hasFinishedCalculationTable;
    // TODO: Optimize to only use currentWidth and currentHeight
    cudaMallocManaged(&hasFinishedCalculationTable, originalWidth * originalHeight * sizeof(bool));


    int blockSize = BLOCK_SIZE;
    int numBlocks = 1; // Don't split across multiple blocks to avoid memory access overhead

    initializeHasFinishedCalculationTableKernel << < numBlocks, blockSize >> > (
        hasFinishedCalculationTable,
        currentWidth * currentHeight
        );

    cudaDeviceSynchronize();


    calculateVerticalCumulativeEnergyTableKernel << < numBlocks, blockSize >> > (
        workingenergyMapArray,
        hasFinishedCalculationTable,
        cumulativeEnergyTable,
        originalWidth,
        currentWidth, currentHeight
        );

    cudaDeviceSynchronize();

    cudaFree(hasFinishedCalculationTable);

    int * result;

    cudaMallocManaged(&result, currentHeight * sizeof(int));


    int mRow = currentHeight - 1;
    int mCol = 0;
    for (int i = 0; i < currentWidth; i++) {
        if (ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, i, mRow)
          < ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, mCol, mRow))
        {
            mCol = i;
        }
    }
    result[mRow] = mCol;
    while (mRow > 0) {
        mRow--;

        int lCol = mCol - 1;
        int rCol = mCol + 1;
        if (lCol > 0 &&
          (ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, lCol, mRow)
          < ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, mCol, mRow)))
        {
            mCol = lCol;
        }
        if (rCol < originalWidth &&
          (ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, rCol, mRow)
          < ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, mCol, mRow)))
        {
            mCol = rCol;
        }
        result[mRow] = mCol;
    }

    cudaFree(cumulativeEnergyTable);

    return result;
} // getVerticalSeamColumnIndices

__global__
void removeVerticalSeamKernel(
    float * workingArray,
    float * workingenergyMapArray,
    int * seamColumnIndices,
    int originalWidth,
    int currentWidth, int currentHeight
)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int row = index; row < currentHeight; row += stride) {
        // remove seam for this row
        int targetColumn = seamColumnIndices[row];
        for (int i = targetColumn; i < currentWidth - 1; i++) {
            // shift over image pixel
            for (int color = 0; color < NUM_COLORS; color++) {
                IMG_VAL(workingArray, originalWidth, i, row, color) =
                  IMG_VAL(workingArray, originalWidth, i + 1, row, color);
            }

            // shift over energy map pixel
            ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, i, row) =
              ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, i + 1, row);

            // energy map is now invalid - it needs to be recalculated along the seam
            // do that outside the kernel to avoid race conditions
        }
    }
}

void removeVerticalSeam(
    float * workingArray,
    float * workingenergyMapArray,
    int * seamColumnIndices,
    int originalWidth, int originalHeight,
    int currentWidth, int currentHeight)
{
    int blockSize = BLOCK_SIZE;
    int numBlocks = 1; // Don't split across multiple blocks to avoid memory access overhead

    removeVerticalSeamKernel << < numBlocks, blockSize >> > (
        workingArray,
        workingenergyMapArray,
        seamColumnIndices,
        originalWidth,
        currentWidth, currentHeight
        );

    cudaDeviceSynchronize();

    // energy map is now invalid - it needs to be recalculated along the seam
    // recalculate energy map values along seam
    currentWidth -= 1;
    for (int row = 0; row < currentHeight; row++) {
        int removedColumn = seamColumnIndices[row];
        if (removedColumn > 0) {
            ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, removedColumn - 1, row) = getPixelEnergy(
                workingArray, removedColumn - 1, row, originalWidth, originalHeight, currentWidth, currentHeight);
        }
        if (removedColumn < currentWidth - 1) {
            ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, removedColumn, row) = getPixelEnergy(
                workingArray, removedColumn, row, originalWidth, originalHeight, currentWidth, currentHeight);
        }
    }
}

void identifyAndRemoveVerticalSeam(
    float * workingArray,
    float * workingenergyMapArray,
    int originalWidth, int originalHeight,
    int currentWidth, int currentHeight)
{
    int * verticalSeamColumnIndices = getVerticalSeamColumnIndices(workingenergyMapArray,
        originalWidth, originalHeight, currentWidth, currentHeight);

    removeVerticalSeam(workingArray, workingenergyMapArray, verticalSeamColumnIndices,
      originalWidth, originalHeight,
      currentWidth, currentHeight);

    cudaFree(verticalSeamColumnIndices);
}

__global__
void calculateHorizontalCumulativeEnergyTableKernel(
    float * workingenergyMapArray,
    bool * hasFinishedCalculationTable,
    float * cumulativeEnergyTable,
    int originalWidth,
    int currentWidth, int currentHeight
)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int col = 0; col < currentWidth; col++) {
        // iterate over every row this kernel is responsible for
        for (int row = index; row < currentHeight; row += stride) {
            // calculate the minimum cumulative energy for this index;
            float energy = ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, col, row);
            if (col > 0) {
                float minPreviousEnergy     = ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, col - 1, row); // guaranteed to be done
                int numDependencyRows       = 0;
                int dependencyRowIndices[2] = { 0, 0 }; // placeholders
                if (row > 0) {
                    dependencyRowIndices[numDependencyRows] = row - 1;
                    numDependencyRows++;
                }
                if (row < currentHeight - 1) {
                    dependencyRowIndices[numDependencyRows] = row + 1;
                    numDependencyRows++;
                }
                bool dependenciesResolved = false;
                // keep checking to see if dependencies (adjacent rows) have been resolved
                while (!dependenciesResolved) {
                    bool newDependenciesResolved = true;
                    for (int i = 0; i < numDependencyRows; i++) {
                        newDependenciesResolved = newDependenciesResolved &&
                          ENERGY_IMG_VAL(hasFinishedCalculationTable, originalWidth, col - 1, dependencyRowIndices[i]);
                    }

                    dependenciesResolved = newDependenciesResolved;
                }

                // dependencies have been resolved - we can calculate the value now;
                for (int i = 0; i < numDependencyRows; i++) {
                    minPreviousEnergy =
                      min(minPreviousEnergy,
                        ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, col - 1, dependencyRowIndices[i]));
                }

                energy += minPreviousEnergy;
            }
            ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, col, row)       = energy;
            ENERGY_IMG_VAL(hasFinishedCalculationTable, originalWidth, col, row) = true;
        }
    }
} // calculateVerticalCumulativeEnergyTableKernel

int * getHorizontalSeamColumnIndices(
    float * workingenergyMapArray,
    int originalWidth, int originalHeight,
    int currentWidth, int currentHeight)
{
    float * cumulativeEnergyTable;

    // TODO: Optimize to only use currentWidth and currentHeight
    cudaMallocManaged(&cumulativeEnergyTable, originalWidth * originalHeight * sizeof(float));


    bool * hasFinishedCalculationTable;
    // TODO: Optimize to only use currentWidth and currentHeight
    cudaMallocManaged(&hasFinishedCalculationTable, originalWidth * originalHeight * sizeof(bool));


    int blockSize = BLOCK_SIZE;
    int numBlocks = 1; // Don't split across multiple blocks to avoid memory access overhead

    initializeHasFinishedCalculationTableKernel << < numBlocks, blockSize >> > (
        hasFinishedCalculationTable,
        currentWidth * currentHeight
        );

    cudaDeviceSynchronize();


    calculateHorizontalCumulativeEnergyTableKernel << < numBlocks, blockSize >> > (
        workingenergyMapArray,
        hasFinishedCalculationTable,
        cumulativeEnergyTable,
        originalWidth,
        currentWidth, currentHeight
        );

    cudaDeviceSynchronize();


    cudaFree(hasFinishedCalculationTable);

    int * result;

    cudaMallocManaged(&result, currentWidth * sizeof(int));

    int mRow = 0;
    int mCol = currentWidth - 1;
    for (int i = 0; i < currentHeight; i++) {
        if (ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, mCol, i)
          < ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, mCol, mRow))
        {
            mRow = i;
        }
    }
    result[mCol] = mRow;
    while (mCol > 0) {
        mCol--;

        int dRow = mRow - 1;
        int uRow = mRow + 1;
        if (dRow > 0 &&
          (ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, mCol, dRow)
          < ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, mCol, mRow)))
        {
            mRow = dRow;
        }
        if (uRow < originalHeight &&
          (ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, mCol, uRow)
          < ENERGY_IMG_VAL(cumulativeEnergyTable, originalWidth, mCol, mRow)))
        {
            mRow = uRow;
        }
        result[mCol] = mRow;
    }

    cudaFree(cumulativeEnergyTable);

    return result;
} // getVerticalSeamColumnIndices

__global__
void removeHorizontalSeamKernel(
    float * workingArray,
    float * workingenergyMapArray,
    int * seamRowIndices,
    int originalWidth,
    int currentWidth, int currentHeight
)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int col = index; col < currentWidth; col += stride) {
        // remove seam for this row
        int targetRow = seamRowIndices[col];

        for (int i = targetRow; i < currentHeight - 1; i++) {
            // shift over image pixel
            for (int color = 0; color < NUM_COLORS; color++) {
                IMG_VAL(workingArray, originalWidth, col, i, color) =
                  IMG_VAL(workingArray, originalWidth, col, i + 1, color);
            }

            // shift over energy map pixel
            ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, col, i) =
              ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, col, i + 1);

            // energy map is now invalid - it needs to be recalculated along the seam
            // do that outside the kernel to avoid race conditions
        }
    }
}

void removeHorizontalSeam(
    float * workingArray,
    float * workingenergyMapArray,
    int * seamRowIndices,
    int originalWidth, int originalHeight,
    int currentWidth, int currentHeight)
{
    int blockSize = BLOCK_SIZE;
    int numBlocks = 1; // Don't split across multiple blocks to avoid memory access overhead

    removeHorizontalSeamKernel << < numBlocks, blockSize >> > (
        workingArray,
        workingenergyMapArray,
        seamRowIndices,
        originalWidth,
        currentWidth, currentHeight
        );

    cudaDeviceSynchronize();

    // energy map is now invalid - it needs to be recalculated along the seam
    // recalculate energy map values along seam
    currentHeight -= 1;
    for (int col = 0; col < currentWidth; col++) {
        int removedRow = seamRowIndices[col];
        if (removedRow > 0) {
            ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, col, removedRow - 1) = getPixelEnergy(
                workingArray, col, removedRow - 1, originalWidth, originalHeight, currentWidth, currentHeight);
        }
        if (removedRow < currentHeight - 1) {
            ENERGY_IMG_VAL(workingenergyMapArray, originalWidth, col, removedRow) = getPixelEnergy(
                workingArray, col, removedRow, originalWidth, originalHeight, currentWidth, currentHeight);
        }
    }
}

void identifyAndRemoveHorizontalSeam(
    float * workingArray,
    float * workingenergyMapArray,
    int originalWidth, int originalHeight,
    int currentWidth, int currentHeight)
{
    int * seamRowIndices = getHorizontalSeamColumnIndices(workingenergyMapArray,
        originalWidth, originalHeight, currentWidth, currentHeight);

    removeHorizontalSeam(workingArray, workingenergyMapArray, seamRowIndices,
      originalWidth, originalHeight,
      currentWidth, currentHeight);

    cudaFree(seamRowIndices);
}
