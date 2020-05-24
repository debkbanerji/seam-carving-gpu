#include <bits/stdc++.h>
#include <iostream>
#include <jpeglib.h>
#include <png.h>
#define cimg_use_png
#define cimg_use_jpeg
#include <CImg.h>
#include <chrono>

#include "SeamCarving/SeamCarvingManager.h"


using namespace cimg_library;
using namespace std;

const string TEST_CASES_DIR = "test_cases";

int main(int argc, char ** argv)
{
    cout << "Running Test" << endl;


    if (argc < 4) {
        cout << "Usage: ./main path/to/input/file <new_width> <new_height>\n";
        return 1;
    }

    CImg<float> image(argv[1]);

    auto width  = image.width();
    auto height = image.height();

    int newWidth  = stoi(argv[2]);
    int newHeight = stoi(argv[3]);

    cout << "Loaded image - resizing from " << width << "x" << height
         << " to " << newWidth << "x" << newHeight << "...\n\n";

    if (newWidth > width || newHeight > height) {
        cout << "Increasing size in either direction not yet supported " << '\n';
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    CImg<float> result = seamCarvingShrink(image, newWidth, newHeight);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto durationMicroSeconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto durationMS  = durationMicroSeconds / 1000;
    auto durationSec = durationMS / 1000;

    cout << "Finished carving image in " << durationSec / 60 << " minutes, "
         << durationSec % 60 << " seconds, "
         << durationMS % 1000 << " milliseconds\n";

    cout << "Displaying image\n";


    CImgDisplay main_disp(result);


    std::cin.ignore();

    return 0;
} // main
