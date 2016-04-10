#include "facelab/homomorphic.hpp"

#include "local_laplacian.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <sstream>
#include <numeric>

static void bilateral(const cv::Mat &input, cv::Mat &output, int spatialRad, int colorRad, int iter)
{
    std::vector<cv::Mat> buffer { cv::Mat(), output };
    int j = 1;
    for(int i = 0; i < iter; i++, j = 1-j)
    {
        cv::bilateralFilter((i == 0) ? input : buffer[1-j], buffer[j], 0, spatialRad, colorRad);
    }
    output = buffer[j];
}

const char *version = "v0.1";

const char *keys =
{
    "{ input     |       | input filename                            }"
    "{ output    |       | output filename                           }"

    // Local Laplacian Filtering:
    "{ levels    |   3   | number of pyramid levels                  }"
    "{ sigma     |  0.3  | threshold distinguishing details from edges. Smaller values limit the manipulation to smaller-amplitude variations }"
    "{ alpha     |  2.0  | controls how details are modified: 0<a<1 amplifies detail, while a>1 attenuates it.     }"
    "{ beta      |  1.0  | intensity range: beta > 1.0 performs expansion, while beta < 1.0 performs compression.  }"

    "{ threads   | false | use worker threads when possible          }"
    "{ verbose   | false | print verbose diagnostics                 }"
    "{ build     | false | print the OpenCV build information        }"
    "{ help      | false | print help message                        }"
    "{ version   | false | print the application version             }"
};

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    
    if(argc < 2 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    else if(parser.has("build"))
    {
        std::cout << cv::getBuildInformation() << std::endl;
        return 0;
    }
    else if(parser.has("version"))
    {
        std::cout << argv[0] << " v" << version << std::endl;
        return 0;
    }

    std::string sInput = parser.get<std::string>("input");
    if(sInput.empty())
    {
        std::cerr << "Must specify input filename" << std::endl;
        return 1;
    }

    std::string sOutput = parser.get<std::string>("output");
    if(sOutput.empty())
    {
        std::cerr << "Msut specify output filename" << std::endl;
        return 1;
    }

    cv::Mat input = cv::imread(sInput);
    if(input.empty())
    {
        std::cerr << "Unable to read input file " << sInput << std::endl;
        return 1;
    }

    cv::Mat output;
    
    if(1)
    { // bilateral smoothing 
        cv::Mat smooth;
        bilateral(input, smooth, 10, 10, 2);
        
        const int order = 4;
        const float cutoff = 0.48f;
        const float boost= 4.0f;
        const float beta = 1.0f / boost;
        const float alpha = 1.0f - beta;

        cv::Mat hef;
        homomorphic(smooth, hef, cutoff, order, alpha, beta, {0.02f, 0.98f}, {});
        
        assert(hef.type() == CV_8UC3);
        
        // smoothing
        bilateral(hef, output, 10, 10, 2);
    }

    { // local laplacian filter:
        const double kSigmaR = parser.get<double>("sigma"); // 0.3
        const double kAlpha = parser.get<double>("alpha");  // 2.0
        const double kBeta = parser.get<double>("beta");    // 1.0
        const int kLevels = parser.get<int>("levels");      // 3
        
        output.convertTo(output, CV_64F, 1 / 255.0);

        std::cout << "Input image: " << sInput << " Size: " << input.cols << " x " << input.rows << " Channels: " << input.channels() << std::endl;

        switch(input.channels())
        {
        case 1:
            output = LocalLaplacianFilter<double>(output, kAlpha, kBeta, kSigmaR, kLevels);
            break;
        case 3:
            output = LocalLaplacianFilter<cv::Vec3d>(output, kAlpha, kBeta, kSigmaR, kLevels);
            break;
        default:
            std::cerr << "Input image must have 1 or 3 channels." << std::endl;
            return 1;
        }

        output *= 255;
        output.convertTo(output, CV_8UC3);
    }
    
    if(1)
    { // mean shift
        int spatialRad = 10, colorRad = 10, maxPyrLevel = 3;
        cv::Mat ms;
        cv::pyrMeanShiftFiltering(output, ms, spatialRad, colorRad, maxPyrLevel);
        cv::swap(output, ms);
    }

    cv::imwrite(sOutput, output);

    return 0;
}
