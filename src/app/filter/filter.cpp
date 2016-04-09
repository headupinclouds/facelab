#include "facelab/homomorphic.hpp"

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
    {
        // smoothing
        cv::Mat smooth;
        bilateral(input, smooth, 10, 10, 2);
        
        const int order = 4;
        const float cutoff = 0.48;
        const float boost= 4.0;
        const float beta = 1.0 / boost;
        const float alpha = 1.0 - beta;

        cv::Mat hef;
        homomorphic(smooth, hef, cutoff, order, alpha, beta, {0.1, 0.90}, {});
        
        assert(hef.type() == CV_8UC3);
        
        // smoothing
        bilateral(hef, output, 10, 10, 2);
    }
    
    if(1)
    {
        int spatialRad = 10, colorRad = 10, maxPyrLevel = 3;
        cv::Mat ms;
        cv::pyrMeanShiftFiltering(output, ms, spatialRad, colorRad, maxPyrLevel);
        cv::swap(output, ms);
    }

    cv::imwrite(sOutput, output);

    return 0;
}
