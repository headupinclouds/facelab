// Copyright (c) 2016, David Hirvonen
// All rights reserved.

#include "facelab/FaceLandmarkMeshMapper.h"
#include "facelab/FaceLandmarker.h"

#include <iostream>
#include <sstream>
#include <numeric>

const char *version = "v0.1";

const char *keys =
{
    "{ input     |       | input filename                            }"
    "{ output    |       | output filename                           }"
    
    "{ width     | 256   | processing width                          }"
    "{ verbose   | false | verbose mode (w/ display)                 }"

    // Tracker file
    "{ regressor |       | face landmark regressor file              }"
    "{ detector  |       | face detector                             }"

    "{ model     |       | model file                                }"
    "{ mapping   |       | mapping file                              }"
    
    "{ threads   | false | use worker threads when possible          }"
    "{ verbose   | false | print verbose diagnostics                 }"
    "{ build     | false | print the OpenCV build information        }"
    "{ help      | false | print help message                        }"
    "{ version   | false | print the application version             }"
};

int main(int argc, char *argv[])
{
    for(int i = 0; i < argc; i++)
    {
        std::cout << argv[i] << std::endl;
    }
    
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

    bool verbose = parser.get<bool>("verbose");
    
    std::string sInput = parser.get<std::string>("input");
    if(sInput.empty())
    {
        std::cerr << "Must specify input filename" << std::endl;
        return 1;
    }

    std::string sOutput = parser.get<std::string>("output"); std::cout << sOutput << std::endl;
    if(sOutput.empty())
    {
        std::cerr << "Must specify output filename" << std::endl;
        return 1;
    }

    cv::Mat input = cv::imread(sInput);
    if(input.empty())
    {
        std::cerr << "Unable to read input file " << sInput << std::endl;
        return 1;
    }
    
    int width = parser.get<int>("width");
    cv::resize(input, input, {width, input.rows * width/input.cols}, cv::INTER_CUBIC);
    
    // ########## FACE MESH LANDMARKER #########
    std::string sModel = parser.get<std::string>("model");
    std::string sMapping = parser.get<std::string>("mapping");
    FaceLandmarkMeshMapper mapper(sModel, sMapping);

    // ######### LANDMARK ######################
    std::shared_ptr<FaceLandmarker> landmarker;
    std::string sDetector = parser.get<std::string>("detector");
    std::string sRegressor = parser.get<std::string>("regressor");
    if(!sRegressor.empty())
    {
        landmarker = std::make_shared<FaceLandmarker>(sRegressor, sDetector);
    }
    
    std::vector<cv::Point2f> landmarks;
    if(landmarker)
    {
        cv::Mat gray;
        cv::extractChannel(input, gray, 1);
        landmarks = (*landmarker)(gray, {});
    }
    
    if(!sMapping.empty() && !sModel.empty() && landmarks.size())
    {
        cv::Mat iso;
        eos::render::Mesh mesh;
        cv::Point3f R = mapper(landmarks, input, mesh, iso);
        
        std::cout << "R = " << R << std::endl;
        
        // (((( Draw mesh for visualization ))))
        if(verbose)
        {
            for(auto & p : mesh.texcoords)
            {
                p[0] *= iso.cols;
                p[1] *= iso.rows;
            }
            
            for(int i = 0; i < mesh.tvi.size(); i++)
            {
                const auto &t = mesh.tvi[i];
                cv::Point2f v0 = mesh.texcoords[t[0]];
                cv::Point2f v1 = mesh.texcoords[t[1]];
                cv::Point2f v2 = mesh.texcoords[t[2]];
                cv::line(iso, v0, v1, {0,255,0}, 1, 8);
                cv::line(iso, v1, v2, {0,255,0}, 1, 8);
                cv::line(iso, v2, v0, {0,255,0}, 1, 8);
            }
            cv::imshow("iso", iso);
            cv::waitKey(0);
        }
    }
}
