// Copyright (c) 2016, David Hirvonen
// All rights reserved.

#include "facelab/Filter.h"
#include "facelab/homomorphic.hpp"
#include "facelab/FaceLandmarker.h"

#include <iostream>
#include <sstream>
#include <numeric>
#include <fstream>

class ScopeTimer
{
public:
    ScopeTimer(const std::string &name, std::ostream &os) : m_name(name), m_os(os)
    {
        m_tic = cv::getTickCount();
    }
    ~ScopeTimer()
    {
        int64 toc = cv::getTickCount();
        double elapsed = double(toc - m_tic)/double(cv::getTickFrequency());
        m_os << m_name << " " << elapsed << std::endl;
    }
protected:
    int64 m_tic;
    std::string m_name;
    std::ostream &m_os;
};

#include <opencv2/photo.hpp>

// ############### homomorphic ###############

class HomomorphicFilter : public Filter
{
public:
    HomomorphicFilter(float cutoff = 0.49, int order = 2, float boost = 4.0)
    : kCutoff(cutoff)
    , kOrder(order)
    , kBoost(boost) {}
    
    virtual cv::Mat operator()(const cv::Mat &src, cv::Mat &dst)
    {
        const float kBeta = (1.0/kBoost);
        const float kAlpha = (1.0 - kBeta);
        homomorphic(src, dst, kCutoff, kOrder, kAlpha, kBeta, {0.00f, 0.94f}, dst.empty() ? cv::Mat() : dst);
        return dst;
    }
    
    virtual const char * getFilterName() const { return "HomomorphicFilter"; }

    int kOrder = 2;
    float kBoost = 4.0;
    float kCutoff = 0.49;
};

// ############### bilateral #################

static void bilateral(const cv::Mat &input, cv::Mat &output, int spatialRad, int colorRad, int iter)
{
    std::vector<cv::Mat> buffer { cv::Mat(), output };
    int j = 1;
    for(int i = 0; i < iter; i++, j = 1-j)
    {
        cv::bilateralFilter((i == 0) ? input : buffer[1-j], buffer[j], 0, spatialRad, colorRad);
    }
    output = buffer[1-j];
}

class BilateralFilter : public Filter
{
public:
    
    BilateralFilter(int spatialRad, int colorRad, int iter)
    : m_spatialRad(spatialRad)
    , m_colorRad(colorRad)
    , m_iter(iter)
    { }
    
    virtual cv::Mat operator()(const cv::Mat &src, cv::Mat &dest)
    {
        bilateral(src, dest, m_spatialRad, m_colorRad, m_iter);
        return dest;
    }
    
    virtual const char * getFilterName() const { return "BilateralFilter"; }
    
    int m_iter = 10;
    int m_spatialRad = 10;
    int m_colorRad = 10;
};

// ########### inpaint filter ################

class InpaintFilter : public Filter
{
public:
    InpaintFilter(int radius) : m_radius(radius) {}
    virtual cv::Mat operator()(const cv::Mat &src, cv::Mat &dst)
    {
        cv::Mat saturated = (src >= 254) | (src <= 2);
        cv::reduce(saturated.reshape(1, saturated.total()), saturated, 1, cv::REDUCE_MAX);
        saturated = saturated.reshape(1, src.rows);
        cv::inpaint(src, saturated, dst, m_radius, cv::INPAINT_TELEA);
        return dst;
    }
    virtual const char * getFilterName() const { return "InpaintFilter"; }
    float m_radius = 5;
};

// ########## Symmetry filter ##################

class SymmetryFilter : public Filter
{
public:
    SymmetryFilter() {}

    virtual cv::Mat operator()(const cv::Mat &src, cv::Mat &dst)
    {
        return src;
    }
    
    virtual const char * getFilterName() const { return "SymmetryFilter"; }
};

// ############# Cartoonize ##################

class CartoonizeFilter : public Filter
{
public:
    
    CartoonizeFilter(float sigmaSpace, float sigmaColor, int iter, int medianKernel)
    : m_sigmaSpace(sigmaSpace)
    , m_sigmaColor(sigmaColor)
    , m_iter(iter)
    {
        
    }
    
    bool m_doLines = false;
    void setDoLines(bool flag)
    {
        m_doLines = flag;
    }
    
    virtual cv::Mat operator()(const cv::Mat &src, cv::Mat &dst)
    {
        cv::Mat small, smooth;
        cv::resize(src, small, {src.cols/2, src.rows/2});
        bilateral(small, smooth, m_sigmaSpace, m_sigmaColor, m_iter * 2);
        cv::resize(smooth, smooth, src.size(), 0, 0, cv::INTER_LINEAR);
        
        if(m_doLines)
        {
            smooth.copyTo(dst, makeLines(src));
        }
        else
        {
            dst = smooth;
        }
        
        return dst;
    }
    
    cv::Mat makeLines(const cv::Mat &src)
    {
        cv::Mat gray, edges, mask;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        for(int i = 0; i < 2; i++)
        {
            cv::medianBlur(gray, gray, m_medianKernel);
        }
        
        cv::Laplacian(gray, edges, CV_8UC1, 5);
        cv::threshold(edges, mask, 0, 255,  cv::THRESH_OTSU  | cv::THRESH_BINARY_INV);
        return mask;
    }
    
    virtual const char * getFilterName() const { return "CartoonizeFilter"; }
    
protected:
    
    int m_medianKernel = 7;
    
    float m_sigmaSpace = 7;
    float m_sigmaColor = 9;
    int m_d = 0;
    int m_iter = 7;
};


const char *keys =
{
    "{ input     |       | input filename                            }"
    "{ hair-mask |       | input mask file                           }"
    "{ output    |       | output filename                           }"
    
    "{ log       |       | log file                                  }"
    
    "{ width     | 512   | processing width                          }"
    "{ verbose   | false | verbose mode (w/ display)                 }"

    // Tracker file
    "{ regressor |       | face landmark regressor file              }"
    "{ detector  |       | face detector                             }"

    "{ model     |       | model file                                }"
    "{ mapping   |       | mapping file                              }"
    
    // Triangulation
    "{ triangles |       | input precomputed triangulation           }"
    "{ triangles-out |   | output triangulation file                 }"

    "{ threads   | false | use worker threads when possible          }"
    "{ build     | false | print the OpenCV build information        }"
    "{ help      | false | print help message                        }"
};

#define SHOW_HISTORY 0
#define DO_INPAINT 0

int main(int argc, char *argv[])
{
    const std::string PATTERN = "HOME";
    std::string home = getenv(PATTERN.c_str());

    cv::CommandLineParser parser(argc, argv, keys);
    if(argc < 2 || parser.get<bool>("help"))
    {
        std::cout << "ARGC: " << argc << std::endl;
        parser.printMessage();
        return 0;
    }
    else if(parser.get<bool>("build"))
    {
        std::cout << cv::getBuildInformation() << std::endl;
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
        std::cerr << "Must specify output filename" << std::endl;
        return 1;
    }

    std::string sHairMask = parser.get<std::string>("hair-mask");
    std::string sLog = parser.get<std::string>("log");
    std::string sDetector = parser.get<std::string>("detector");
    std::string sRegressor = parser.get<std::string>("regressor");
    std::string sTriangles = parser.get<std::string>("triangles");
    std::vector<std::string*> args
    {
        &sInput,
        &sOutput,
        &sDetector,
        &sRegressor,
        &sTriangles,
        &sLog,
        &sHairMask
    };
    for(auto &arg : args)
    {
        size_t pos = arg->find(PATTERN);
        if(pos != std::string::npos)
        {
            arg->replace(pos, PATTERN.size(), home);
        }
    }

    // Create log stream:
    std::ostream *os;
    std::ofstream log;
    if(!sLog.empty())
    {
        log.open(sLog);
        os = &log;
    }
    else
    {
        os = &std::cout;
    }        

    bool verbose =  parser.get<bool>("verbose");
    
    cv::Mat input = cv::imread(sInput, cv::IMREAD_COLOR);
    if(input.empty())
    {
        std::cerr << "Unable to read input file " << sInput << std::endl;
        return 1;
    }
    
    cv::Mat hairMask = cv::imread(sHairMask, cv::IMREAD_GRAYSCALE);
    if(hairMask.empty())
    {
        hairMask = cv::Mat1b::zeros(input.size()); // create empty hair mask
    }
    CV_Assert(hairMask.size() == input.size());
    
    std::vector<std::pair<std::string,cv::Mat>> drawings;

    float sigmaSpace = 7;
    float sigmaColor = 7;
    int iter = 7;
    int medianKernel = 7;
    CartoonizeFilter cartoonizeFilter(sigmaSpace, sigmaColor, iter, medianKernel);
    
    *os << sInput << ": " << input.size() << std::endl;
    
    int width = parser.get<int>("width");
    cv::Size size = input.size();
    cv::resize(input, input, {width, input.rows * width/input.cols}, cv::INTER_CUBIC);
    cv::resize(hairMask, hairMask, {width, input.rows * width/input.cols}, cv::INTER_NEAREST);
    hairMask = hairMask > 0;

    //cv:: q("hairMask", hairMask);
    
    *os << sInput << ": " << input.size() << std::endl;
    
    // ######### Homomorhpic filter ############
    HomomorphicFilter homomorphicFilter(0.4, 2, 2.0);

    // ######### LANDMARK ######################
    *os << "Create landmarker:" << sRegressor << "," << sDetector << std::endl;
    std::shared_ptr<FaceLandmarker> landmarker;
    if(!sRegressor.empty())
    {
        landmarker = std::make_shared<FaceLandmarker>(sRegressor, sDetector);
        landmarker->canvas = input.clone();
        if(!sTriangles.empty())
        { // optional
            landmarker->readTriangulation(sTriangles);
        }
    }

    *os << "Do landmarks" << std::endl;
    std::vector<cv::Point2f> landmarks;
    if(landmarker)
    {
        cv::Mat gray;
        cv::extractChannel(input, gray, 1);

        { // FaceLandmarker:
            ScopeTimer timer("FaceLandmarker+Detection", *os);
            landmarks = (*landmarker)(gray, {});
            
            *os << landmarker->getRoi() << " " << double(landmarker->getRoi().area())/double(gray.size().area()) << std::endl;
        }
        
        std::string sTrianglesOut = parser.get<std::string>("triangles-out");
        if(!sTrianglesOut.empty())
        {
            landmarker->writeTriangulation(sTrianglesOut);
        }
        
        // DEBUG
        cv::Mat canvas = input.clone();
        landmarker->draw(canvas);
        drawings.emplace_back("FaceLandmarker", canvas);
    }
    
    // ######### INPAINT ################
    InpaintFilter inpaintFilter(int(landmarker->iod() * 0.25 + 0.5f));
    
    // ######### BILATERAL ##############
    BilateralFilter bilateralFilter(int(landmarker->iod() * 0.1 + 0.5f), 5, 10);
    
    cv::Mat filled;

#if DO_INPAINT
    *os << "Do inpaint" << std::endl;    
    {
        ScopeTimer timer("InpaintFilter", *os);
        inpaintFilter(input, filled);
    }
#if SHOW_HISTORY
    drawings.emplace_back( inpaintFilter.getNamedDrawing(filled) );
#endif // endif(SHOW_HISTORY)
    
#else // else(DO_INPAINT)
    filled = input.clone();
#endif // endif(DO_INPAINT)
    
    cv::Mat even;
    {
        *os << "Do homomorphic" << std::endl;            
        ScopeTimer timer("HomomorphicFilter", *os);
        homomorphicFilter(filled, even);
    }
    drawings.emplace_back( homomorphicFilter.getNamedDrawing(even) );
    
    cv::Mat smoothFull;
    {
        ScopeTimer timer("BilateralFilter", *os);
        bilateralFilter(even, smoothFull);
    }
    
    cv::Mat symmetric;
    {
        ScopeTimer timer("SymmetryFilter", *os);
        landmarker->balance(smoothFull, symmetric);
    }
#if SHOW_HISTORY
    drawings.emplace_back("Symmetry", symmetric);
#endif
    
    cv::Mat smooth;
    {
        ScopeTimer timer("BilateralFilter", *os);
        bilateralFilter(symmetric, smooth);
    }
#if SHOW_HISTORY
    drawings.emplace_back( bilateralFilter.getNamedDrawing(smooth) );
#endif
    
    // Create a face mask
    cv::Mat faceMask = (symmetric > 0);
    cv::reduce( faceMask.reshape(1, faceMask.total()), faceMask, 1, cv::REDUCE_MAX);
    faceMask = faceMask.reshape(1, input.rows);

    CV_Assert(hairMask.size() == faceMask.size());
    cv::Mat mask = (faceMask | hairMask);
    
    //cv::imshow("both-masks", mask);
    
    //cv::imshow("even", even);
    //cv::imshow("mask", mask);
    //cv::imshow("mask", mask); cv::waitKey(0);

    { // Head segmentation:
        ScopeTimer timer("Grabcut", *os);
        auto info = landmarker->segmentHead(even, mask);
        auto head = info.first;
        
        cv::Mat labels = landmarker->getLabels() * 255/3;
        cv::cvtColor(labels, labels, cv::COLOR_GRAY2BGR);
        drawings.emplace_back("labels", labels);

        //cv::medianBlur(smoothFull, smoothFull, 3);
        smoothFull.setTo(cv::Scalar(255,0,0), ~head);
        
        // Paste symmetric face back into head:
        symmetric.copyTo(smoothFull, faceMask);
        drawings.emplace_back("comp", smoothFull);
    }
    
    //cv::Mat shiftedFull;
    //cv::pyrMeanShiftFiltering(smoothFull, shiftedFull, 10, 10, 5);
    //cv::Mat final = shiftedFull;
    //drawings.emplace_back("final", final.clone());
    
    cv::Mat final = smoothFull.clone();
    
    cv::resize(final, final, size, 0, 0, cv::INTER_LANCZOS4);
    cv::imwrite(sOutput, final);

    // Dump the output
    if(verbose)
    {
        cv::Mat canvas;
        std::vector<cv::Mat> images;
        for(auto &i : drawings)
        {
            images.push_back(i.second);
        }
        cv::hconcat(images, canvas);
        cv::imshow("canvas", canvas); cv::waitKey(0);
    }
 
    return 0;
}
