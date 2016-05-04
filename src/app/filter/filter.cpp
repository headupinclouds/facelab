#include "facelab/Filter.h"
#include "facelab/homomorphic.hpp"
#include "facelab/FaceLandmarkMeshMapper.h"
#include "facelab/DiffusionFilter.h"

#include "local_laplacian.h"
#include "FaceLandmarker.h"

#include <opencv2/bioinspired/bioinspired.hpp>

#include <iostream>
#include <sstream>
#include <numeric>

#include <opencv2/features2d.hpp>

// ################ retina ####################

class RetinaFilter
{
public:
    RetinaFilter(const std::string &filename) : m_filename(filename) {}
    
    virtual cv::Mat operator()(const cv::Mat &src, cv::Mat &dst)
    {
        init(src.size());
        
        // reset all retina buffers (imagine you close your eyes for a long time)
        m_retina->clearBuffers();
        
        // declare retina output buffers
        // processing loop with no stop condition
        // run retina filter on the loaded input frame
        m_retina->run(src);
        
        // Retrieve and display retina output
        //cv::Mat retinaOutput_parvo;
        //myRetina->getParvo(retinaOutput_parvo);
        
        cv::Mat retinaOutput_magno;
        m_retina->getMagno(retinaOutput_magno);
        
        cv::Mat hsi, channels[3];
        cv::cvtColor(src, hsi, cv::COLOR_BGR2HSV_FULL);
        cv::split(hsi, channels);
        channels[2] = retinaOutput_magno;
        cv::merge(channels, 3, hsi);
        cv::cvtColor(hsi, dst, cv::COLOR_HSV2BGR_FULL);
        
        return dst;
    }
    
    void init(const cv::Size &size)
    {
        if(m_retina.empty() || m_retina->getInputSize() != size)
        {
            m_retina = cv::bioinspired::createRetina(size, m_colorModel, m_colorSamplingMethod, m_useRetinaLogSampling, m_reductionFactor, m_samplingStrength);
            m_retina->setup(m_filename);
        }
     }
    
    cv::Mat draw(const cv::Mat &src)
    {
        return src;
    }
    
    virtual const char * getFilterName() const { return "RetinaFilter"; }
    
    std::string m_filename;
    cv::Ptr<cv::bioinspired::Retina> m_retina;
    
    bool m_colorModel = true;
    int m_colorSamplingMethod= cv::bioinspired::RETINA_COLOR_BAYER;
    const bool m_useRetinaLogSampling=false;
    const float m_reductionFactor=1.0f;
    const float m_samplingStrength=10.0f;
};

// ############### homomorphic ###############

class HomomorphicFilter : public Filter
{
public:
    HomomorphicFilter() {}
    
    virtual cv::Mat operator()(const cv::Mat &src, cv::Mat &dst)
    {
        const float kBeta = (1.0/kBoost);
        const float kAlpha = (1.0 - kBeta);
        homomorphic(src, dst, kCutoff, kOrder, kAlpha, kBeta, {0.00f, 0.96f});
        return dst;
    }
    
    virtual const char * getFilterName() const { return "HomomorphicFilter"; }

    int kOrder = 4;
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

// ########## Local laplacian filter ############

class FFLocalLaplacianFilter : public Filter
{
public:
    FFLocalLaplacianFilter(double kAlpha, double kBeta, double kSigmaR, int kLevels)
    : kAlpha(kAlpha)
    , kBeta(kBeta)
    , kSigmaR(kSigmaR)
    , kLevels(kLevels)
    {
    }
    
    virtual cv::Mat operator()(const cv::Mat &src, cv::Mat &dest)
    {
        cv::Mat scratch;
        src.convertTo(scratch, CV_64F, 1 / 255.0);
        switch(scratch.channels())
        {
            case 1:
                scratch = LocalLaplacianFilter<double>(scratch, kAlpha, kBeta, kSigmaR, kLevels);
                scratch *= 255;
                scratch.convertTo(dest, CV_8UC1);
                break;
            case 3:
                scratch = LocalLaplacianFilter<cv::Vec3d>(scratch, kAlpha, kBeta, kSigmaR, kLevels);
                scratch *= 255;
                scratch.convertTo(dest, CV_8UC3);
                break;
            default:
                std::cerr << "Input image must have 1 or 3 channels." << std::endl;
                break;
        }
        return dest;
    }
    
    double kAlpha, kBeta, kSigmaR;
    int kLevels;
    
    virtual const char * getFilterName() const { return "LocalLaplacianFilter"; }
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

static cv::Mat drawIsoMesh(FaceLandmarkMeshMapper &mapper, const std::vector<cv::Point2f> &landmarks, cv::Mat &image)
{
    cv::Mat iso;
    eos::render::Mesh mesh;
    mapper(landmarks, image, mesh, iso);
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
    return iso;
}


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
    "{ retina    |       | retina parameters                         }"

    "{ model     |       | model file                                }"
    "{ mapping   |       | mapping file                              }"
    
    // Triangulation
    "{ triangles |       | input precomputed triangulation           }"
    "{ triangles-out |   | output triangulation file                 }"
    
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
    
    std::vector<std::pair<std::string,cv::Mat>> drawings;
    
    float sigmaSpace = 7;
    float sigmaColor = 7;
    int iter = 7;
    int medianKernel = 7;
    CartoonizeFilter cartoonizeFilter(sigmaSpace, sigmaColor, iter, medianKernel);
    
    // ############ RETINA ###############
    std::string sRetina = parser.get<std::string>("retina");
    if(sRetina.empty())
    {
        std::cerr << "Unable to read the input file " << sRetina << std::endl;
        return 1;
    }
    RetinaFilter retinaFilter(sRetina);
    
    int width = parser.get<int>("width");
    cv::resize(input, input, {width, input.rows * width/input.cols}, cv::INTER_CUBIC);
    
    // ########## FACE MESH LANDMARKER #########
    std::string sModel = parser.get<std::string>("model");
    std::string sMapping = parser.get<std::string>("mapping");
    FaceLandmarkMeshMapper mapper(sModel, sMapping);
    
    // ######### Homomorhpic filter ############
    HomomorphicFilter homomorphicFilter;
    
    // ######### LOCAL LAPLACIAN ###############
    const double kSigmaR = parser.get<double>("sigma"); // 0.3
    const double kAlpha = parser.get<double>("alpha");  // 2.0
    const double kBeta = parser.get<double>("beta");    // 1.0
    const int kLevels = parser.get<int>("levels");      // 3
    FFLocalLaplacianFilter localLaplacianFilter(kAlpha, kBeta, kSigmaR, kLevels);
    
    // ######### LANDMARK ######################
    std::shared_ptr<FaceLandmarker> landmarker;
    std::string sDetector = parser.get<std::string>("detector");
    std::string sRegressor = parser.get<std::string>("regressor");
    std::string sTriangles = parser.get<std::string>("triangles");
    if(!sRegressor.empty())
    {
        landmarker = std::make_shared<FaceLandmarker>(sRegressor, sDetector);
        if(!sTriangles.empty())
        { // optional
            landmarker->readTriangulation(sTriangles);
        }
    }
    
    std::vector<cv::Point2f> landmarks;
    if(landmarker)
    {
        cv::Mat gray;
        cv::extractChannel(input, gray, 1);
        landmarks = (*landmarker)(gray, {});
        
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
    BilateralFilter bilateralFilter(int(landmarker->iod() * 0.125f + 0.5f), 3, 4);
    
    cv::Mat filled;
    inpaintFilter(input, filled);
    drawings.emplace_back( inpaintFilter.getNamedDrawing(filled) );
    
    cv::Mat even;
    homomorphicFilter(filled, even);
    drawings.emplace_back( homomorphicFilter.getNamedDrawing(even) );
    
    cv::Mat smooth;
    bilateralFilter(even, smooth);
    drawings.emplace_back( bilateralFilter.getNamedDrawing(smooth) );
        
    cv::Mat symmetric;
    landmarker->balance(even, symmetric);
    drawings.emplace_back("Symmetry", symmetric);
        
    {// Create a face mask
        cv::Mat mask = (symmetric > 0);
        cv::reduce( mask.reshape(1, mask.total()), mask, 1, cv::REDUCE_MAX);
        mask = mask.reshape(1, input.rows);
        cv::Mat head = landmarker->segmentHead(even, mask);
        even.setTo(0, ~head);
    }
    
    if(!sMapping.empty() && !sModel.empty() && landmarks.size())
    {
        cv::Mat iso = drawIsoMesh(mapper, landmarks, input);
        cv::imshow("iso", iso);
    }

    cv::Mat canvas;
    std::vector<cv::Mat> images;
    for(auto &i : drawings)
    {
        images.push_back(i.second);
    }
    cv::hconcat(images, canvas);
    cv::imshow("input_homomorphic_symmetry", canvas);
    cv::imwrite("/tmp/cartoon.jpg", canvas);
    cv::waitKey(0);
    exit(1);
 
    return 0;
}
