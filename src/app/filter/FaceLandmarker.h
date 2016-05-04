#ifndef FACE_LANDMARKER_H
#define FACE_LANDMARKER_H 1

#include <dest/dest.h> 
#include <dest/util/convert.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
class FaceLandmarker
{
public:

    FaceLandmarker(const std::string &sRegressor, const std::string &sDetector);
    
    cv::RotatedRect getFaceEllipse(const cv::Size &size, float width=2.0, float height=3.0) const;
       
    std::vector<cv::Point2f> fitLandmarks(const cv::Mat &gray);
    
    std::vector<cv::Point2f>& operator()(const cv::Mat1b &gray, const cv::Rect &roi);
    
    std::vector<cv::Point2f>& operator()(const cv::Mat1b &gray);

    cv::Point2f getMean(const cv::Range &range);
    
    cv::Mat1b segmentHead(const cv::Mat &image, const cv::Mat1b &mask);
    
    void balance(const cv::Mat &image, cv::Mat &symmetric);
    
    float iod() const;
    
    void getMode(const cv::Mat &canvas);

    
    // 17-21 : right eye brow
    // 36-41 : right eye
    // 22-26 : left eye brow
    // 42-47 : left eye
    // 27-35 : nose
    // 48-59 : mouth
    
    cv::Mat1b getMask(const cv::Size &size, bool doFeatures = false) const;

    void draw(cv::Mat &canvas, const cv::Point2f &tl = {});
    int writeTriangulation(const std::string &filename) const;
    int readTriangulation(const std::string &filename);

protected:
    
    void delaunay(const cv::Size &size);
    void mirrorTriangulation(const std::vector<cv::Point2f> &landmarks, const std::vector<std::array<int,2>> &mirrorMap);
    
    cv::Rect m_roi;
    
    std::vector<cv::Point2f> m_landmarks;
    
    std::vector<cv::Vec6f> m_triangles[2];
    
    std::vector<cv::Vec3i> m_indices;
    
    static const std::vector<cv::Range> kContours;
    static const std::vector<cv::Range> kCurves;
    static const std::vector<std::array<int,2>> kMirrorMap;
    
    cv::CascadeClassifier m_detector;
    dest::core::Tracker m_tracker;
};

#endif // FACE_LANDMARKER_H
