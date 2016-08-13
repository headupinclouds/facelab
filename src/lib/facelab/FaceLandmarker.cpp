// Copyright (c) 2016, David Hirvonen
// All rights reserved.

#include "FaceLandmarker.h"

#include <iostream>
#include <array>


// Not working on VS 14 2015 (Appveyor)
//#define _USE_MATH_DEFINES // for C++
//#include <cmath>

#define FL_M_SQRT2 1.41421356237309504880

template <typename T>
std::vector<std::vector<cv::Point>> pointToContours(const std::vector<cv::Point_<T>> &points)
{
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    
    std::vector<std::vector<cv::Point>> poly(1);
    std::copy(hull.begin(), hull.end(), std::back_inserter(poly[0]));
    return poly;
}

template <>
std::vector<std::vector<cv::Point>> pointToContours(const std::vector<cv::Point> &points)
{
    std::vector<cv::Point2f> tmp;
    std::copy(points.begin(), points.end(), std::back_inserter(tmp));
    return pointToContours(tmp);
}

static cv::Point2f mean(const std::vector<cv::Point2f> &points)
{
    cv::Point2f p;
    for(const auto &q : points)
    {
        p += q;
    }
    return (p * (1.0f / float(points.size())));
}

static cv::Mat getAffineLeastSquares(const cv::Vec6f &T1, const cv::Vec6f &T2)
{
    std::vector<cv::Point2f> pt1 { {T1[0],T1[1]}, {T1[2],T1[3]}, {T1[4],T1[5]} };
    std::vector<cv::Point2f> pt2 { {T2[0],T2[1]}, {T2[2],T2[3]}, {T2[4],T2[5]} };
    return cv::getAffineTransform(pt1, pt2);
}

static bool contains(cv::Vec6f &triangle, const cv::Point2f &p)
{
    cv::Point2f p1(triangle[0],triangle[1]);
    cv::Point2f p2(triangle[2],triangle[3]);
    cv::Point2f p3(triangle[4],triangle[5]);
    
    float fAB = (p.y-p1.y)*(p2.x-p1.x) - (p.x-p1.x)*(p2.y-p1.y);
    float fCA = (p.y-p3.y)*(p1.x-p3.x) - (p.x-p3.x)*(p1.y-p3.y);
    float fBC = (p.y-p2.y)*(p3.x-p2.x) - (p.x-p2.x)*(p3.y-p2.y);
    return ((fAB*fBC)>0 && (fBC*fCA)>0);
}

////

FaceLandmarker::FaceLandmarker(const std::string &sRegressor, const std::string &sDetector)
{
    if(!sDetector.empty())
    {
        m_detector.load(sDetector);
    }

    assert(!sRegressor.empty());
    m_tracker.load(sRegressor); //  "/Users/dhirvonen/Downloads/dest_tracker_VJ_ibug.bin";
}

float FaceLandmarker::iod() const
{
    std::vector<cv::Point2f> eyeLPoints(m_landmarks.begin() + kContours[0].start, m_landmarks.begin() + kContours[0].end);
    std::vector<cv::Point2f> eyeRPoints(m_landmarks.begin() + kContours[1].start, m_landmarks.begin() + kContours[1].end);
    cv::Point2f eyeL = mean(eyeLPoints);
    cv::Point2f eyeR = mean(eyeRPoints);
    return cv::norm(eyeL - eyeR);
}
    
cv::RotatedRect FaceLandmarker::getFaceEllipse(const cv::Size &size, float width, float height) const
{
    std::vector<cv::Point2f> eyeLPoints(m_landmarks.begin() + kContours[0].start, m_landmarks.begin() + kContours[0].end);
    std::vector<cv::Point2f> eyeRPoints(m_landmarks.begin() + kContours[1].start, m_landmarks.begin() + kContours[1].end);
    cv::Point2f eyeL = mean(eyeLPoints);
    cv::Point2f eyeR = mean(eyeRPoints);
    
    const cv::Point2f v = eyeL - eyeR;
    const float iod = cv::norm(v);
    const float angle = std::atan2(v.y, v.x);
    cv::Point2f center = (m_landmarks[30] + eyeL + eyeR) * (1.0 / 3.0);
    return cv::RotatedRect(center, {iod*width, iod*height}, angle * 180.0 / M_PI);
}
    
std::vector<cv::Point2f> FaceLandmarker::fitLandmarks(const cv::Mat &gray)
{
    dest::core::Rect r, ur = dest::core::unitRectangle();
    dest::util::toDest(cv::Rect({0,0}, gray.size()), r);
    dest::core::ShapeTransform shapeToImage;
    shapeToImage = dest::core::estimateSimilarityTransform(ur, r);
    dest::core::MappedImage mappedGray = dest::util::toDestHeaderOnly(gray);
    dest::core::Shape s = m_tracker.predict(mappedGray, shapeToImage);
        
    std::vector<cv::Point2f> landmarks(s.cols());
    for(int i = 0; i < s.cols(); i++)
    {
        landmarks[i] = { s(0,i), s(1,i) };
    }
    return landmarks;
}
    
std::vector<cv::Point2f>& FaceLandmarker::operator()(const cv::Mat1b &gray, const cv::Rect &roi)
{
    // =========
    cv::Rect crop = roi;
    if(!m_detector.empty())
    {
        std::vector<cv::Rect> faces;
        cv::Size mini(gray.cols/2, gray.cols/2);
        cv::Size maxi(gray.cols, gray.cols);
        m_detector.detectMultiScale(gray, faces, 1.1, 1, 0, mini, maxi); // TODO: set reasonable upper lower sizes
        crop = faces.front();
    }
    m_roi = crop;
    
    // =========
    auto & landmarks = (*this)(gray(crop));
    for(auto &p : landmarks)
    {
        p += cv::Point2f(crop.x, crop.y);
    }
    
    delaunay(gray.size());
    
    return landmarks;
}
    
std::vector<cv::Point2f>& FaceLandmarker::operator()(const cv::Mat1b &gray)
{
    m_landmarks = fitLandmarks(gray);
    return m_landmarks;
}

cv::Point2f FaceLandmarker::getMean(const cv::Range &range)
{
    cv::Point2f p;
    for(int i = range.start; i < range.end; i++)
    {
        p += m_landmarks[i];
    }
    return (p * 1.0f / float(m_landmarks.size()));
}

cv::Point2f clip(const cv::Point2f &p, const cv::Rect &roi)
{
    cv::Point2f q
    (
     std::max(std::min(float(roi.br().x-1), p.x), float(roi.tl().x)),
     std::max(std::min(float(roi.br().y-1), p.y), float(roi.tl().y))
    );
    return q;
}

void FaceLandmarker::delaunay(const cv::Size &size)
{
    auto landmarks = m_landmarks;
    auto mirrorMap = kMirrorMap;
    
    { // Synthesize some new points, making sure to update the mirror map:
        cv::Point2f p27 = m_landmarks[27];
        cv::Point2f p30 = m_landmarks[30];
        cv::Point2f v = p27 - p30;
        cv::Point2f vn = cv::normalize(cv::Vec2f(v));

        const float scale = 1.0;

        //cv::Mat drawing = canvas.clone();
        
        for(int i = 9; i < 13; i++)
        {
            std::array<int, 2> indices;
            for(int j = 0; j < 2; j++)
            {
                const cv::Point2f fh0 = m_landmarks[mirrorMap[i][j]];
                const cv::Point2f v0 = cv::normalize(cv::Vec2f(fh0 - p30));
                const float w0 = (1.5 + std::abs(v0.dot(vn))) * cv::norm(v) * scale;
                const int index0 = landmarks.size();
                indices[j] = index0;
                cv::Point2f l0 = clip(p30 + (v0 * w0), cv::Rect({0,0}, size));
                landmarks.push_back(l0);

                //cv::circle(drawing, fh0, 2, {255,255,0}, -1, 8);
                //cv::circle(drawing, l0, 2, {0,255,0}, -1, 8);
                //cv::imshow("c", drawing); cv::waitKey(0);
            }
            mirrorMap.push_back(indices);
        }
        
        { // Add the center ray:
            cv::Point2f c = (m_landmarks[mirrorMap[11][0]] + m_landmarks[mirrorMap[11][1]]) * 0.5f;
            const cv::Point2f v0 = cv::normalize(cv::Vec2f(c - p30));
            const float w0 = (1.5 + std::abs(v0.dot(vn))) * cv::norm(v) * scale;
            cv::Point2f l = clip(p30 + (v0 * w0), cv::Rect({0,0}, size));
            
            const int index = landmarks.size();
            landmarks.push_back(l);
            mirrorMap.push_back( {{ index, index }} );
        }

    }
    
    if(m_indices.size())
    {
        // Create triangles from indices
        m_triangles[0].resize(m_indices.size());
        m_triangles[1].resize(m_indices.size());
        for(int i = 0; i < m_indices.size(); i++)
        {
            cv::Point2f p1 = landmarks[ m_indices[i][0] ];
            cv::Point2f p2 = landmarks[ m_indices[i][1] ];
            cv::Point2f p3 = landmarks[ m_indices[i][2] ];
            m_triangles[0][i] = cv::Vec6f(p1.x,p1.y,p2.x,p2.y,p3.x,p3.y);
        }
        mirrorTriangulation(landmarks, mirrorMap);
    }
    else
    {
        // Perform the delaunay triangulation
        
        // Use a delaunay subdivision and balance mirror triangles:
        cv::Rect roi({0,0}, size);
        cv::Subdiv2D subdiv;
        subdiv.initDelaunay(roi);
        for(int i = 0; i < mirrorMap.size(); i++)
        {
            auto p = landmarks[ mirrorMap[i][0] ];
            subdiv.insert(p);
        }
        std::vector<cv::Vec3i> indices;
        subdiv.getTriangleList(m_triangles[0]);
        
        auto pruner = [&](const cv::Vec6f &triangle)
        {
            cv::Point2f p1(triangle[0], triangle[1]);
            cv::Point2f p2(triangle[2], triangle[3]);
            cv::Point2f p3(triangle[4], triangle[5]);
            return (! (roi.contains(p1) && roi.contains(p2) && roi.contains(p3)) );
        };
        m_triangles[0].erase(std::remove_if(m_triangles[0].begin(), m_triangles[0].end(), pruner), m_triangles[0].end());
        m_indices.resize(m_triangles[0].size());
        m_triangles[1].resize(m_triangles[0].size());
        
        for(int i = 0; i < m_triangles[0].size(); i++)
        {
            const auto &triangle = m_triangles[0][i];
            cv::Point2f p1(triangle[0], triangle[1]);
            cv::Point2f p2(triangle[2], triangle[3]);
            cv::Point2f p3(triangle[4], triangle[5]);
            
            int k1 = 0, k2 = 0, k3 = 0;
            for(k1 = 0; k1 < mirrorMap.size(); k1++) { if(landmarks[mirrorMap[k1][0]] == p1) break; }
            for(k2 = 0; k2 < mirrorMap.size(); k2++) { if(landmarks[mirrorMap[k2][0]] == p2) break; }
            for(k3 = 0; k3 < mirrorMap.size(); k3++) { if(landmarks[mirrorMap[k3][0]] == p3) break; }
            
            m_indices[i] = cv::Vec3i(k1,k2,k3);
        }
        
        mirrorTriangulation(landmarks, mirrorMap);
    }
}

// Input:
//  1) m_indices : for triangles on the left side
//  2) mirrorMap : mapping from vertices of triangles on left side to corresponding vertices on right
//  3) landmarks : the landmark vector
void FaceLandmarker::mirrorTriangulation(const std::vector<cv::Point2f> &landmarks, const std::vector<std::array<int,2>> &mirrorMap)
{
    for(int i = 0; i < m_triangles[0].size(); i++)
    {
        cv::Point2f q1( landmarks[mirrorMap[m_indices[i][0]][1]] );
        cv::Point2f q2( landmarks[mirrorMap[m_indices[i][1]][1]] );
        cv::Point2f q3( landmarks[mirrorMap[m_indices[i][2]][1]] );
        m_triangles[1][i] = cv::Vec6f(q1.x,q1.y,q2.x,q2.y,q3.x,q3.y);
    }
}

// Load in the left side triangles
int FaceLandmarker::readTriangulation(const std::string &filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(fs.isOpened())
    {
        cv::FileNode n = fs["triangles"];
        if (n.type() != cv::FileNode::SEQ)
        {
            std::cerr << "triangles is not a sequence! FAIL" << std::endl;
            return 1;
        }
        
        cv::FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it)
        {
            cv::Vec3i triangle;
            (*it) >> triangle;
            m_indices.push_back(triangle);
        }
    }
    
    return 0;
}

int FaceLandmarker::writeTriangulation(const std::string &filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if(fs.isOpened())
    {
        fs << "triangles" << "[";
        for(int i = 0; i < m_indices.size(); i++)
        {
            fs << m_indices[i];
        }
        fs << "]";
    }
    
    return 0;
}

void FaceLandmarker::balance(const cv::Mat &image, cv::Mat &symmetric)
{
    CV_Assert(m_triangles[0].size() == m_triangles[1].size());
    
    const int n = m_triangles[0].size();
    std::vector<cv::Matx33f> M(n * 2, cv::Matx33f::eye());
    
    cv::Mat1b mask(image.size(), 0);
    for(int i = 0; i < m_triangles[0].size(); i++)
    {
        for(int j = 0; j < 2; j++)
        {
            const auto &t = m_triangles[j][i];
            std::vector<std::vector<cv::Point>> points
            {{
                cv::Point(t[0],t[1]),
                cv::Point(t[2],t[3]),
                cv::Point(t[4],t[5])
            }};
            int value = (i + 1) + (j * n);
            cv::fillPoly(mask, points, value, 4);
        }
        
        // Create affine transformations to mirror face:
        int i0 = i, i1 = i0 + n;
        cv::Mat H = getAffineLeastSquares(m_triangles[0][i], m_triangles[1][i]);
        H.convertTo(cv::Mat1f(M[i0].rows, M[i0].cols, (float*)&M[i0](0,0))(cv::Rect({0,0},H.size())), CV_32F);
        M[i1] = M[i0].inv();
    }

    // cv::dilate(mask, mask, cv::Mat(), {-1,-1}, 3);
    // cv::Mat canvas;
    // cv::normalize(mask, canvas, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // cv::imshow("zones", canvas); cv::waitKey(0);

    assert(m_triangles[0].size() == m_triangles[1].size());
    
    { // Mirror image and take mean:
        cv::Mat1f mapx(image.size(), 0.f);
        cv::Mat1f mapy(image.size(), 0.f);
        for(int y = 0; y < image.rows; y++)
        {
            for(int x = 0; x < image.cols; x++)
            {
                int index = mask(y,x);
                if(index-- > 0)
                {
                    cv::Point3f p(x,y,1.f), q = M[index] * p;
                    mapx(y,x) = q.x;
                    mapy(y,x) = q.y;
                }
            }
        }
        cv::Mat imageA = image.clone(), imageB;
        cv::remap(imageA, imageB, mapx, mapy, cv::INTER_LINEAR);
        imageA.setTo(0, ~(mask>0));
        imageB.setTo(0, ~(mask>0));
        
        //symmetric = cv::min(imageA, imageB);
        cv::addWeighted(imageA, 0.5, imageB, 0.5, 0.0, symmetric);
    }
}

std::pair<cv::Mat1b, cv::Rect> FaceLandmarker::segmentHead(const cv::Mat &image, const cv::Mat1b &faceMask)
{
    cv::Size size = image.size();
    auto face = getFaceEllipse(size);
    const float scale = FL_M_SQRT2;
    face.size.width *= scale;
    face.size.height *= scale;
    cv::Rect box = face.boundingRect();

    int y = 0;    
#if 0
    for(auto &p : m_landmarks)
    {
        if(p.y > y)
        {
            y = p.y;
        }
    }
#else
    y = image.rows;
#endif
    
    cv::Point2f tl = box.tl(), br = box.br();
    tl = face.center + (tl - face.center) * FL_M_SQRT2;
    br = face.center + (br - face.center) * FL_M_SQRT2;

#define DO_HEAD_AND_SHOULDERS 0
#if DO_HEAD_AND_SHOULDERS
    br.y = image.rows;
    box.height = (image.rows - box.y) + 1;
#endif
    
    cv::Rect roi(tl, br);
    roi &= cv::Rect(0,0,image.cols, image.rows);
    
    cv::Mat1b head;
    
    { // Segment the head:
        
        cv::Mat1b faceMaskSmall;
        cv::erode(faceMask, faceMaskSmall, {}, {-1,-1}, 8);
        
        cv::Mat1b labels(size, cv::GC_BGD);

        cv::rectangle(labels, box, cv::GC_PR_FGD, -1);
        labels.setTo(cv::GC_FGD, faceMaskSmall);

        cv::Mat fg, bg;
        cv::grabCut(image(roi), labels(roi), box, fg, bg, 10, cv::GC_EVAL);
        head = ((labels & 1) > 0);
        
        std::vector<cv::Vec4i> hierarchy;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(head, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1);
        
        std::pair<int, int> best(-1, -1);
        for(int i = 0; i < contours.size(); i++)
        {
            int a = cv::contourArea(contours[i]);
            if(a > best.second)
            {
                best = { i, a };
            }
        }
        
        std::vector<std::vector<cv::Point>> curve(1);
        cv::approxPolyDP(contours[best.first], curve[0], std::max(2.0, iod()/32.0) , true);
        
        head.setTo(0);
        cv::drawContours(head, curve, 0, 255, -1, 8);
        
        if(0)
        { // (((( Display ))))
            cv::Mat canvas = image.clone();
            cv::imshow("f", faceMask);
            cv::imshow("a", labels * 255/3);
            
            cv::ellipse(canvas, face, {0,255,0}, 1, 8);
            cv::rectangle(canvas, box, {255,255,255}, 1, 8);
            cv::imshow("head_face", canvas);
            cv::imshow("head_labels", labels * (255/3)); cv::waitKey(0);
        }
    }
    
    
    return std::make_pair(head, roi);
}

// 17-21 : right eye brow
// 36-41 : right eye
// 22-26 : left eye brow
// 42-47 : left eye
// 27-35 : nose
// 48-59 : mouth
    
cv::Mat1b FaceLandmarker::getMask(const cv::Size &size, bool doFeatures) const
{
    if(!m_landmarks.size())
    {
        return cv::Mat1b();
    }
        
    const auto &landmarks = m_landmarks;
    const float iod = cv::norm(landmarks[36] - landmarks[42]);

    // Basic face mask:
    cv::Mat1b mask(size, 0);
    cv::fillPoly(mask, pointToContours(m_landmarks), 255);
        
    if(!doFeatures)
    {
        for(auto &c : kContours)
        {
            std::vector<cv::Point2f> points;
            std::copy(landmarks.begin() + c.start, landmarks.begin() + c.end, std::back_inserter(points));
            cv::fillPoly(mask, pointToContours(points), 0);
        }
            
        for(auto &c : kCurves)
        {
            std::vector<cv::Point2f> points, hull;
            std::copy(landmarks.begin() + c.start, landmarks.begin() + c.end, std::back_inserter(points));
            for(int i = c.start+1; i < c.end-1; i++)
            {
                cv::Point2f d = cv::normalize(cv::Vec2f(landmarks[i-1] - landmarks[i+1])), v(d.y, -d.x);
                points.push_back(landmarks[i] + (v * iod/32.0));
                points.push_back(landmarks[i] - (v * iod/32.0));
            }
            cv::fillPoly(mask, pointToContours(points), 0);
        }
    }
        
    return mask;
}

void FaceLandmarker::getMode(const cv::Mat &canvas)
{
    cv::Mat hist;
    
    // We will use grabcut here:
    cv::Mat1b mask = getMask(canvas.size(), false);
    const int n = 3;
    const int sizes[] = { 32, 32, 32 };
    const int channels[] = { 0, 1, 2};
    const float range[] = { 0.f, 255.f };
    const float *ranges[] = { range, range, range };
    cv::calcHist(&canvas, 1, channels, mask, hist, n, sizes, ranges);
    
    assert(hist.type() == CV_32F);
    
    float best = 0.f;
    cv::Point3i p;
    for (int i=0; i<sizes[0]; i++)
    {
        for (int j=0; j<sizes[1]; j++)
        {
            for (int k=0; k<sizes[2]; k++)
            {
                if(hist.at<float>(i,j,k) > best)
                {
                    best = hist.at<float>(i,j,k);
                    p = cv::Point3i(i,j,k);
                }
            }
        }
    }
    
    cv::Scalar mu(p.x, p.y, p.z);
    for(int i = 0; i < 3; i++)
    {
        mu[i] *= range[1]/sizes[i];
    }
}

void FaceLandmarker::draw(cv::Mat &canvas, const cv::Point2f &tl)
{
    for(int i = 0; i < 2; i++)
    {
        for(auto &t : m_triangles[i])
        {
            cv::Point2f p1(t[0], t[1]);
            cv::Point2f p2(t[2], t[3]);
            cv::Point2f p3(t[4], t[5]);
            cv::line(canvas, p1, p2, {0,255,0}, 1, 8);
            cv::line(canvas, p2, p3, {0,255,0}, 1, 8);
            cv::line(canvas, p3, p1, {0,255,0}, 1, 8);
        }
    }
    
    for(const auto &p : m_landmarks)
    {
        cv::circle(canvas, p, 1, {0,255,0}, -1, 8);
    }
    
    //getModel(canvas);
}

const std::vector<cv::Range> FaceLandmarker::kContours
{
    {36, 41+1},
    {42, 47+1},
    {27, 35+1},
    {48, 59+1}
};

const std::vector<cv::Range> FaceLandmarker::kCurves
{
    {17, 21+1},
    {22, 26+1}
};

const std::vector<std::array<int,2>> FaceLandmarker::kMirrorMap
{
    // http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
    
    // Contour
    {{0,16}}, // 0
    {{1,15}},
    {{2,14}},
    {{3,13}},
    {{4,12}},
    {{5,11}},
    {{6,10}},
    {{7,9}},
    {{8,8}},
    
    // Eyebrow
    {{17,26}}, // 9
    {{18,25}},
    {{19,24}}, // 11
    {{20,23}},
    {{21,22}}, // 13
    
    // Nose
    {{27,27}}, // 14
    {{28,28}},
    {{29,29}},
    {{30,30}},
    
    {{31,35}},
    {{32,34}},
    {{33,33}},
    
    // Eye
    {{39,42}},
    {{38,43}},
    {{37,44}},
    {{36,45}},
    {{40,47}},
    {{41,46}},
    
    // Mouth
    {{48,54}},
    {{49,53}},
    {{50,52}},
    {{51,51}},
    
    {{59,55}},
    {{58,56}},
    {{57,57}},
    
    {{60,64}},
    {{61,63}},
    {{62,62}},
    
    {{67,65}},
    {{66,66}}
};
