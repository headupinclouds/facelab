// Copyright (c) 2016, David Hirvonen
// All rights reserved.

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

class Filter
{
public:
    virtual cv::Mat operator()(const cv::Mat &src, cv::Mat &dest) = 0;
    virtual cv::Mat draw(const cv::Mat &src)
    {
        return src;
    }
    virtual const char *getFilterName() const = 0;
    
    virtual std::pair<std::string, cv::Mat> getNamedDrawing(const cv::Mat &src)
    {
        cv::Mat canvas = draw(src.clone());
        cv::putText(canvas, getFilterName(), {0,canvas.rows}, CV_FONT_HERSHEY_SIMPLEX, 0.75, {0,255,0}, 1, 8, false);
        return std::make_pair(getFilterName(), canvas);
    }
};

#endif // FILTER_H
