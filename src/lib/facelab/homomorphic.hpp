// Copyright (c) 2016, David Hirvonen
// All rights reserved.

#ifndef HOMOMORPHIC_H
#define HOMOMORPHIC_H  1

#include <opencv2/core.hpp>

// See: http://www.peterkovesi.com/matlabfns/FrequencyFilt/homomorphic.m
void homomorphic(const cv::Mat &image, cv::Mat &output, float cutoff, int n, float alpha, float beta, const cv::Vec2f &range, const cv::Mat &mask=cv::Mat());

#endif // HOMOMORPHIC_H 
