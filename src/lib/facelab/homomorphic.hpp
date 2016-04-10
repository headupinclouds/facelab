#ifndef HOMOMORPHIC_H
#define HOMOMORPHIC_H  1

#include <opencv2/core.hpp>

void homomorphic(const cv::Mat &image, cv::Mat &output, float cutoff, int n, float alpha, float beta, const cv::Vec2f &range, const cv::Mat &mask=cv::Mat());

#endif // HOMOMORPHIC_H 
