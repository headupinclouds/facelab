#include "facelab/homomorphic.hpp"
#include "facelab/shift.hpp"
#include <opencv2/core.hpp>

void fftshift(const cv::Mat& input, cv::Mat& output)
{
    shift(input, output, cv::Point2f(input.cols / 2, input.rows / 2), cv::BORDER_WRAP);
}

void ifftshift(const cv::Mat& input, cv::Mat& output)
{
    shift(input, output, cv::Point2f(-input.cols / 2, -input.rows / 2), cv::BORDER_WRAP);
}

cv::Mat complex(const cv::Mat &image)
{
    cv::Mat output;
    cv::Mat zeros = cv::Mat::zeros(image.size(), image.type());
    std::vector<cv::Mat> args{image, zeros}    
    cv::merge(args, output);
    return output;
}

cv::Vec2f findLimits(const cv::Mat &image, const cv::Mat &mask, const cv::Vec2f &range)
{
    cv::Mat1f hist;
    
    const int bins = 1000;
    cv::Vec2d vals;
    cv::minMaxLoc(image, &vals[0], &vals[1]);
    
    cv::Vec2f limits = vals;
    const float *ranges[] = { &limits[0] };
    cv::calcHist(&image, 1, 0, mask, hist, 1, &bins, ranges, true, false);
    
    const float total = cv::sum(hist)[0];
    const float step = (limits[1] - limits[0]) / static_cast<float>(bins);
    float tally = 0.f, lower = 0.f, upper = 0.f;
    float *h = hist.ptr<float>(0);
    
    int i = 0;
    for(; i < hist.cols; i++, h++)
    {
        tally += h[0];
        if((tally / total) > range[0])
        {
            break;
        }
    }
    lower = vals[0] + (step * i);
    
    for(; i < hist.cols; i++, h++)
    {
        tally += h[0];
        if((tally / total) > range[1])
        {
            break;
        }
    }
    upper = vals[1] + (step * i);
    
    return cv::Vec2f(lower, upper);
}

cv::Mat1f &butterworth(cv::Mat1f &filter, float cutoff, int n)
{
    cv::Point2f center(filter.cols/2, filter.rows/2);
    for(int y = 0; y < filter.rows; y++)
    {
        for(int x = 0; x < filter.cols; x++)
        {
            cv::Point2f p((float(x) - center.x) / float(filter.cols), (float(y) - center.y) / float(filter.rows));
            float r = cv::norm(p);
            filter(y,x) = 1.0 / (1.0 + std::pow(r/cutoff, 2.0*n));
        }
    }
    return filter;
}

void highboost(const cv::Mat1f &image, cv::Mat1f &output, float cutoff, int order, float alpha, float beta, const cv::Vec2f &range, const cv::Mat1b &mask)
{
    // 0) get optimal DFT size
    cv::Size size(cv::getOptimalDFTSize(image.cols), cv::getOptimalDFTSize(image.rows));
    
    // 1) create butterworth filter with padded image dimensions
    cv::Mat1f filter(size, 0.f);
    butterworth(filter, cutoff, order); // low pass filter
    ifftshift(alpha * (1.0 - filter) + beta, filter);

    // 2) padd the image
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, 0, size.height-image.rows, 0, size.width-image.cols, cv::BORDER_REFLECT);
   
    // 3) log transform
    cv::Mat scratch;
    cv::log((padded + 0.01f), scratch);

    // 5) FFT
    cv::Mat response, transform;
    cv::dft(complex(scratch), transform, cv::DFT_COMPLEX_OUTPUT);

    // 6) convolution
    cv::mulSpectrums(transform, complex(filter), response, cv::DFT_COMPLEX_OUTPUT, false);
    
    // 7) IFFT
    cv::idft(response, scratch, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    // 8) exp
    cv::exp(scratch, scratch);
    
    // 9) range clip
    cv::Vec2f thresholds = findLimits(scratch, mask, range);
    scratch = cv::min(cv::max(scratch, thresholds[0]), thresholds[1]);
    
    // 10) normalize
    cv::normalize(scratch, output, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    
    output = output(cv::Rect({0,0}, image.size()));
}

void homomorphic(const cv::Mat &image, cv::Mat &output, float cutoff, int n, float alpha, float beta, const cv::Vec2f &range, const cv::Mat &mask)
{
    cv::Mat hsi;
    cv::cvtColor(image, hsi, cv::COLOR_BGR2HSV_FULL);
    
    cv::Mat3f hsi32f(image.size());
    hsi.convertTo(hsi32f, CV_32F, 1.0/255.0);
    
    cv::Mat1f channels[3], reflectance;
    cv::split(hsi32f, channels);
    highboost(channels[2], reflectance, cutoff, n, alpha, beta, range, mask);

    cv::swap(channels[2], reflectance);
    cv::merge(channels, 3, hsi32f);
    hsi32f.convertTo(hsi, CV_8U, 255.0);
    cv::cvtColor(hsi, output, cv::COLOR_HSV2BGR_FULL);
}
