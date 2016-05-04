#include "DiffusionFilter.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

DiffusionFilter::DiffusionFilter(int level_height, int level_width)
{
    options_.img_height = level_height;
    options_.img_width = level_width;
    Allocate_Memory_Evolution();
}
    
void DiffusionFilter::Allocate_Memory_Evolution(void)
{
    float rfactor = 0.0f;
    int level_height = 0, level_width = 0;
        
#if 1
    for(int i = 0; i < options_.omax; i++)
    {
        level_height = (int)(options_.img_height);
        level_width = (int)(options_.img_width);

        for(int j = 0; j < options_.nsublevels; j++)
        {
            dv::TEvolution step;
            step.Lx = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Ly = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lxx = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lxy = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lyy = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lt = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lsmooth = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.esigma = options_.soffset*pow(2.f, (float)(j) / (float)(options_.nsublevels) + i);
            step.sigma_size = std::round(step.esigma);
            step.etime = 0.5f*(step.esigma*step.esigma);
            step.octave = 0;
            step.sublevel = j;
            evolution_.push_back(step);
        }
    }
        
#else
    // Allocate the dimension of the matrices for the evolution
    for (int i = 0, power = 1; i <= options_.omax - 1; i++, power *= 2)
    {
        rfactor = 1.0f / power;
        level_height = (int)(options_.img_height*rfactor);
        level_width = (int)(options_.img_width*rfactor);
            
        // Smallest possible octave and allow one scale if the image is small
        if ((level_width < 80 || level_height < 40) && i != 0)
        {
            options_.omax = i;
            break;
        }
            
        for (int j = 0; j < options_.nsublevels; j++)
        {
            dv::TEvolution step;
            step.Lx = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Ly = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lxx = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lxy = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lyy = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lt = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.Lsmooth = cv::Mat::zeros(level_height, level_width, CV_32F);
            step.esigma = options_.soffset*pow(2.f, (float)(j) / (float)(options_.nsublevels) + i);
            step.sigma_size = std::round(step.esigma);
            step.etime = 0.5f*(step.esigma*step.esigma);
            step.octave = i;
            step.sublevel = j;
            evolution_.push_back(step);
        }
    }
#endif
        
    // Allocate memory for the number of cycles and time steps
    for (size_t i = 1; i < evolution_.size(); i++) {
        int naux = 0;
        std::vector<float> tau;
        float ttime = 0.0f;
        ttime = evolution_[i].etime - evolution_[i - 1].etime;
        naux = fed_tau_by_process_time(ttime, 1, 0.25f, reordering_, tau);

            
        CV_Assert((naux > 0) && (tau.size() > 0));
        for(auto &t : tau)
        {
            if(std::isnan(t) || (t == 0.f))
            {
                break;
            }
        }
        nsteps_.push_back(naux);
        tsteps_.push_back(tau);
        ncycles_++;
    }
        
    int pause = 1;
}

cv::Mat DiffusionFilter::operator()(const cv::Mat &img, cv::Mat &dst)
{
    cv::Mat gray;
    switch(img.channels())
    {
    case 1: gray = img; break;
    case 3: cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); break;
    default: break;
    }
        
    cv::Mat1f input(img.size()), output(img.size());
    gray.convertTo(input, input.type(), 1.0/255.0);
    (*this)(input, output);
        
    output.convertTo(gray, CV_8UC1);
    dst = gray;
    return gray;
}
    
void DiffusionFilter::Create_Nonlinear_Scale_Space(const cv::Mat& img)
{
    CV_Assert(evolution_.size() > 0);
        
    // Copy the original image to the first level of the evolution
    img.copyTo(evolution_[0].Lt);
    dv::gaussian_2D_convolution(evolution_[0].Lt, evolution_[0].Lt, 0, 0, options_.soffset);
    evolution_[0].Lt.copyTo(evolution_[0].Lsmooth);
        
    // Allocate memory for the flow and step images
    cv::Mat Lflow = cv::Mat::zeros(evolution_[0].Lt.rows, evolution_[0].Lt.cols, CV_32F);
    cv::Mat Lstep = cv::Mat::zeros(evolution_[0].Lt.rows, evolution_[0].Lt.cols, CV_32F);
        
    // First compute the kcontrast factor
    options_.kcontrast = dv::compute_k_percentile(img, options_.kcontrast_percentile, 1.0f, options_.kcontrast_nbins, 0, 0);
        
    // Now generate the rest of evolution levels
    for (size_t i = 1; i < evolution_.size(); i++)
    {
        if (evolution_[i].octave > evolution_[i - 1].octave)
        {
            dv::halfsample_image(evolution_[i - 1].Lt, evolution_[i].Lt);
            options_.kcontrast = options_.kcontrast*0.75f;
                
            // Allocate memory for the resized flow and step images
            Lflow = cv::Mat::zeros(evolution_[i].Lt.rows, evolution_[i].Lt.cols, CV_32F);
            Lstep = cv::Mat::zeros(evolution_[i].Lt.rows, evolution_[i].Lt.cols, CV_32F);
        }
        else
        {
            evolution_[i - 1].Lt.copyTo(evolution_[i].Lt);
        }
            
        dv::gaussian_2D_convolution(evolution_[i].Lt, evolution_[i].Lsmooth, 0, 0, 1.0f);
            
        // Compute the Gaussian derivatives Lx and Ly
        dv::image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Lx, 1, 0);
        dv::image_derivatives_scharr(evolution_[i].Lsmooth, evolution_[i].Ly, 0, 1);
            
        // Compute the conductivity equation
        switch (options_.diffusivity)
        {
        case cv::KAZE::DIFF_PM_G1:
            dv::pm_g1(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
            break;
        case cv::KAZE::DIFF_PM_G2:
            dv::pm_g2(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
            break;
        case cv::KAZE::DIFF_WEICKERT:
            dv::weickert_diffusivity(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
            break;
        case cv::KAZE::DIFF_CHARBONNIER:
            dv::charbonnier_diffusivity(evolution_[i].Lx, evolution_[i].Ly, Lflow, options_.kcontrast);
            break;
        default:
            CV_Error(options_.diffusivity, "Diffusivity is not supported");
            break;
        }
            
        // Perform FED n inner steps
        for (int j = 0; j < nsteps_[i - 1]; j++)
        {
            dv::nld_step_scalar(evolution_[i].Lt, Lflow, Lstep, tsteps_[i - 1][j]);
        }

        if(1)
        {
            std::cout << "FED: " << i << std::endl;
            cv::Mat canvas;
            cv::normalize(evolution_[i].Lt, canvas, 0, 1, cv::NORM_MINMAX, CV_32F);
            cv::imshow("canvas", canvas);
            cv::waitKey(0);
        }
    }
}
    
cv::Mat1f DiffusionFilter::operator()(const cv::Mat1f &img, cv::Mat1f &dst)
{
    Create_Nonlinear_Scale_Space(img);
    return dst;
}
