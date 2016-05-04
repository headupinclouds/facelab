#ifndef DIFFUSION_FILTER
#define DIFFUSION_FILTER

#include "facelab/nldiffusion_functions.h"
#include "facelab/fed.h"

#include "Filter.h"

#include <opencv2/features2d.hpp>

class DiffusionFilter
{
    /// AKAZE configuration options structure
    struct AKAZEOptions
    {
        AKAZEOptions()
        : omax(2)
        , nsublevels(10)
        , img_width(0)
        , img_height(0)
        , soffset(1.6) // 10) //1.6f)
        , diffusivity(cv::KAZE::DIFF_WEICKERT)
        , kcontrast(0.001f)
        , kcontrast_percentile(0.5f)
        , kcontrast_nbins(300)
        {
        }
        
        int omax;                       ///< Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
        int nsublevels;                 ///< Default number of sublevels per scale level
        int img_width;                  ///< Width of the input image
        int img_height;                 ///< Height of the input image
        float soffset;                  ///< Base scale offset (sigma units)
        int diffusivity;                ///< Diffusivity type
        
        float kcontrast;                ///< The contrast factor parameter
        float kcontrast_percentile;     ///< Percentile level for the contrast factor
        int kcontrast_nbins;            ///< Number of bins for the contrast factor histogram
    };
    
    /// FED parameters
    int ncycles_;                               ///< Number of cycles
    bool reordering_ = false;                   ///< Flag for reordering time steps
    std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
    std::vector<int> nsteps_;                   ///< Vector of number of steps per cycle

public:
    DiffusionFilter(int level_height, int level_width);
    
    void Allocate_Memory_Evolution(void);
    
    virtual cv::Mat operator()(const cv::Mat &img, cv::Mat &dst);
    
    void Create_Nonlinear_Scale_Space(const cv::Mat& img);
    
    virtual cv::Mat1f operator()(const cv::Mat1f &img, cv::Mat1f &dst);
    
    std::vector<dv::TEvolution> evolution_;  ///< Vector of nonlinear diffusion evolution
    
    AKAZEOptions options_;
};

#endif // DIFFUSION_FILTER
