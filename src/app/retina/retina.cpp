#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/bioinspired/bioinspired.hpp>

// http://docs.opencv.org/3.1.0/d2/d94/bioinspired_retina.html#gsc.tab=0

int process(int argc, char **argv)
{
    cv::Mat inputFrame = cv::imread(argv[1], cv::IMREAD_COLOR);
    
    // create a retina instance with default parameters setup, uncomment the initialisation you wanna test
    bool colorModel = true;
    int colorSamplingMethod= cv::bioinspired::RETINA_COLOR_BAYER;
    const bool useRetinaLogSampling=false;
    const float reductionFactor=1.0f;
    const float samplingStrength=10.0f;
    auto myRetina = cv::bioinspired::createRetina(inputFrame.size(), colorModel, colorSamplingMethod, useRetinaLogSampling, reductionFactor, samplingStrength);
    
    // save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup"
    //myRetina->write("RetinaDefaultParameters.xml");

    // load parameters if file exists
    myRetina->setup("RetinaSpecificParameters.xml");

    // reset all retina buffers (imagine you close your eyes for a long time)
    myRetina->clearBuffers();

    // declare retina output buffers
    cv::Mat retinaOutput_parvo;
    cv::Mat retinaOutput_magno;

    // processing loop with no stop condition
    // run retina filter on the loaded input frame
    myRetina->run(inputFrame);
    
    // Retrieve and display retina output
    myRetina->getParvo(retinaOutput_parvo);
    myRetina->getMagno(retinaOutput_magno);
    //cv::imshow("retina input", inputFrame);
    cv::imshow("Retina Parvo", retinaOutput_parvo);
    cv::imshow("Retina Magno", retinaOutput_magno);

    cv::Mat hsi, channels[3];
    cv::cvtColor(inputFrame, hsi, cv::COLOR_BGR2HSV_FULL);
    cv::split(hsi, channels);
    channels[2] = retinaOutput_magno;
    cv::merge(channels, 3, hsi);
    cv::cvtColor(hsi, inputFrame, cv::COLOR_HSV2BGR_FULL);
    cv::imshow("post", inputFrame);
    
    cv::waitKey(0);
    
    return 0;
}

int main(int argc, char **argv)
{
    return process(argc, argv);
}
