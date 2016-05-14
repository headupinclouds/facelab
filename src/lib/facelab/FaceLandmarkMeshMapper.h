// Copyright (c) 2016, David Hirvonen
// All rights reserved.

#ifndef FACE_LANDMARK_MESH_MAPPER_H
#define FACE_LANDMARK_MESH_MAPPER_H 1

// experimental eos stuff
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/render/Mesh.hpp"

#include "opencv2/core/core.hpp"

#include <iostream>
#include <memory>

class FaceLandmarkMeshMapper
{
public:

    struct Impl;
    
    FaceLandmarkMeshMapper(const std::string &modelfile, const std::string &mappingsfile);
    
    cv::Point3f operator()(const eos::core::LandmarkCollection<cv::Vec2f> &landmarks, const cv::Mat &image, eos::render::Mesh &mesh, cv::Mat &isomap);

    cv::Point3f operator()(const std::vector<cv::Point2f> &landmarks, const cv::Mat &image, eos::render::Mesh &mesh, cv::Mat &isomap);
    
    static void save(const eos::render::Mesh &mesh, const std::string &filename);

protected:

    std::shared_ptr<Impl> m_pImpl;
};

#endif // FACE_LANDMARK_MESH_MAPPER_H 
