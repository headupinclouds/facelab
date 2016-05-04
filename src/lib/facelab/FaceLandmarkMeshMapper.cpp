#include "facelab/FaceLandmarkMeshMapper.h"

#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/fitting/nonlinear_camera_estimation.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"

static cv::Point2f interpolate(const cv::Point2f &p, const cv::Point2f &q, float f);
static eos::core::LandmarkCollection<cv::Vec2f> convertLandmarks(const std::vector<cv::Point2f> &points);

struct FaceLandmarkMeshMapper::Impl
{
    Impl(const std::string &modelfile, const std::string &mappingsfile)    
    {
        morphable_model = eos::morphablemodel::load_model(modelfile);
        landmark_mapper = mappingsfile.empty() ? eos::core::LandmarkMapper() : eos::core::LandmarkMapper(mappingsfile);
    }

    cv::Point3f operator()(const eos::core::LandmarkCollection<cv::Vec2f> &landmarks, const cv::Mat &image, eos::render::Mesh &mesh, cv::Mat &isomap)
    {
        // These will be the final 2D and 3D points used for the fitting:
        std::vector<cv::Vec4f> model_points; // the points in the 3D shape model
        std::vector<int> vertex_indices; // their vertex indices
        std::vector<cv::Vec2f> image_points; // the corresponding 2D landmark points
        
        // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
        for (int i = 0; i < landmarks.size(); ++i)
        {
            auto converted_name = landmark_mapper.convert(landmarks[i].name);
            if (!converted_name)
            { // no mapping defined for the current landmark
                continue;
            }
            int vertex_idx = std::stoi(converted_name.get());
            cv::Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
            model_points.emplace_back(vertex);
            vertex_indices.emplace_back(vertex_idx);
            image_points.emplace_back(landmarks[i].coordinates);
        }
        
        // Estimate the camera (pose) from the 2D - 3D point correspondences
        eos::fitting::RenderingParameters rendering_params = eos::fitting::estimate_orthographic_camera(image_points, model_points, image.cols, image.rows);
        cv::Mat affine_from_ortho = get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
        
        // The 3D head pose can be recovered as follows:
        float yaw = glm::degrees(rendering_params.r_y);
        float pitch = glm::degrees(rendering_params.r_x);
        float roll = glm::degrees(rendering_params.r_z);
        
        // std::cout << "Y,P,R = (" << yaw << " " << pitch << " " << roll << ")" << std::endl;
        
        // Estimate the shape coefficients by fitting the shape to the landmarks:
        auto fitted_coeffs = eos::fitting::fit_shape_to_landmarks_linear(morphable_model, affine_from_ortho, image_points, vertex_indices);
        
        // Obtain the full mesh with the estimated coefficients:
        mesh = morphable_model.draw_sample(fitted_coeffs, std::vector<float>());
        
        // Extract the texture from the image using given mesh and camera parameters:
        if(!image.empty())
        {
            isomap = eos::render::extract_texture(mesh, affine_from_ortho, image);
        }

        return cv::Point3f(pitch, yaw, roll);
    }
    
    cv::Point3f operator()(const std::vector<cv::Point2f> &points, const cv::Mat &image, eos::render::Mesh &mesh, cv::Mat &isomap)
    {
        return (*this)(convertLandmarks(points), image, mesh, isomap);
    }

    eos::morphablemodel::MorphableModel morphable_model;
    eos::core::LandmarkMapper landmark_mapper;    
};

FaceLandmarkMeshMapper::FaceLandmarkMeshMapper(const std::string &modelfile, const std::string &mappingsfile)
{
    m_pImpl = std::make_shared<Impl>(modelfile, mappingsfile);
}
    
cv::Point3f FaceLandmarkMeshMapper::operator()(const eos::core::LandmarkCollection<cv::Vec2f> &landmarks, const cv::Mat &image, eos::render::Mesh &mesh, cv::Mat &isomap)
{
    return (*m_pImpl)(landmarks, image, mesh, isomap);
}
    
cv::Point3f FaceLandmarkMeshMapper::operator()(const std::vector<cv::Point2f> &points, const cv::Mat &image, eos::render::Mesh &mesh, cv::Mat &isomap)
{
    return (*m_pImpl)(points, image, mesh, isomap);
}

void FaceLandmarkMeshMapper::save(const eos::render::Mesh &mesh, const std::string &filename)
{
    // Save the mesh as textured obj:
    eos::render::write_textured_obj(mesh, filename);
}

/// ===== UTILITY  ====

static cv::Point2f interpolate(const cv::Point2f &p, const cv::Point2f &q, float f)
{
    return p + (q - p) * f;
}

static eos::core::LandmarkCollection<cv::Vec2f> convertLandmarks(const std::vector<cv::Point2f> &points)
{
    int ibugId = 1;
    eos::core::LandmarkCollection<cv::Vec2f> landmarks;
    for(const auto &p : points)
    {
        eos::core::Landmark<cv::Vec2f> landmark;
        landmark.name = std::to_string(ibugId++);
        landmark.coordinates = { p.x, p.y };
        landmarks.emplace_back(landmark);
    }
    
    return landmarks;
}
