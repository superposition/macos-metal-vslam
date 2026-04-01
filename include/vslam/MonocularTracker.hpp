#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "vslam/MetalLumaProcessor.hpp"

namespace vslam {

struct ColoredPoint {
  cv::Point3d position;
  cv::Vec3b bgr = cv::Vec3b(255, 255, 255);
};

struct TrackingStats {
  bool initialized = false;
  bool pose_updated = false;
  bool metal_enabled = false;
  int keypoints = 0;
  int keyframes = 0;
  int matches = 0;
  int inliers = 0;
  int map_points = 0;
  double translation_norm = 0.0;
  cv::Matx44d pose_matrix = cv::Matx44d::eye();
  std::string status;
};

struct TrackingFrame {
  cv::Mat display_bgr;
  cv::Mat geometry_bgr;
  std::vector<ColoredPoint> world_points;
  std::vector<cv::Point3d> trajectory_points;
  TrackingStats stats;
};

class MonocularTracker {
 public:
  explicit MonocularTracker(bool prefer_metal = true);
  ~MonocularTracker();

  TrackingFrame process(const cv::Mat& frame_bgr);
  bool usingMetal() const;

 private:
  struct FactorGraphState;

  struct Keyframe {
    int frame_index = 0;
    int graph_index = -1;
    cv::Matx44d T_w_c_initial = cv::Matx44d::eye();
    cv::Matx44d T_w_c_optimized = cv::Matx44d::eye();
    cv::Mat image_bgr;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
  };

  struct AnchoredColoredPoint {
    int keyframe_index = -1;
    cv::Point3d position_in_keyframe;
    cv::Vec3b bgr = cv::Vec3b(255, 255, 255);
  };

  void initializeIntrinsics(const cv::Size& frame_size);
  void seedKeyframe(const cv::Mat& frame_bgr, const std::vector<cv::KeyPoint>& keypoints,
                    const cv::Mat& descriptors, const cv::Matx44d& T_w_c);
  void addKeyframe(const cv::Mat& frame_bgr, const std::vector<cv::KeyPoint>& keypoints,
                   const cv::Mat& descriptors, const cv::Matx44d& T_w_c);
  void optimizePoseGraph();
  void rebuildWorldGeometry();

  bool prefer_metal_ = true;
  bool intrinsics_initialized_ = false;
  int frame_index_ = 0;
  cv::Size frame_size_;
  cv::Matx33d camera_matrix_ = cv::Matx33d::eye();
  cv::Matx44d T_c_w_ = cv::Matx44d::eye();
  cv::Ptr<cv::ORB> orb_;
  cv::BFMatcher matcher_;
  MetalLumaProcessor metal_;
  cv::Mat prev_gray_;
  std::vector<cv::KeyPoint> prev_keypoints_;
  cv::Mat prev_descriptors_;
  int active_keyframe_index_ = -1;
  std::vector<Keyframe> keyframes_;
  std::unique_ptr<FactorGraphState> factor_graph_;
  std::vector<cv::Point3d> trajectory_history_;
  std::vector<AnchoredColoredPoint> map_points_local_;
  std::vector<ColoredPoint> map_points_world_;
};

}  // namespace vslam
