#include "vslam/MonocularTracker.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace vslam {

namespace {

cv::Matx44d ComposePose(const cv::Mat& rotation, const cv::Mat& translation) {
  cv::Matx44d pose = cv::Matx44d::eye();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      pose(row, col) = rotation.at<double>(row, col);
    }
    pose(row, 3) = translation.at<double>(row, 0);
  }
  return pose;
}

cv::Vec3b SampleColorAt(const cv::Mat& image_bgr, const cv::Point2f& point) {
  if (image_bgr.empty() || image_bgr.type() != CV_8UC3) {
    return cv::Vec3b(255, 255, 255);
  }

  const int x = std::clamp(static_cast<int>(std::lround(point.x)), 0, image_bgr.cols - 1);
  const int y = std::clamp(static_cast<int>(std::lround(point.y)), 0, image_bgr.rows - 1);
  return image_bgr.at<cv::Vec3b>(y, x);
}

std::vector<cv::DMatch> RatioTestMatches(const cv::BFMatcher& matcher,
                                         const cv::Mat& previous_descriptors,
                                         const cv::Mat& current_descriptors) {
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher.knnMatch(previous_descriptors, current_descriptors, knn_matches, 2);

  std::vector<cv::DMatch> filtered_matches;
  filtered_matches.reserve(knn_matches.size());
  for (const auto& pair : knn_matches) {
    if (pair.size() < 2) {
      continue;
    }
    if (pair[0].distance < 0.75f * pair[1].distance) {
      filtered_matches.push_back(pair[0]);
    }
  }
  return filtered_matches;
}

void GatherMatchedPoints(const std::vector<cv::KeyPoint>& source_keypoints,
                         const std::vector<cv::KeyPoint>& target_keypoints,
                         const std::vector<cv::DMatch>& matches,
                         std::vector<cv::Point2f>& source_points,
                         std::vector<cv::Point2f>& target_points) {
  source_points.clear();
  target_points.clear();
  source_points.reserve(matches.size());
  target_points.reserve(matches.size());
  for (const auto& match : matches) {
    source_points.push_back(source_keypoints[match.queryIdx].pt);
    target_points.push_back(target_keypoints[match.trainIdx].pt);
  }
}

double MedianPixelMotion(const std::vector<cv::Point2f>& previous_points,
                         const std::vector<cv::Point2f>& current_points) {
  if (previous_points.size() != current_points.size() || previous_points.empty()) {
    return 0.0;
  }

  std::vector<double> motions;
  motions.reserve(previous_points.size());
  for (size_t i = 0; i < previous_points.size(); ++i) {
    motions.push_back(cv::norm(current_points[i] - previous_points[i]));
  }
  const size_t midpoint = motions.size() / 2;
  std::nth_element(motions.begin(), motions.begin() + midpoint, motions.end());
  return motions[midpoint];
}

cv::Matx44d RelativePose(const cv::Matx44d& from_pose, const cv::Matx44d& to_pose) {
  return to_pose.inv() * from_pose;
}

cv::Matx33d RotationFromPose(const cv::Matx44d& pose) {
  cv::Matx33d rotation = cv::Matx33d::eye();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      rotation(row, col) = pose(row, col);
    }
  }
  return rotation;
}

cv::Vec3d TranslationFromPose(const cv::Matx44d& pose) {
  return cv::Vec3d(pose(0, 3), pose(1, 3), pose(2, 3));
}

bool ShouldCreateKeyframe(int frames_since_keyframe, double baseline, double median_pixel_motion,
                          int inliers) {
  if (inliers < 40) {
    return false;
  }
  if (frames_since_keyframe >= 18) {
    return true;
  }
  if (frames_since_keyframe < 6) {
    return false;
  }
  return baseline > 0.35 || median_pixel_motion > 36.0;
}

cv::Mat RenderGeometryView(const std::vector<ColoredPoint>& points,
                           const std::vector<cv::Point3d>& trajectory) {
  cv::Mat canvas(520, 640, CV_8UC3, cv::Scalar(20, 22, 28));
  cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols - 1, canvas.rows - 1),
                cv::Scalar(60, 66, 80), 1, cv::LINE_AA);
  cv::putText(canvas, "Recovered geometry / top-down trajectory", cv::Point(18, 28),
              cv::FONT_HERSHEY_DUPLEX, 0.65, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
  cv::putText(canvas, "x-z plane, arbitrary monocular scale", cv::Point(18, 52),
              cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(150, 160, 170), 1, cv::LINE_AA);

  constexpr int margin = 56;
  const cv::Rect plot(margin, 72, canvas.cols - (2 * margin), canvas.rows - 120);
  cv::rectangle(canvas, plot, cv::Scalar(52, 58, 72), 1, cv::LINE_AA);

  double min_x = -1.0;
  double max_x = 1.0;
  double min_z = -1.0;
  double max_z = 1.0;

  for (const auto& point : points) {
    min_x = std::min(min_x, point.position.x);
    max_x = std::max(max_x, point.position.x);
    min_z = std::min(min_z, point.position.z);
    max_z = std::max(max_z, point.position.z);
  }
  for (const auto& point : trajectory) {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_z = std::min(min_z, point.z);
    max_z = std::max(max_z, point.z);
  }

  const double span_x = std::max(1e-6, max_x - min_x);
  const double span_z = std::max(1e-6, max_z - min_z);
  const double scale =
      std::min(static_cast<double>(plot.width) / span_x, static_cast<double>(plot.height) / span_z) *
      0.85;
  const cv::Point2d center(plot.x + plot.width * 0.5, plot.y + plot.height * 0.5);

  auto project = [&](double x, double z) {
    return cv::Point(static_cast<int>(std::lround(center.x + x * scale)),
                     static_cast<int>(std::lround(center.y - z * scale)));
  };

  cv::line(canvas, cv::Point(plot.x, static_cast<int>(center.y)),
           cv::Point(plot.x + plot.width, static_cast<int>(center.y)),
           cv::Scalar(45, 90, 120), 1, cv::LINE_AA);
  cv::line(canvas, cv::Point(static_cast<int>(center.x), plot.y),
           cv::Point(static_cast<int>(center.x), plot.y + plot.height),
           cv::Scalar(45, 90, 120), 1, cv::LINE_AA);

  for (const auto& point : points) {
    const cv::Point projected = project(point.position.x, point.position.z);
    if (plot.contains(projected)) {
      const cv::Scalar color(point.bgr[0], point.bgr[1], point.bgr[2]);
      cv::circle(canvas, projected, 2, color, cv::FILLED, cv::LINE_AA);
    }
  }

  for (size_t index = 1; index < trajectory.size(); ++index) {
    const cv::Point a = project(trajectory[index - 1].x, trajectory[index - 1].z);
    const cv::Point b = project(trajectory[index].x, trajectory[index].z);
    cv::line(canvas, a, b, cv::Scalar(90, 240, 120), 2, cv::LINE_AA);
  }
  if (!trajectory.empty()) {
    const cv::Point current = project(trajectory.back().x, trajectory.back().z);
    cv::circle(canvas, current, 5, cv::Scalar(60, 90, 255), cv::FILLED, cv::LINE_AA);
  }

  return canvas;
}

void DrawTrackedPoints(cv::Mat& image,
                       const std::vector<cv::Point2f>& points,
                       const cv::Scalar& color,
                       int max_points) {
  const int count = std::min<int>(static_cast<int>(points.size()), max_points);
  for (int i = 0; i < count; ++i) {
    cv::circle(image, points[i], 2, color, cv::FILLED, cv::LINE_AA);
  }
}

std::vector<ColoredPoint> TriangulatePositiveDepthPoints(
    const cv::Matx33d& camera_matrix, const cv::Matx33d& rotation, const cv::Vec3d& translation,
    const cv::Mat& source_image_bgr,
    const std::vector<cv::Point2f>& previous_points,
    const std::vector<cv::Point2f>& current_points) {
  if (previous_points.size() < 8 || current_points.size() < 8) {
    return {};
  }

  const cv::Matx34d projection_prev(1.0, 0.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0);
  cv::Matx34d projection_curr(0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      projection_curr(row, col) = rotation(row, col);
    }
    projection_curr(row, 3) = translation[row];
  }

  cv::Mat homogeneous_points;
  cv::triangulatePoints(camera_matrix * projection_prev, camera_matrix * projection_curr,
                        previous_points, current_points, homogeneous_points);

  std::vector<ColoredPoint> positive_depth_points;
  positive_depth_points.reserve(std::min(homogeneous_points.cols, 1200));
  for (int col = 0; col < homogeneous_points.cols; ++col) {
    const double w = homogeneous_points.at<double>(3, col);
    if (std::abs(w) < 1e-9) {
      continue;
    }
    const double x = homogeneous_points.at<double>(0, col) / w;
    const double y = homogeneous_points.at<double>(1, col) / w;
    const double z = homogeneous_points.at<double>(2, col) / w;
    if (z <= 0.0) {
      continue;
    }

    const cv::Vec3d point_previous(x, y, z);
    const cv::Vec3d point_current = rotation * point_previous + translation;
    if (point_current[2] > 0.0) {
      const cv::Vec3d projection_prev_vec = camera_matrix * point_previous;
      const cv::Vec3d projection_curr_vec = camera_matrix * point_current;
      if (projection_prev_vec[2] <= 1e-9 || projection_curr_vec[2] <= 1e-9) {
        continue;
      }
      const cv::Point2f reproj_prev(static_cast<float>(projection_prev_vec[0] / projection_prev_vec[2]),
                                    static_cast<float>(projection_prev_vec[1] / projection_prev_vec[2]));
      const cv::Point2f reproj_curr(static_cast<float>(projection_curr_vec[0] / projection_curr_vec[2]),
                                    static_cast<float>(projection_curr_vec[1] / projection_curr_vec[2]));
      const double reproj_error =
          0.5 * (cv::norm(reproj_prev - previous_points[col]) + cv::norm(reproj_curr - current_points[col]));
      if (reproj_error > 2.5) {
        continue;
      }
      positive_depth_points.push_back({cv::Point3d(x, y, z), SampleColorAt(source_image_bgr, previous_points[col])});
    }
  }
  return positive_depth_points;
}

std::vector<ColoredPoint> TransformPointsToWorld(const std::vector<ColoredPoint>& camera_points,
                                                 const cv::Matx44d& T_w_c) {
  std::vector<ColoredPoint> world_points;
  world_points.reserve(camera_points.size());
  for (const auto& point : camera_points) {
    const cv::Vec4d homogeneous(point.position.x, point.position.y, point.position.z, 1.0);
    const cv::Vec4d world = T_w_c * homogeneous;
    world_points.push_back({cv::Point3d(world[0], world[1], world[2]), point.bgr});
  }
  return world_points;
}

std::string PoseSummary(const TrackingStats& stats) {
  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(3);
  stream << stats.status << " | metal: " << (stats.metal_enabled ? "on" : "off")
         << " | kp: " << stats.keypoints << " | kf: " << stats.keyframes
         << " | matches: " << stats.matches
         << " | inliers: " << stats.inliers << " | map: " << stats.map_points
         << " | travel: " << stats.translation_norm;
  return stream.str();
}

}  // namespace

MonocularTracker::MonocularTracker(bool prefer_metal)
    : prefer_metal_(prefer_metal),
      orb_(cv::ORB::create(2800)),
      matcher_(cv::NORM_HAMMING, false) {}

bool MonocularTracker::usingMetal() const {
  return prefer_metal_ && metal_.isAvailable();
}

void MonocularTracker::initializeIntrinsics(const cv::Size& frame_size) {
  frame_size_ = frame_size;
  const double focal = static_cast<double>(frame_size.width) * 0.9;
  camera_matrix_ = cv::Matx33d(focal, 0.0, static_cast<double>(frame_size.width) * 0.5,
                               0.0, focal, static_cast<double>(frame_size.height) * 0.5,
                               0.0, 0.0, 1.0);
  intrinsics_initialized_ = true;
  T_c_w_ = cv::Matx44d::eye();
  prev_gray_.release();
  prev_keypoints_.clear();
  prev_descriptors_.release();
  has_active_keyframe_ = false;
  map_points_world_.clear();
  trajectory_history_.clear();
  trajectory_history_.emplace_back(0.0, 0.0, 0.0);
  frame_index_ = 0;
}

TrackingFrame MonocularTracker::process(const cv::Mat& frame_bgr) {
  ++frame_index_;
  TrackingFrame result;
  if (frame_bgr.empty()) {
    result.stats.status = "No frame available.";
    result.geometry_bgr = RenderGeometryView({}, trajectory_history_);
    return result;
  }

  if (!intrinsics_initialized_ || frame_bgr.size() != frame_size_) {
    initializeIntrinsics(frame_bgr.size());
  }

  cv::Mat gray;
  bool metal_used = false;
  if (prefer_metal_ && metal_.isAvailable() && metal_.convertBgrToGray(frame_bgr, gray)) {
    metal_used = true;
  } else {
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
  }

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

  result.display_bgr = frame_bgr.clone();
  result.geometry_bgr = RenderGeometryView(map_points_world_, trajectory_history_);
  result.world_points = map_points_world_;
  result.trajectory_points = trajectory_history_;
  result.stats.initialized = !prev_descriptors_.empty();
  result.stats.metal_enabled = metal_used;
  result.stats.keypoints = static_cast<int>(keypoints.size());
  result.stats.keyframes = has_active_keyframe_ ? 1 : 0;
  result.stats.status = "Bootstrapping";
  result.stats.pose_matrix = T_c_w_.inv();

  std::vector<cv::Point2f> current_points;
  current_points.reserve(keypoints.size());
  for (const auto& keypoint : keypoints) {
    current_points.push_back(keypoint.pt);
  }
  DrawTrackedPoints(result.display_bgr, current_points, cv::Scalar(255, 200, 0), 250);

  if (prev_descriptors_.empty() || descriptors.empty() || prev_keypoints_.size() < 40 ||
      keypoints.size() < 40) {
    prev_gray_ = gray;
    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors.clone();
    if (!has_active_keyframe_ && !descriptors.empty() && keypoints.size() >= 80) {
      active_keyframe_.frame_index = frame_index_;
      active_keyframe_.T_w_c = T_c_w_.inv();
      active_keyframe_.image_bgr = frame_bgr.clone();
      active_keyframe_.keypoints = keypoints;
      active_keyframe_.descriptors = descriptors.clone();
      has_active_keyframe_ = true;
      result.stats.keyframes = 1;
      result.stats.status = "Seeded keyframe";
    }
    cv::putText(result.display_bgr, PoseSummary(result.stats), cv::Point(20, 32),
                cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(30, 255, 30), 2, cv::LINE_AA);
    return result;
  }

  const std::vector<cv::DMatch> matches =
      RatioTestMatches(matcher_, prev_descriptors_, descriptors);
  result.stats.matches = static_cast<int>(matches.size());
  if (matches.size() < 24) {
    result.stats.status = "Need more stable matches";
    prev_gray_ = gray;
    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors.clone();
    cv::putText(result.display_bgr, PoseSummary(result.stats), cv::Point(20, 32),
                cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(30, 255, 30), 2, cv::LINE_AA);
    return result;
  }

  std::vector<cv::Point2f> previous_points;
  current_points.clear();
  GatherMatchedPoints(prev_keypoints_, keypoints, matches, previous_points, current_points);

  cv::Mat inlier_mask;
  const cv::Mat essential =
      cv::findEssentialMat(previous_points, current_points, camera_matrix_, cv::RANSAC,
                           0.999, 1.0, inlier_mask);
  if (essential.empty()) {
    result.stats.status = "Essential matrix failed";
    prev_gray_ = gray;
    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors.clone();
    cv::putText(result.display_bgr, PoseSummary(result.stats), cv::Point(20, 32),
                cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(30, 255, 30), 2, cv::LINE_AA);
    return result;
  }

  cv::Mat rotation;
  cv::Mat translation;
  const int inlier_count = cv::recoverPose(essential, previous_points, current_points,
                                           camera_matrix_, rotation, translation, inlier_mask);
  result.stats.inliers = inlier_count;
  if (inlier_count < 16) {
    result.stats.status = "Pose recovery unstable";
    prev_gray_ = gray;
    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors.clone();
    cv::putText(result.display_bgr, PoseSummary(result.stats), cv::Point(20, 32),
                cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(30, 255, 30), 2, cv::LINE_AA);
    return result;
  }

  std::vector<cv::Point2f> previous_inliers;
  std::vector<cv::Point2f> current_inliers;
  previous_inliers.reserve(inlier_count);
  current_inliers.reserve(inlier_count);
  for (int index = 0; index < inlier_mask.rows; ++index) {
    if (inlier_mask.at<uchar>(index, 0) == 0) {
      continue;
    }
    previous_inliers.push_back(previous_points[index]);
    current_inliers.push_back(current_points[index]);
  }
  const cv::Matx44d delta_pose = ComposePose(rotation, translation);
  T_c_w_ = delta_pose * T_c_w_;
  const cv::Matx44d T_w_c = T_c_w_.inv();
  result.stats.pose_matrix = T_w_c;
  result.stats.translation_norm =
      cv::norm(cv::Vec3d(T_w_c(0, 3), T_w_c(1, 3), T_w_c(2, 3)));
  result.stats.pose_updated = true;
  result.stats.status = "Tracking";
  trajectory_history_.emplace_back(T_w_c(0, 3), T_w_c(1, 3), T_w_c(2, 3));
  if (trajectory_history_.size() > 512) {
    trajectory_history_.erase(trajectory_history_.begin(),
                              trajectory_history_.begin() + (trajectory_history_.size() - 512));
  }

  std::vector<ColoredPoint> triangulated_world_points;
  if (has_active_keyframe_ && !active_keyframe_.descriptors.empty()) {
    const std::vector<cv::DMatch> keyframe_matches =
        RatioTestMatches(matcher_, active_keyframe_.descriptors, descriptors);
    std::vector<cv::Point2f> keyframe_points;
    std::vector<cv::Point2f> current_keyframe_points;
    GatherMatchedPoints(active_keyframe_.keypoints, keypoints, keyframe_matches,
                        keyframe_points, current_keyframe_points);

    const cv::Matx44d T_c_k = RelativePose(active_keyframe_.T_w_c, T_w_c);
    const cv::Vec3d keyframe_translation = TranslationFromPose(T_c_k);
    const double keyframe_baseline = cv::norm(keyframe_translation);
    const double keyframe_motion = MedianPixelMotion(keyframe_points, current_keyframe_points);

    if (keyframe_matches.size() >= 32 && keyframe_baseline > 0.12 && keyframe_motion > 14.0) {
      const std::vector<ColoredPoint> keyframe_points_3d = TriangulatePositiveDepthPoints(
          camera_matrix_, RotationFromPose(T_c_k), keyframe_translation, active_keyframe_.image_bgr, keyframe_points,
          current_keyframe_points);
      triangulated_world_points =
          TransformPointsToWorld(keyframe_points_3d, active_keyframe_.T_w_c);
    }

    const int frames_since_keyframe = frame_index_ - active_keyframe_.frame_index;
    if (ShouldCreateKeyframe(frames_since_keyframe, keyframe_baseline, keyframe_motion, inlier_count)) {
      active_keyframe_.frame_index = frame_index_;
      active_keyframe_.T_w_c = T_w_c;
      active_keyframe_.image_bgr = frame_bgr.clone();
      active_keyframe_.keypoints = keypoints;
      active_keyframe_.descriptors = descriptors.clone();
      result.stats.status = "Tracking + keyframe";
    }
  } else {
    active_keyframe_.frame_index = frame_index_;
    active_keyframe_.T_w_c = T_w_c;
    active_keyframe_.image_bgr = frame_bgr.clone();
    active_keyframe_.keypoints = keypoints;
    active_keyframe_.descriptors = descriptors.clone();
    has_active_keyframe_ = true;
    result.stats.status = "Tracking + seeded keyframe";
  }

  if (!triangulated_world_points.empty()) {
    for (const auto& point : triangulated_world_points) {
      map_points_world_.push_back(point);
    }
  }
  if (map_points_world_.size() > 6000) {
    map_points_world_.erase(map_points_world_.begin(),
                            map_points_world_.begin() + (map_points_world_.size() - 6000));
  }
  result.stats.map_points = static_cast<int>(map_points_world_.size());
  result.stats.keyframes = has_active_keyframe_ ? 1 : 0;
  result.geometry_bgr = RenderGeometryView(map_points_world_, trajectory_history_);
  result.world_points = map_points_world_;
  result.trajectory_points = trajectory_history_;

  DrawTrackedPoints(result.display_bgr, current_inliers, cv::Scalar(40, 255, 40), 350);
  cv::putText(result.display_bgr, PoseSummary(result.stats), cv::Point(20, 32),
              cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(30, 255, 30), 2, cv::LINE_AA);

  prev_gray_ = gray;
  prev_keypoints_ = keypoints;
  prev_descriptors_ = descriptors.clone();
  return result;
}

}  // namespace vslam
