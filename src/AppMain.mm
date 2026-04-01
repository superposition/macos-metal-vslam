#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#import <AppKit/AppKit.h>
#import <SceneKit/SceneKit.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "vslam/MonocularTracker.hpp"

namespace {

NSImage* MatToImage(const cv::Mat& bgr) {
  if (bgr.empty()) {
    return nil;
  }

  cv::Mat rgba;
  cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);

  NSData* data =
      [NSData dataWithBytes:rgba.data length:static_cast<NSUInteger>(rgba.total() * rgba.elemSize())];
  CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
  CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
  const CGBitmapInfo bitmap_info =
      static_cast<CGBitmapInfo>(kCGImageAlphaLast) | kCGBitmapByteOrderDefault;
  CGImageRef image_ref = CGImageCreate(
      static_cast<size_t>(rgba.cols), static_cast<size_t>(rgba.rows), 8, 32,
      static_cast<size_t>(rgba.step[0]), color_space, bitmap_info, provider, nullptr, false,
      kCGRenderingIntentDefault);
  NSImage* image =
      [[NSImage alloc] initWithCGImage:image_ref size:NSMakeSize(rgba.cols, rgba.rows)];
  CGImageRelease(image_ref);
  CGColorSpaceRelease(color_space);
  CGDataProviderRelease(provider);
  return image;
}

NSString* StatusString(const vslam::TrackingStats& stats, const std::string& metal_error) {
  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(3);
  stream << "status: " << stats.status << "    metal: " << (stats.metal_enabled ? "on" : "cpu")
         << "    keypoints: " << stats.keypoints << "    keyframes: " << stats.keyframes
         << "    matches: " << stats.matches
         << "    inliers: " << stats.inliers << "    map: " << stats.map_points
         << "    travel: " << stats.translation_norm;
  if (!metal_error.empty() && !stats.metal_enabled) {
    stream << "    metal-note: " << metal_error;
  }
  return [NSString stringWithUTF8String:stream.str().c_str()];
}

NSString* MatrixString(const cv::Matx44d& matrix) {
  std::ostringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(3);
  stream << "T_world_camera\n";
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      stream.width(8);
      stream << matrix(row, col);
      if (col < 3) {
        stream << "  ";
      }
    }
    stream << '\n';
  }
  return [NSString stringWithUTF8String:stream.str().c_str()];
}

NSTextField* MakeLabel(NSString* text, NSFont* font) {
  NSTextField* label = [[NSTextField alloc] initWithFrame:NSZeroRect];
  label.stringValue = text;
  label.editable = NO;
  label.bezeled = NO;
  label.drawsBackground = NO;
  label.selectable = NO;
  label.font = font;
  label.textColor = NSColor.labelColor;
  return label;
}

NSColor* DashboardBackgroundColor() {
  return [NSColor colorWithRed:0.06 green:0.08 blue:0.10 alpha:1.0];
}

NSColor* CardBackgroundColor() {
  return [NSColor colorWithRed:0.10 green:0.13 blue:0.16 alpha:1.0];
}

NSColor* CardBorderColor() {
  return [NSColor colorWithRed:0.20 green:0.26 blue:0.30 alpha:1.0];
}

NSColor* AccentColor() {
  return [NSColor colorWithRed:0.38 green:0.93 blue:0.62 alpha:1.0];
}

NSFont* DashboardFont(NSString* name, CGFloat size, NSFont* fallback) {
  NSFont* font = [NSFont fontWithName:name size:size];
  if (font != nil) {
    return font;
  }
  return fallback;
}

NSView* MakeCardView(NSRect frame) {
  NSView* view = [[NSView alloc] initWithFrame:frame];
  view.wantsLayer = YES;
  view.layer.backgroundColor = CardBackgroundColor().CGColor;
  view.layer.borderColor = CardBorderColor().CGColor;
  view.layer.borderWidth = 1.0;
  view.layer.cornerRadius = 18.0;
  return view;
}

NSTextField* MakeValueLabel(NSString* text, CGFloat size, NSColor* color) {
  NSTextField* label = MakeLabel(text, DashboardFont(@"Avenir Next Demi Bold", size,
                                                     [NSFont systemFontOfSize:size
                                                                        weight:NSFontWeightSemibold]));
  label.textColor = color;
  return label;
}

NSImageView* MakeImageSurface(NSRect frame) {
  NSImageView* image_view = [[NSImageView alloc] initWithFrame:frame];
  image_view.imageScaling = NSImageScaleProportionallyUpOrDown;
  image_view.imageAlignment = NSImageAlignCenter;
  image_view.wantsLayer = YES;
  image_view.layer.backgroundColor = [NSColor colorWithRed:0.08 green:0.09 blue:0.11 alpha:1.0].CGColor;
  image_view.layer.borderColor = CardBorderColor().CGColor;
  image_view.layer.borderWidth = 1.0;
  image_view.layer.cornerRadius = 14.0;
  image_view.layer.masksToBounds = YES;
  return image_view;
}

NSView* MakeMetricCard(NSRect frame, NSString* title, NSTextField* __strong* value_label_out) {
  NSView* card = MakeCardView(frame);

  NSTextField* title_label = MakeLabel(title,
                                       DashboardFont(@"Avenir Next Medium", 12.0,
                                                     [NSFont systemFontOfSize:12.0
                                                                        weight:NSFontWeightMedium]));
  title_label.frame = NSMakeRect(16.0, frame.size.height - 30.0, frame.size.width - 32.0, 16.0);
  title_label.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
  title_label.textColor = [NSColor colorWithRed:0.60 green:0.67 blue:0.73 alpha:1.0];
  [card addSubview:title_label];

  NSTextField* value_label = MakeValueLabel(@"--", 24.0, AccentColor());
  value_label.frame = NSMakeRect(16.0, 16.0, frame.size.width - 32.0, 34.0);
  value_label.autoresizingMask = NSViewWidthSizable | NSViewMaxYMargin;
  value_label.lineBreakMode = NSLineBreakByTruncatingTail;
  [card addSubview:value_label];

  if (value_label_out != nullptr) {
    *value_label_out = value_label;
  }
  return card;
}

NSTextView* MakeTextView(NSRect frame) {
  NSTextView* text_view = [[NSTextView alloc] initWithFrame:frame];
  text_view.editable = NO;
  text_view.selectable = YES;
  text_view.drawsBackground = NO;
  text_view.font = [NSFont monospacedSystemFontOfSize:13.0 weight:NSFontWeightRegular];
  text_view.textColor = NSColor.labelColor;
  text_view.verticallyResizable = YES;
  text_view.horizontallyResizable = NO;
  return text_view;
}

SCNMatrix4 SceneMatrixFromPose(const cv::Matx44d& pose) {
  SCNMatrix4 matrix = SCNMatrix4Identity;
  matrix.m11 = static_cast<float>(pose(0, 0));
  matrix.m12 = static_cast<float>(pose(0, 1));
  matrix.m13 = static_cast<float>(pose(0, 2));
  matrix.m14 = static_cast<float>(pose(0, 3));
  matrix.m21 = static_cast<float>(pose(1, 0));
  matrix.m22 = static_cast<float>(pose(1, 1));
  matrix.m23 = static_cast<float>(pose(1, 2));
  matrix.m24 = static_cast<float>(pose(1, 3));
  matrix.m31 = static_cast<float>(pose(2, 0));
  matrix.m32 = static_cast<float>(pose(2, 1));
  matrix.m33 = static_cast<float>(pose(2, 2));
  matrix.m34 = static_cast<float>(pose(2, 3));
  matrix.m41 = static_cast<float>(pose(3, 0));
  matrix.m42 = static_cast<float>(pose(3, 1));
  matrix.m43 = static_cast<float>(pose(3, 2));
  matrix.m44 = static_cast<float>(pose(3, 3));
  return matrix;
}

SCNGeometry* MakePointCloudGeometry(const std::vector<vslam::ColoredPoint>& points) {
  if (points.empty()) {
    return nil;
  }

  std::vector<SCNVector3> vertices;
  std::vector<SCNVector4> colors;
  vertices.reserve(points.size());
  colors.reserve(points.size());
  for (const auto& point : points) {
    vertices.push_back(SCNVector3Make(static_cast<float>(point.position.x),
                                      static_cast<float>(point.position.y),
                                      static_cast<float>(point.position.z)));
    colors.push_back(SCNVector4Make(static_cast<float>(point.bgr[2]) / 255.0f,
                                    static_cast<float>(point.bgr[1]) / 255.0f,
                                    static_cast<float>(point.bgr[0]) / 255.0f, 1.0f));
  }

  SCNGeometrySource* vertex_source =
      [SCNGeometrySource geometrySourceWithVertices:vertices.data() count:vertices.size()];
  NSData* color_data = [NSData dataWithBytes:colors.data() length:colors.size() * sizeof(SCNVector4)];
  SCNGeometrySource* color_source =
      [SCNGeometrySource geometrySourceWithData:color_data
                                       semantic:SCNGeometrySourceSemanticColor
                                    vectorCount:colors.size()
                                floatComponents:YES
                              componentsPerVector:4
                                bytesPerComponent:sizeof(float)
                                      dataOffset:0
                                      dataStride:sizeof(SCNVector4)];

  std::vector<uint32_t> indices(vertices.size());
  for (uint32_t index = 0; index < indices.size(); ++index) {
    indices[index] = index;
  }
  NSData* index_data =
      [NSData dataWithBytes:indices.data() length:indices.size() * sizeof(uint32_t)];
  SCNGeometryElement* element =
      [SCNGeometryElement geometryElementWithData:index_data
                                     primitiveType:SCNGeometryPrimitiveTypePoint
                                    primitiveCount:indices.size()
                                     bytesPerIndex:sizeof(uint32_t)];
  element.pointSize = 10.0;
  element.minimumPointScreenSpaceRadius = 6.0;
  element.maximumPointScreenSpaceRadius = 16.0;

  SCNGeometry* geometry =
      [SCNGeometry geometryWithSources:@[ vertex_source, color_source ] elements:@[ element ]];
  geometry.firstMaterial.lightingModelName = SCNLightingModelConstant;
  geometry.firstMaterial.diffuse.contents = NSColor.whiteColor;
  geometry.firstMaterial.doubleSided = YES;
  geometry.firstMaterial.readsFromDepthBuffer = NO;
  geometry.firstMaterial.writesToDepthBuffer = NO;
  return geometry;
}

SCNGeometry* MakeLineGeometry(const std::vector<cv::Point3d>& trajectory) {
  if (trajectory.size() < 2) {
    return nil;
  }

  std::vector<SCNVector3> vertices;
  vertices.reserve(trajectory.size());
  for (const auto& point : trajectory) {
    vertices.push_back(SCNVector3Make(static_cast<float>(point.x), static_cast<float>(point.y),
                                      static_cast<float>(point.z)));
  }

  SCNGeometrySource* source =
      [SCNGeometrySource geometrySourceWithVertices:vertices.data() count:vertices.size()];

  std::vector<uint32_t> indices;
  indices.reserve((trajectory.size() - 1) * 2);
  for (uint32_t index = 1; index < trajectory.size(); ++index) {
    indices.push_back(index - 1);
    indices.push_back(index);
  }

  NSData* index_data =
      [NSData dataWithBytes:indices.data() length:indices.size() * sizeof(uint32_t)];
  SCNGeometryElement* element =
      [SCNGeometryElement geometryElementWithData:index_data
                                     primitiveType:SCNGeometryPrimitiveTypeLine
                                    primitiveCount:trajectory.size() - 1
                                     bytesPerIndex:sizeof(uint32_t)];

  SCNGeometry* geometry = [SCNGeometry geometryWithSources:@[ source ] elements:@[ element ]];
  geometry.firstMaterial.lightingModelName = SCNLightingModelConstant;
  geometry.firstMaterial.diffuse.contents = [NSColor colorWithRed:0.18 green:0.92 blue:0.42 alpha:1.0];
  return geometry;
}

struct VisualizationBounds {
  SCNVector3 center = SCNVector3Make(0.0f, 0.0f, 0.0f);
  float scale = 1.0f;
};

VisualizationBounds MakeVisualizationBounds(const vslam::TrackingFrame& tracked) {
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  bool has_geometry = false;

  auto extend = [&](double x, double y, double z) {
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    min_z = std::min(min_z, z);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
    max_z = std::max(max_z, z);
    has_geometry = true;
  };

  for (const auto& point : tracked.world_points) {
    extend(point.position.x, point.position.y, point.position.z);
  }
  for (const auto& point : tracked.trajectory_points) {
    extend(point.x, point.y, point.z);
  }
  const cv::Matx44d& pose = tracked.stats.pose_matrix;
  extend(pose(0, 3), pose(1, 3), pose(2, 3));

  if (!has_geometry) {
    return {};
  }

  const double span_x = std::max(1e-3, max_x - min_x);
  const double span_y = std::max(1e-3, max_y - min_y);
  const double span_z = std::max(1e-3, max_z - min_z);
  const double max_span = std::max(span_x, std::max(span_y, span_z));

  VisualizationBounds bounds;
  bounds.center = SCNVector3Make(static_cast<float>(0.5 * (min_x + max_x)),
                                 static_cast<float>(0.5 * (min_y + max_y)),
                                 static_cast<float>(0.5 * (min_z + max_z)));
  bounds.scale = static_cast<float>(std::clamp(2.2 / max_span, 0.08, 12.0));
  return bounds;
}

}  // namespace

@interface AppController : NSObject <NSApplicationDelegate>
@end

@implementation AppController {
  NSWindow* window_;
  NSImageView* image_view_;
  SCNView* scene_view_;
  NSTextField* status_label_;
  NSTextField* subtitle_label_;
  NSTextField* tracking_value_label_;
  NSTextField* backend_value_label_;
  NSTextField* keypoints_value_label_;
  NSTextField* matches_value_label_;
  NSTextField* inliers_value_label_;
  NSTextField* keyframes_value_label_;
  NSTextField* cloud_value_label_;
  NSTextField* travel_value_label_;
  NSTextView* transform_text_view_;
  NSImageView* map_overview_view_;
  NSTimer* timer_;
  std::unique_ptr<cv::VideoCapture> capture_;
  vslam::MonocularTracker tracker_;
  SCNNode* world_root_node_;
  SCNNode* point_cloud_node_;
  SCNNode* trajectory_node_;
  SCNNode* tracked_camera_node_;
}

- (instancetype)init {
  self = [super init];
  return self;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)sender {
  (void)sender;
  return YES;
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
  (void)notification;

  const NSRect frame = NSMakeRect(0.0, 0.0, 1420.0, 920.0);
  window_ = [[NSWindow alloc] initWithContentRect:frame
                                        styleMask:(NSWindowStyleMaskTitled |
                                                   NSWindowStyleMaskClosable |
                                                   NSWindowStyleMaskMiniaturizable |
                                                   NSWindowStyleMaskResizable)
                                          backing:NSBackingStoreBuffered
                                            defer:NO];
  window_.title = @"macos-metal-vslam";
  [window_ center];

  NSView* content_view = window_.contentView;
  content_view.wantsLayer = YES;
  content_view.layer.backgroundColor = DashboardBackgroundColor().CGColor;

  NSTextField* title_label = MakeLabel(@"MacBook Webcam vSLAM",
                                       DashboardFont(@"Avenir Next Bold", 32.0,
                                                     [NSFont boldSystemFontOfSize:32.0]));
  title_label.frame = NSMakeRect(24.0, 854.0, 500.0, 36.0);
  title_label.autoresizingMask = NSViewMaxXMargin | NSViewMinYMargin;
  title_label.textColor = [NSColor colorWithRed:0.96 green:0.97 blue:0.98 alpha:1.0];
  [content_view addSubview:title_label];

  subtitle_label_ = MakeLabel(@"Native AppKit dashboard, C++ tracking core, Metal preprocessing, GTSAM iSAM2 pose graph",
                              [NSFont systemFontOfSize:14.0 weight:NSFontWeightMedium]);
  subtitle_label_.frame = NSMakeRect(24.0, 826.0, 1000.0, 24.0);
  subtitle_label_.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
  subtitle_label_.textColor = [NSColor colorWithRed:0.62 green:0.69 blue:0.75 alpha:1.0];
  [content_view addSubview:subtitle_label_];

  const CGFloat metric_y = 734.0;
  const CGFloat metric_width = 165.0;
  const CGFloat metric_height = 78.0;
  const CGFloat metric_gap = 10.0;
  const CGFloat metric_x0 = 24.0;

  NSArray<NSView*>* metric_cards = @[
    MakeMetricCard(NSMakeRect(metric_x0 + (metric_width + metric_gap) * 0, metric_y,
                              metric_width, metric_height),
                   @"Tracking", &tracking_value_label_),
    MakeMetricCard(NSMakeRect(metric_x0 + (metric_width + metric_gap) * 1, metric_y,
                              metric_width, metric_height),
                   @"Backend", &backend_value_label_),
    MakeMetricCard(NSMakeRect(metric_x0 + (metric_width + metric_gap) * 2, metric_y,
                              metric_width, metric_height),
                   @"Keypoints", &keypoints_value_label_),
    MakeMetricCard(NSMakeRect(metric_x0 + (metric_width + metric_gap) * 3, metric_y,
                              metric_width, metric_height),
                   @"Matches", &matches_value_label_),
    MakeMetricCard(NSMakeRect(metric_x0 + (metric_width + metric_gap) * 4, metric_y,
                              metric_width, metric_height),
                   @"Inliers", &inliers_value_label_),
    MakeMetricCard(NSMakeRect(metric_x0 + (metric_width + metric_gap) * 5, metric_y,
                              metric_width, metric_height),
                   @"Keyframes", &keyframes_value_label_),
    MakeMetricCard(NSMakeRect(metric_x0 + (metric_width + metric_gap) * 6, metric_y,
                              metric_width, metric_height),
                   @"Cloud", &cloud_value_label_),
    MakeMetricCard(NSMakeRect(metric_x0 + (metric_width + metric_gap) * 7, metric_y,
                              metric_width, metric_height),
                   @"Travel", &travel_value_label_)
  ];
  for (NSView* card in metric_cards) {
    card.autoresizingMask = NSViewMaxXMargin | NSViewMinYMargin;
    [content_view addSubview:card];
  }

  NSView* camera_card = MakeCardView(NSMakeRect(24.0, 244.0, 560.0, 470.0));
  camera_card.autoresizingMask = NSViewMaxXMargin | NSViewHeightSizable;
  [content_view addSubview:camera_card];

  NSTextField* camera_title = MakeValueLabel(@"Camera Feed", 20.0,
                                             [NSColor colorWithRed:0.94 green:0.95 blue:0.98 alpha:1.0]);
  camera_title.frame = NSMakeRect(20.0, 428.0, 240.0, 28.0);
  camera_title.autoresizingMask = NSViewMaxXMargin | NSViewMinYMargin;
  [camera_card addSubview:camera_title];

  NSTextField* camera_caption = MakeLabel(@"Live webcam input with tracked feature overlays",
                                          DashboardFont(@"Avenir Next Medium", 13.0,
                                                        [NSFont systemFontOfSize:13.0
                                                                           weight:NSFontWeightMedium]));
  camera_caption.frame = NSMakeRect(20.0, 404.0, 360.0, 18.0);
  camera_caption.textColor = [NSColor colorWithRed:0.60 green:0.67 blue:0.73 alpha:1.0];
  camera_caption.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
  [camera_card addSubview:camera_caption];

  image_view_ = MakeImageSurface(NSMakeRect(20.0, 20.0, 520.0, 372.0));
  image_view_.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  [camera_card addSubview:image_view_];

  NSView* geometry_card = MakeCardView(NSMakeRect(606.0, 244.0, 790.0, 470.0));
  geometry_card.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  [content_view addSubview:geometry_card];

  NSTextField* geometry_title = MakeValueLabel(@"3D Reconstruction", 20.0,
                                               [NSColor colorWithRed:0.94 green:0.95 blue:0.98 alpha:1.0]);
  geometry_title.frame = NSMakeRect(20.0, 428.0, 300.0, 28.0);
  geometry_title.autoresizingMask = NSViewMaxXMargin | NSViewMinYMargin;
  [geometry_card addSubview:geometry_title];

  NSTextField* geometry_caption =
      MakeLabel(@"Interactive SceneKit viewport for the colored sparse map and camera pose",
                DashboardFont(@"Avenir Next Medium", 13.0,
                              [NSFont systemFontOfSize:13.0 weight:NSFontWeightMedium]));
  geometry_caption.frame = NSMakeRect(20.0, 404.0, 520.0, 18.0);
  geometry_caption.textColor = [NSColor colorWithRed:0.60 green:0.67 blue:0.73 alpha:1.0];
  geometry_caption.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
  [geometry_card addSubview:geometry_caption];

  scene_view_ = [[SCNView alloc] initWithFrame:NSMakeRect(20.0, 20.0, 750.0, 372.0)];
  scene_view_.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  scene_view_.allowsCameraControl = YES;
  scene_view_.backgroundColor = [NSColor colorWithRed:0.08 green:0.09 blue:0.11 alpha:1.0];
  scene_view_.wantsLayer = YES;
  scene_view_.layer.cornerRadius = 14.0;
  scene_view_.layer.masksToBounds = YES;
  scene_view_.layer.borderWidth = 1.0;
  scene_view_.layer.borderColor = CardBorderColor().CGColor;
  [geometry_card addSubview:scene_view_];

  NSView* map_card = MakeCardView(NSMakeRect(24.0, 90.0, 560.0, 136.0));
  map_card.autoresizingMask = NSViewMaxXMargin | NSViewMinYMargin;
  [content_view addSubview:map_card];

  NSTextField* map_title = MakeValueLabel(@"Map Overview", 18.0,
                                          [NSColor colorWithRed:0.94 green:0.95 blue:0.98 alpha:1.0]);
  map_title.frame = NSMakeRect(20.0, 98.0, 220.0, 24.0);
  map_title.autoresizingMask = NSViewMaxXMargin | NSViewMinYMargin;
  [map_card addSubview:map_title];

  NSTextField* map_caption = MakeLabel(@"Top-down diagnostic view of the recovered geometry",
                                       DashboardFont(@"Avenir Next Medium", 12.0,
                                                     [NSFont systemFontOfSize:12.0
                                                                        weight:NSFontWeightMedium]));
  map_caption.frame = NSMakeRect(20.0, 78.0, 320.0, 16.0);
  map_caption.textColor = [NSColor colorWithRed:0.60 green:0.67 blue:0.73 alpha:1.0];
  map_caption.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
  [map_card addSubview:map_caption];

  map_overview_view_ = MakeImageSurface(NSMakeRect(360.0, 16.0, 180.0, 104.0));
  map_overview_view_.autoresizingMask = NSViewMinXMargin | NSViewHeightSizable;
  [map_card addSubview:map_overview_view_];

  NSView* transform_card = MakeCardView(NSMakeRect(606.0, 90.0, 790.0, 136.0));
  transform_card.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
  [content_view addSubview:transform_card];

  NSTextField* transform_title = MakeValueLabel(@"Transforms", 18.0,
                                                [NSColor colorWithRed:0.94 green:0.95 blue:0.98 alpha:1.0]);
  transform_title.frame = NSMakeRect(20.0, 98.0, 220.0, 24.0);
  transform_title.autoresizingMask = NSViewMaxXMargin | NSViewMinYMargin;
  [transform_card addSubview:transform_title];

  NSTextField* transform_caption = MakeLabel(@"Current world-to-camera pose matrix",
                                             DashboardFont(@"Avenir Next Medium", 12.0,
                                                           [NSFont systemFontOfSize:12.0
                                                                              weight:NSFontWeightMedium]));
  transform_caption.frame = NSMakeRect(20.0, 78.0, 260.0, 16.0);
  transform_caption.textColor = [NSColor colorWithRed:0.60 green:0.67 blue:0.73 alpha:1.0];
  transform_caption.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
  [transform_card addSubview:transform_caption];

  transform_text_view_ = MakeTextView(NSMakeRect(20.0, 14.0, 750.0, 56.0));
  transform_text_view_.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  transform_text_view_.string = @"T_world_camera\n1.000  0.000  0.000  0.000\n0.000  1.000  0.000  0.000\n0.000  0.000  1.000  0.000\n0.000  0.000  0.000  1.000\n";
  [transform_card addSubview:transform_text_view_];

  SCNScene* scene = [SCNScene scene];
  scene_view_.scene = scene;

  SCNNode* observer = [SCNNode node];
  observer.camera = [SCNCamera camera];
  observer.position = SCNVector3Make(0.0f, 1.2f, 3.5f);
  [scene.rootNode addChildNode:observer];
  scene_view_.pointOfView = observer;

  SCNNode* ambient = [SCNNode node];
  ambient.light = [SCNLight light];
  ambient.light.type = SCNLightTypeAmbient;
  ambient.light.color = [NSColor colorWithWhite:0.55 alpha:1.0];
  [scene.rootNode addChildNode:ambient];

  SCNNode* omni = [SCNNode node];
  omni.light = [SCNLight light];
  omni.light.type = SCNLightTypeOmni;
  omni.position = SCNVector3Make(2.5f, 3.5f, 3.0f);
  [scene.rootNode addChildNode:omni];

  SCNNode* floor = [SCNNode nodeWithGeometry:[SCNFloor floor]];
  floor.geometry.firstMaterial.diffuse.contents = [NSColor colorWithWhite:0.18 alpha:1.0];
  floor.geometry.firstMaterial.lightingModelName = SCNLightingModelConstant;
  floor.position = SCNVector3Make(0.0f, -1.0f, 0.0f);
  [scene.rootNode addChildNode:floor];

  world_root_node_ = [SCNNode node];
  point_cloud_node_ = [SCNNode node];
  trajectory_node_ = [SCNNode node];
  tracked_camera_node_ = [SCNNode nodeWithGeometry:[SCNPyramid pyramidWithWidth:0.10
                                                                          height:0.08
                                                                          length:0.14]];
  tracked_camera_node_.geometry.firstMaterial.diffuse.contents = NSColor.systemRedColor;
  [scene.rootNode addChildNode:world_root_node_];
  [world_root_node_ addChildNode:point_cloud_node_];
  [world_root_node_ addChildNode:trajectory_node_];
  [world_root_node_ addChildNode:tracked_camera_node_];

  status_label_ = MakeLabel(@"Opening camera…",
                            [NSFont monospacedSystemFontOfSize:13.0
                                                         weight:NSFontWeightRegular]);
  status_label_.frame = NSMakeRect(24.0, 32.0, 1372.0, 32.0);
  status_label_.lineBreakMode = NSLineBreakByTruncatingTail;
  status_label_.autoresizingMask = NSViewWidthSizable | NSViewMaxYMargin;
  status_label_.textColor = [NSColor colorWithRed:0.65 green:0.72 blue:0.78 alpha:1.0];
  [content_view addSubview:status_label_];

  [window_ makeKeyAndOrderFront:nil];
  [NSApp activateIgnoringOtherApps:YES];

  capture_ = std::make_unique<cv::VideoCapture>(0, cv::CAP_AVFOUNDATION);
  if (!capture_->isOpened()) {
    status_label_.stringValue =
        @"Camera open failed. Grant camera permission in System Settings and relaunch.";
    return;
  }

  capture_->set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  capture_->set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  capture_->set(cv::CAP_PROP_FPS, 30);

  timer_ = [NSTimer scheduledTimerWithTimeInterval:(1.0 / 30.0)
                                            target:self
                                          selector:@selector(tick:)
                                          userInfo:nil
                                           repeats:YES];
  timer_.tolerance = 0.01;
}

- (void)applicationWillTerminate:(NSNotification*)notification {
  (void)notification;
  [timer_ invalidate];
  timer_ = nil;
  capture_.reset();
}

- (void)tick:(NSTimer*)timer {
  (void)timer;
  if (!capture_ || !capture_->isOpened()) {
    status_label_.stringValue = @"Camera unavailable.";
    return;
  }

  cv::Mat frame;
  if (!capture_->read(frame) || frame.empty()) {
    status_label_.stringValue = @"Failed to read from the camera.";
    return;
  }

  const vslam::TrackingFrame tracked = tracker_.process(frame);
  image_view_.image = MatToImage(tracked.display_bgr.empty() ? frame : tracked.display_bgr);
  map_overview_view_.image = MatToImage(tracked.geometry_bgr);
  transform_text_view_.string = MatrixString(tracked.stats.pose_matrix);
  point_cloud_node_.geometry = MakePointCloudGeometry(tracked.world_points);
  trajectory_node_.geometry = MakeLineGeometry(tracked.trajectory_points);
  tracked_camera_node_.transform = SceneMatrixFromPose(tracked.stats.pose_matrix);
  const VisualizationBounds bounds = MakeVisualizationBounds(tracked);
  world_root_node_.scale = SCNVector3Make(bounds.scale, bounds.scale, bounds.scale);
  world_root_node_.position = SCNVector3Make(-bounds.center.x * bounds.scale,
                                             -bounds.center.y * bounds.scale,
                                             -bounds.center.z * bounds.scale);
  tracking_value_label_.stringValue = [NSString stringWithUTF8String:tracked.stats.status.c_str()];
  backend_value_label_.stringValue = tracked.stats.metal_enabled ? @"Metal + iSAM2" : @"CPU + iSAM2";
  keypoints_value_label_.stringValue = [NSString stringWithFormat:@"%d", tracked.stats.keypoints];
  matches_value_label_.stringValue = [NSString stringWithFormat:@"%d", tracked.stats.matches];
  inliers_value_label_.stringValue = [NSString stringWithFormat:@"%d", tracked.stats.inliers];
  keyframes_value_label_.stringValue = [NSString stringWithFormat:@"%d", tracked.stats.keyframes];
  cloud_value_label_.stringValue = [NSString stringWithFormat:@"%d", tracked.stats.map_points];
  travel_value_label_.stringValue = [NSString stringWithFormat:@"%.2f", tracked.stats.translation_norm];
  status_label_.stringValue = StatusString(tracked.stats, "");
}

@end

int main(int argc, const char* argv[]) {
  (void)argc;
  (void)argv;
  @autoreleasepool {
    NSApplication* application = [NSApplication sharedApplication];
    application.activationPolicy = NSApplicationActivationPolicyRegular;
    AppController* delegate = [[AppController alloc] init];
    application.delegate = delegate;
    [application run];
  }
  return 0;
}
