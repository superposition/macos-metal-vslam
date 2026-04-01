#include <memory>
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
         << "    keypoints: " << stats.keypoints << "    matches: " << stats.matches
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

SCNGeometry* MakePointCloudGeometry(const std::vector<cv::Point3d>& points) {
  if (points.empty()) {
    return nil;
  }

  std::vector<SCNVector3> vertices;
  vertices.reserve(points.size());
  for (const auto& point : points) {
    vertices.push_back(SCNVector3Make(static_cast<float>(point.x), static_cast<float>(point.y),
                                      static_cast<float>(point.z)));
  }

  SCNGeometrySource* source =
      [SCNGeometrySource geometrySourceWithVertices:vertices.data() count:vertices.size()];

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
  element.pointSize = 6.0;
  element.minimumPointScreenSpaceRadius = 3.0;
  element.maximumPointScreenSpaceRadius = 8.0;

  SCNGeometry* geometry = [SCNGeometry geometryWithSources:@[ source ] elements:@[ element ]];
  geometry.firstMaterial.lightingModelName = SCNLightingModelConstant;
  geometry.firstMaterial.diffuse.contents = NSColor.systemOrangeColor;
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
  geometry.firstMaterial.diffuse.contents = NSColor.systemGreenColor;
  return geometry;
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
  NSTextView* transform_text_view_;
  NSTimer* timer_;
  std::unique_ptr<cv::VideoCapture> capture_;
  vslam::MonocularTracker tracker_;
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

  NSTextField* title_label = MakeLabel(@"Hello vSLAM", [NSFont boldSystemFontOfSize:30.0]);
  title_label.frame = NSMakeRect(24.0, 844.0, 400.0, 36.0);
  title_label.autoresizingMask = NSViewMaxXMargin | NSViewMinYMargin;
  [content_view addSubview:title_label];

  subtitle_label_ = MakeLabel(@"Native AppKit shell, C++ tracking core, Metal preprocessing",
                              [NSFont systemFontOfSize:14.0 weight:NSFontWeightMedium]);
  subtitle_label_.frame = NSMakeRect(24.0, 816.0, 700.0, 24.0);
  subtitle_label_.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
  subtitle_label_.textColor = [NSColor secondaryLabelColor];
  [content_view addSubview:subtitle_label_];

  NSBox* camera_box = [[NSBox alloc] initWithFrame:NSMakeRect(24.0, 320.0, 560.0, 470.0)];
  camera_box.title = @"Camera";
  camera_box.autoresizingMask = NSViewMaxXMargin | NSViewHeightSizable;
  [content_view addSubview:camera_box];

  image_view_ = [[NSImageView alloc] initWithFrame:NSMakeRect(18.0, 18.0, 524.0, 420.0)];
  image_view_.imageScaling = NSImageScaleProportionallyUpOrDown;
  image_view_.imageAlignment = NSImageAlignCenter;
  image_view_.wantsLayer = YES;
  image_view_.layer.backgroundColor = NSColor.windowBackgroundColor.CGColor;
  image_view_.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  [camera_box.contentView addSubview:image_view_];

  NSBox* transform_box = [[NSBox alloc] initWithFrame:NSMakeRect(606.0, 560.0, 790.0, 230.0)];
  transform_box.title = @"Transforms";
  transform_box.autoresizingMask = NSViewWidthSizable | NSViewMinYMargin;
  [content_view addSubview:transform_box];

  transform_text_view_ = MakeTextView(NSMakeRect(16.0, 12.0, 758.0, 186.0));
  transform_text_view_.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  transform_text_view_.string = @"T_world_camera\n1.000  0.000  0.000  0.000\n0.000  1.000  0.000  0.000\n0.000  0.000  1.000  0.000\n0.000  0.000  0.000  1.000\n";
  [transform_box.contentView addSubview:transform_text_view_];

  NSBox* geometry_box = [[NSBox alloc] initWithFrame:NSMakeRect(606.0, 148.0, 790.0, 390.0)];
  geometry_box.title = @"3D Reconstruction";
  geometry_box.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  [content_view addSubview:geometry_box];

  scene_view_ = [[SCNView alloc] initWithFrame:NSMakeRect(16.0, 16.0, 758.0, 340.0)];
  scene_view_.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
  scene_view_.allowsCameraControl = YES;
  scene_view_.backgroundColor = [NSColor colorWithRed:0.08 green:0.09 blue:0.11 alpha:1.0];
  [geometry_box.contentView addSubview:scene_view_];

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
  [scene.rootNode addChildNode:floor];

  point_cloud_node_ = [SCNNode node];
  trajectory_node_ = [SCNNode node];
  tracked_camera_node_ = [SCNNode nodeWithGeometry:[SCNPyramid pyramidWithWidth:0.10
                                                                          height:0.08
                                                                          length:0.14]];
  tracked_camera_node_.geometry.firstMaterial.diffuse.contents = NSColor.systemRedColor;
  [scene.rootNode addChildNode:point_cloud_node_];
  [scene.rootNode addChildNode:trajectory_node_];
  [scene.rootNode addChildNode:tracked_camera_node_];

  status_label_ = MakeLabel(@"Opening camera…", [NSFont monospacedSystemFontOfSize:13.0
                                                                             weight:NSFontWeightRegular]);
  status_label_.frame = NSMakeRect(24.0, 36.0, 1372.0, 40.0);
  status_label_.lineBreakMode = NSLineBreakByTruncatingTail;
  status_label_.autoresizingMask = NSViewWidthSizable | NSViewMaxYMargin;
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
  transform_text_view_.string = MatrixString(tracked.stats.pose_matrix);
  point_cloud_node_.geometry = MakePointCloudGeometry(tracked.world_points);
  trajectory_node_.geometry = MakeLineGeometry(tracked.trajectory_points);
  tracked_camera_node_.transform = SceneMatrixFromPose(tracked.stats.pose_matrix);
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
