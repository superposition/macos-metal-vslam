#include "vslam/MetalLumaProcessor.hpp"

#include <mach-o/dyld.h>

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace vslam {

namespace {

std::string ExecutableDirectory() {
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::vector<char> buffer(size + 1, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
    return ".";
  }
  std::error_code error;
  const auto resolved = std::filesystem::weakly_canonical(buffer.data(), error);
  if (error) {
    return std::filesystem::path(buffer.data()).parent_path().string();
  }
  return resolved.parent_path().string();
}

std::string DefaultMetallibPath() {
  return (std::filesystem::path(ExecutableDirectory()) / "luma.metallib").string();
}

}  // namespace

struct MetalLumaProcessor::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> pipeline = nil;
};

MetalLumaProcessor::MetalLumaProcessor() : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (impl_->device == nil) {
      last_error_ = "Metal device unavailable.";
      return;
    }

    impl_->queue = [impl_->device newCommandQueue];
    if (impl_->queue == nil) {
      last_error_ = "Failed to create Metal command queue.";
      return;
    }

    const std::string metallib_path = DefaultMetallibPath();
    NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:path];
    NSError* error = nil;
    impl_->library = [impl_->device newLibraryWithURL:url error:&error];
    if (impl_->library == nil) {
      last_error_ = error == nil ? "Failed to load luma.metallib."
                                 : std::string([[error localizedDescription] UTF8String]);
      return;
    }

    id<MTLFunction> function = [impl_->library newFunctionWithName:@"bgrToLuma"];
    if (function == nil) {
      last_error_ = "Could not find Metal function bgrToLuma.";
      return;
    }

    impl_->pipeline = [impl_->device newComputePipelineStateWithFunction:function error:&error];
    if (impl_->pipeline == nil) {
      last_error_ = error == nil ? "Failed to create Metal compute pipeline."
                                 : std::string([[error localizedDescription] UTF8String]);
      return;
    }

    last_error_.clear();
  }
}

MetalLumaProcessor::~MetalLumaProcessor() = default;

MetalLumaProcessor::MetalLumaProcessor(MetalLumaProcessor&&) noexcept = default;
MetalLumaProcessor& MetalLumaProcessor::operator=(MetalLumaProcessor&&) noexcept = default;

bool MetalLumaProcessor::isAvailable() const {
  return impl_ != nullptr && impl_->pipeline != nil;
}

const std::string& MetalLumaProcessor::lastError() const {
  return last_error_;
}

bool MetalLumaProcessor::convertBgrToGray(const cv::Mat& bgr, cv::Mat& gray) {
  if (!isAvailable()) {
    return false;
  }

  if (bgr.empty() || bgr.type() != CV_8UC3) {
    last_error_ = "Expected a non-empty CV_8UC3 frame.";
    return false;
  }

  const cv::Mat contiguous = bgr.isContinuous() ? bgr : bgr.clone();
  gray.create(bgr.rows, bgr.cols, CV_8UC1);

  struct Dimensions {
    uint32_t width;
    uint32_t height;
  } dims{static_cast<uint32_t>(bgr.cols), static_cast<uint32_t>(bgr.rows)};

  @autoreleasepool {
    const NSUInteger input_bytes = static_cast<NSUInteger>(contiguous.total() * contiguous.elemSize());
    const NSUInteger output_bytes = static_cast<NSUInteger>(gray.total() * gray.elemSize());

    id<MTLBuffer> input_buffer = [impl_->device newBufferWithBytes:contiguous.data
                                                            length:input_bytes
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [impl_->device newBufferWithLength:output_bytes
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> dims_buffer = [impl_->device newBufferWithBytes:&dims
                                                           length:sizeof(Dimensions)
                                                          options:MTLResourceStorageModeShared];
    if (input_buffer == nil || output_buffer == nil || dims_buffer == nil) {
      last_error_ = "Failed to allocate Metal buffers.";
      return false;
    }

    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:impl_->pipeline];
    [encoder setBuffer:input_buffer offset:0 atIndex:0];
    [encoder setBuffer:output_buffer offset:0 atIndex:1];
    [encoder setBuffer:dims_buffer offset:0 atIndex:2];

    const NSUInteger thread_count =
        std::min<NSUInteger>(impl_->pipeline.maxTotalThreadsPerThreadgroup, 256);
    const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(dims.width) * dims.height, 1, 1);
    const MTLSize threadgroup_size = MTLSizeMake(thread_count, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      last_error_ = "Metal command buffer failed to complete.";
      return false;
    }

    std::memcpy(gray.data, [output_buffer contents], output_bytes);
    last_error_.clear();
    return true;
  }
}

}  // namespace vslam
