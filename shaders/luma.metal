#include <metal_stdlib>

using namespace metal;

kernel void bgrToLuma(device const uchar* input [[buffer(0)]],
                      device uchar* output [[buffer(1)]],
                      constant uint2& dimensions [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
  const uint pixel_count = dimensions.x * dimensions.y;
  if (index >= pixel_count) {
    return;
  }

  const uint offset = index * 3;
  const float b = static_cast<float>(input[offset + 0]);
  const float g = static_cast<float>(input[offset + 1]);
  const float r = static_cast<float>(input[offset + 2]);
  const float luma = 0.114f * b + 0.587f * g + 0.299f * r;
  output[index] = static_cast<uchar>(clamp(luma, 0.0f, 255.0f));
}
