# macos-metal-vslam

A native macOS hello-world for monocular vSLAM on Apple Silicon.

This project is intentionally small and honest:

- `AppKit` provides the native Mac window and camera preview shell.
- `C++` owns the tracking core.
- `Metal` accelerates grayscale conversion in the frontend.
- `SceneKit` renders a live 3D view of the recovered sparse world and camera trajectory.
- `OpenCV` handles camera capture, feature extraction, and two-view geometry.
- `GTSAM` runs an incremental `iSAM2` keyframe pose graph so the sparse map and trajectory can be refined without restarting the whole solve.

The current implementation is a vSLAM starter, not a finished production SLAM system. It tracks ORB features from your built-in camera, estimates relative motion with the essential matrix, bootstraps sparse map points by triangulation, and shows live tracking stats, transforms, and a 3D reconstruction panel in a native macOS window.

## Why this stack

For a Mac-only vSLAM app, the cleanest architecture is:

- native `AppKit` UI instead of a cross-platform widget toolkit
- Objective-C++ only at the platform boundary
- a portable C++ core for the actual tracking logic
- Metal where GPU acceleration is most straightforward in the frontend

That keeps the repo native to Apple Silicon without locking the SLAM logic into an Apple-only code path.

## Features

- native macOS window with live camera feed
- transform panel showing the current camera pose matrix
- SceneKit viewport for a colored sparse world point cloud and camera trajectory
- ORB feature extraction and matching
- monocular relative pose estimation with OpenCV
- lightweight keyframe-based triangulation that preserves webcam color on world points
- incremental `GTSAM` / `iSAM2` pose-graph updates across keyframes
- sparse triangulation count for map bootstrap feedback
- Metal-accelerated BGR-to-luma preprocessing with CPU fallback
- CMake build for Apple Silicon

## Prerequisites

- macOS on Apple Silicon
- Xcode command line tools
- Homebrew
- OpenCV
- GTSAM installed locally under `.deps/gtsam-install`

Install dependencies:

```bash
brew install opencv
```

The current local build expects a GTSAM install at `.deps/gtsam-install`. `run.sh` automatically adds that prefix if it exists.

## Build

```bash
./run.sh
```

The first launch may prompt for camera access.

## Notes

- Pose scale is arbitrary because this is monocular tracking.
- Camera intrinsics are approximated from image size. For serious tracking, replace that with real calibration.
- The tracker estimates pose on every frame, but only promotes occasional high-parallax frames to keyframes for reconstruction so the webcam UI stays responsive.
- The Metal path currently accelerates image preprocessing. The backend uses OpenCV for tracking and GTSAM `iSAM2` for incremental keyframe optimization.

## Roadmap

- calibrated camera model loading
- stronger multi-keyframe landmark management
- bundle adjustment on local windows
- Metal acceleration for pyramid construction and descriptor preprocessing
- relocalization and loop closure

## License

MIT
