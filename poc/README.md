# USB webcam proof of concept

This PoC runs the application pipeline on a laptop before an FPGA board is
available:

```text
USB webcam/video
  -> marker detector
  -> ArUco ID decode
  -> distance or apparent-size filtering
  -> robot stop or round-trip mission state machine
  -> overlay + CSV benchmark
```

The displayed `speed` is a normalized simulated motor command. No serial/GPIO
command is sent to a physical robot.

Round-trip mode adds separate normalized linear/angular commands. See
[ROUNDTRIP_DEMO.md](ROUNDTRIP_DEMO.md) for the complete two-marker workflow.

## 1. Print a marker

The default dictionary is `DICT_4X4_50`. Print marker ID 0 at a known physical
size, for example 10 cm.

## 2. Immediate demo without a trained CNN

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --stop-side-px 180 --slow-side-px 100
```

This mode proves the webcam, ArUco, UI, logging and robot-stop logic. It uses
marker apparent size as a distance proxy.

Controls:

- `Q` or `Esc`: quit
- `R`: release the latched STOP and restart
- `S`: save an annotated snapshot

Results are written to `artifacts/webcam_poc/metrics.csv`.

## 3. Calibrate approximate distance

Place a marker of known size at a measured distance from the camera. For a
10 cm marker positioned 1 m away:

```powershell
python poc/calibrate_focal_length.py --source 0 --marker-id 0 `
  --marker-size-m 0.10 --distance-m 1.0
```

Use the reported focal length:

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --marker-size-m 0.10 --focal-length-px 615.2 `
  --slow-distance-m 0.90 --stop-distance-m 0.45
```

This pinhole estimate is sufficient for a PoC. Thesis pose experiments should
use a camera matrix, distortion coefficients and `estimatePoseSingleMarkers`
or `solvePnP`.

### Full intrinsic calibration and solvePnP

Use a checkerboard with 9×6 inner corners and known square size:

```powershell
python poc/calibrate_camera.py --source 0 `
  --board-cols 9 --board-rows 6 --square-size-m 0.025 `
  --samples 20 --output artifacts/calibration/camera.npz
```

Then run:

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --marker-size-m 0.10 `
  --camera-calibration artifacts/calibration/camera.npz
```

CSV adds `distance_method`, `pose_x_m`, `pose_y_m` and `pose_z_m`. When a
calibration file is supplied, solvePnP takes precedence over the focal-only
approximation.

## 4. CNN and hybrid modes

CNN-only:

```powershell
python poc/live_webcam_demo.py --source 0 --mode cnn `
  --checkpoint artifacts/aruco_mbv2_035/best.pt --target-id 0
```

Hybrid:

```powershell
python poc/live_webcam_demo.py --source 0 --mode hybrid `
  --checkpoint artifacts/aruco_mbv2_035/best.pt --target-id 0
```

Hybrid first runs the MobileNetV2-0.35 ROI detector. If no decoded marker is
returned, it falls back to full-frame OpenCV ArUco detection. The CSV
`target_source` column records `cnn+opencv` or `opencv`, so fallback usage can
be measured rather than hidden.

Do not use a smoke-test checkpoint as an accuracy result. Train on reviewed
real-camera data before evaluating CNN or hybrid mode.

## 5. Offline demo without a webcam

Generate a marker approaching the camera:

```powershell
python poc/make_demo_video.py --marker-id 0 `
  --output artifacts/webcam_poc/aruco_approach.avi
```

Run the complete PoC headlessly:

```powershell
python poc/live_webcam_demo.py `
  --source artifacts/webcam_poc/aruco_approach.avi `
  --mode classical --target-id 0 --headless `
  --output-video artifacts/webcam_poc/result.avi
```

FPS measured from a file in headless mode is not a camera real-time result:
video files are consumed as fast as possible. Report FPS from the USB webcam
run under a fixed resolution, backend and display configuration.

## 6. Round-trip demo: start -> target -> start

Run the validated end-to-end workflow with one command:

```powershell
python poc/run_roundtrip_workflow.py
```

Or run the individual stages:

```powershell
python poc/make_roundtrip_demo_video.py --start-id 0 --target-id 1 `
  --output artifacts/webcam_poc/roundtrip_input.avi

python poc/live_webcam_demo.py `
  --source artifacts/webcam_poc/roundtrip_input.avi `
  --mode classical --mission roundtrip --start-id 0 --target-id 1 `
  --headless --output-video artifacts/webcam_poc/roundtrip_result.avi `
  --csv artifacts/webcam_poc/roundtrip_metrics.csv
```

The expected terminal state is `HOME_COMPLETE`. For a webcam, use `--source 0`
and remove `--headless`. The turn is frame-timed in this PoC; calibrate
`--turn-frames` on a physical robot or replace it with encoder/IMU feedback.

## Logged fields

The CSV contains:

- vision and total latency;
- exponentially smoothed FPS;
- detected IDs;
- detector source;
- distance or marker side length;
- state (`SEARCHING`, `APPROACHING`, `SLOWING`, `STOPPED`);
- round-trip states (`WAITING_FOR_START` through `HOME_COMPLETE`);
- start/target visibility, distance, pose and marker side;
- active marker ID and mission milestone flags;
- simulated normalized linear and angular speed.
