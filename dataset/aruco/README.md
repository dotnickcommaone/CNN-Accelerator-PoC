# ArUco dataset location

This directory is intentionally kept free of generated images in Git.

Generate the initial synthetic dataset with:

```powershell
python model/scripts/generate_synthetic_aruco.py --output dataset/aruco --count 1000
```

The generator creates:

```text
dataset/aruco/
├── images/
├── labels/
├── train.txt
├── val.txt
└── test.txt
```

Synthetic data is only the starting point. Record a separate real-camera
dataset and do not place near-identical frames from one video sequence into
different splits, as that would leak test information into training.

Real-camera collection helper:

```powershell
python model/scripts/collect_aruco_dataset.py --source 0 `
  --output dataset/aruco --session room_a_daylight --display
```
