# Camera Calibration

## Introduction
This project is to calibrate a camera using a set of chessboard images. The calibration process is done in two steps:

1. **Camera Calibration**: This step is to find the camera matrix and distortion coefficients. The camera matrix is used to project 3D points to 2D image points. The distortion coefficients are used to correct the distortion in the image.

2. **Get Camera rotation matrix**: This step is to find the rotation matrix of the camera. The rotation matrix is used to rotate the camera to the world coordinate system.

## Tools

### Convert JPG to MP4
To convert a series of JPG images into an MP4 video, follow these steps:

1. Navigate to the `jpgtomp4` folder.
2. Place the folder containing your JPG images inside the `jpgtomp4` directory.
3. In the `jpgtomp4` folder, run the following command to generate the MP4 video:
```
# example
python main.py --input_folder camera_1 --output_video chessboard_1.mp4 --frame_rate 30
```