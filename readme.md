## Stereo Webcam Demo
### 1. Stereo Camera Calibration 
    python stereo_calibration.py
---
#### There are two ways to do stereo camera calibrate on this repository:
 1. Load the stereo images from folder [uncompleted]
 2. Use D435 load the left frame and right frame directly.
---
#### Demo UI Info: "[o] 2 2"
**first num =  2** :  left frame number of detecting chessboard.
 <br> **sencond num = 2** : right frame num of detecting chessboard.
 <br> **state = [o]** : detect chessboard corners in both left frame and right frame.
 <br> **state = [x]** : one of frames not be detected chessboard corners.

<br>When info show** [o]** you can continusely press **'S'** util the info chane the number of count.
<br>When you wanna finsh to detect chessboard, you can press **'Q'** to break. The calibrate will show in Info and create the pkl file on the folder.

----
### 2. Stereo Camera Rectify and Create Depth Frame.
    python stereo_rectify.py

#### Demo statues:
- [x] 2*D435
- [x] 2*Image Files
- [x] 2*Fisheyes (unstable)

----
### Demo:
<div>
<img src="./demo/rgbFrames.gif" width="300" height="360"/>
<img src="./demo/depth.gif" width="200" height="360"/>
</div>
<img src="./demo/d435.jpg" width="369" height="492"/>
