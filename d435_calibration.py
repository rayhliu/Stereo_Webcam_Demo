#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:10:36 2020

@author: rayhliu
"""


import pyrealsense2 as rs
import numpy as np
import cv2
import pickle

CHECKERBOARD = (9,7)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('832112072526')
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

imgpoints = []
objpoints = []
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)



def getChessBoard(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        corners = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    return ret,corners

try:
    while True:
        print (">>>",len(imgpoints))
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color_image = np.rot90(color_image,3)
        
        ret,corners = getChessBoard(color_image)
        
        if ret == True:
            color_image = cv2.drawChessboardCorners(color_image,CHECKERBOARD, corners,ret)
            
        cv2.namedWindow('demo',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('demo',(540,960))
        cv2.imshow('demo', color_image)
        if cv2.waitKey(32) & 0xff == ord('q'):
            break
        if cv2.waitKey(32) & 0xff == ord('s'):
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
                
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, color_image.shape[::-1][1:],None,None)
    print ('retval:',retval)
    print ('cameraMatrix:',cameraMatrix)
    print ('distCoeffs:',distCoeffs)
    print ('rvecs:',rvecs)
    print ('tvecs:',tvecs)
    camera_param_dict = {'retval':retval,
                        'cameraMatrix':cameraMatrix,
                        'distCoeffs':distCoeffs,
                        'rvecs':rvecs,
                        'tvecs':tvecs}
    
    with open('./cmaera_caliParam.pkl','wb') as file:
        pickle.dump(camera_param_dict, file)
        
finally:
    pipeline.stop()