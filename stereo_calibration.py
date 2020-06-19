#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:10:36 2020

@author: rayhliu
"""

import numpy as np
import cv2
import pickle
import glob
import os
import time

CHECKERBOARD = (9,7)

class StereoCalibration(object):
    def __init__(self,fisheyeCalibration=False):
        # termination criteria
        self.fisheyeCali = fisheyeCalibration
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

        self.objpFe = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        self.objpFe[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        self.objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
        self.imgShape = None
        

        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.objpointsFe = []
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.
        
    def _getChessBoard(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            corners = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        return ret,corners
    
    def run_stereo_image_on_chessboard(self,imgL,imgR):
        retR = False
        retL = False
        cornersL = None
        cornersR = None
        if imgL is not None and imgR is not None:
            retL,cornersL = self._getChessBoard(imgL)
            if retL: 
                cv2.drawChessboardCorners(imgL,CHECKERBOARD, cornersL, retL)
                retR,cornersR = self._getChessBoard(imgR)
                if retR:
                    cv2.drawChessboardCorners(imgR,CHECKERBOARD, cornersR, retR)
            
        if retL == True and retR == True:
            mode = '[O]'
        else:
            mode = '[X]'
        
        print (">>>",mode,len(self.imgpoints_l),len(self.imgpoints_r))
        return retL,imgL,cornersL,retR,imgR,cornersR
    

    def read_image_by_dir(self,dirPath):
        l_files = sorted(glob.glob(os.path.join(dirPath,'*_l.jpg')))
        r_files = sorted(glob.glob(os.path.join(dirPath,'*_r.jpg')))
        for index in range(len(l_files)):
            imgL = cv2.imread(l_files[index])
            imgR = cv2.imread(r_files[index])
            assert imgR is not None and imgL is not None, 'image is None.'
            if self.imgShape is None :
                self.imgShape = imgL.shape
            
            retL,imgL,cornersL,retR,imgR,cornersR = self.run_stereo_image_on_chessboard(imgL,imgR)

            if retL == True and retR == True:
                self.imgpoints_l.append(cornersL)
                self.imgpoints_r.append(cornersR)
                self.objpoints.append(self.objp)
                self.objpointsFe.append(self.objpFe)

            mergeImg = np.hstack((imgL,imgR))
            cv2.namedWindow('demo',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('demo',(mergeImg.shape[1]//3,mergeImg.shape[0]//3))
            cv2.imshow('demo', mergeImg)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        
        
        """ run calibration """
        self.calibate()

    def read_image_by_webcam(self,leftCamId,rightCamId,webCam,camInfo=None):
        if webCam == 'D435':
            import pyrealsense2 as rs
            from realsense_device_manager import DeviceManager
            rs_config = rs.config()
            rs_config.enable_stream(rs.stream.color, camInfo[0], camInfo[1], rs.format.bgr8, camInfo[2])

            device_manager = DeviceManager(rs.context(), rs_config)
            device_manager.enable_device(leftCamId, enable_ir_emitter=False)
            device_manager.enable_device(rightCamId, enable_ir_emitter=False)
            print ('Use D435 to do stereo calibration. ')
        else:
            cap_r = cv2.VideoCapture(rightCamId)
            cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            cap_l = cv2.VideoCapture(leftCamId)
            cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print ('Use V2L to do stereo calibration. ')

        imgL = None
        imgR = None
        while True:
            if  webCam == 'D435':
                # Wait for a coherent pair of frames: depth and color
                frames_devices = device_manager.poll_frames()
                
                for (device, frame) in frames_devices.items():
                    image = np.rot90(np.asarray(frame[rs.stream.color].get_data()),3)
                    if device == leftCamId:
                        imgL = image
                    elif device == rightCamId:
                        imgR = image
            else:
                ret_r, imgR = cap_r.read()
                assert ret_r == True
                
                st1 = time.time()    
                ret_l, imgL = cap_l.read()
                assert ret_l == True

                if self.imgShape is None :
                    self.imgShape = imgL.shape
            
            origImgL = imgL.copy()
            origImgR = imgR.copy()
            retL,imgL,cornersL,retR,imgR,cornersR = self.run_stereo_image_on_chessboard(imgL,imgR)
            
            mergeImg = np.hstack((imgL,imgR))
                
            cv2.namedWindow('demo',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('demo',(mergeImg.shape[1]//3,mergeImg.shape[0]//3))
            cv2.imshow('demo', mergeImg)
            
            if cv2.waitKey(50) & 0xff == ord('q'):
                break
            if cv2.waitKey(50) & 0xff == ord('s'):
                if retL == True and retR == True:
                    cv2.imwrite('./tmp/'+str(len(self.imgpoints_l)).zfill(2)+'_l.jpg',origImgL)
                    cv2.imwrite('./tmp/'+str(len(self.imgpoints_r)).zfill(2)+'_r.jpg',origImgR)
                    self.imgpoints_l.append(cornersL)
                    self.imgpoints_r.append(cornersR)
                    self.objpoints.append(self.objp)
                    self.objpointsFe.append(self.objpFe)

        if webCam == 'D435':  
            device_manager.disable_streams()
        cv2.destroyAllWindows()

        """ run calibration """
        self.calibate()
    
    def calibate(self):
        doPreCali = False
        img_shape = self.imgShape[::-1][1:] 
        if doPreCali:
            if len(self.imgpoints_l) == 0 or len(self.imgpoints_r) == 0:
                print ('No detecting any chessboard corners, and finish it.')
            
            else:
                if  self.fisheyeCali:
                    print ('fisheye calibration...')
                    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
                    N_OK = len(self.objpointsFe)
                    init_K = np.zeros((3, 3))
                    init_D = np.zeros((4, 1))
                    init_rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
                    init_tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
                    rt, self.M1, self.d1, self.r1, self.t1 =  cv2.fisheye.calibrate(self.objpointsFe,
                                                                                    self.imgpoints_l,
                                                                                    img_shape, # [w*h]
                                                                                    init_K,
                                                                                    init_D,
                                                                                    init_rvecs,
                                                                                    init_tvecs,
                                                                                    calibration_flags,
                                                                                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
                    
                    rt, self.M2, self.d2, self.r2, self.t2 =  cv2.fisheye.calibrate(self.objpointsFe,
                                                                                    self.imgpoints_r,
                                                                                    img_shape, # [w*h]
                                                                                    init_K,
                                                                                    init_D,
                                                                                    init_rvecs,
                                                                                    init_tvecs,
                                                                                    calibration_flags,
                                                                                    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
                else:
                    print ('normal calibration...')
                    rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, None, None)
                    rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, None, None)
            # print (self.d1,self.d1.shape)

        self.stereo_calibrate(img_shape,doPreCali)
            
    def stereo_calibrate(self, img_shape, doPreCali):
        if not self.fisheyeCali:
            flags = 0
            flags |= cv2.CALIB_FIX_INTRINSIC
            # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_ASPECT_RATIO
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
            # flags |= cv2.CALIB_RATIONAL_MODEL
            # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_K3
            # flags |= cv2.CALIB_FIX_K4
            # flags |= cv2.CALIB_FIX_K5

            stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(self.objpoints, 
                                                                self.imgpoints_l,self.imgpoints_r, 
                                                                self.M1, self.d1, self.M2, self.d2, 
                                                                img_shape, criteria=stereocalib_criteria, flags=flags)

            print('Intrinsic_mtx_1', M1)
            print('distCoeffs1', d1)
            print('Intrinsic_mtx_2', M2)
            print('distCoeffs2', d2)
            print('R', R)
            print('T', T)
            print('')

            stereo_camera_model = dict([('Intrinsic_mtx_1', M1), ('Intrinsic_mtx_2', M2), ('dist1', d1),
                                            ('dist2', d2), ('rvecs1', self.r1),
                                            ('rvecs2', self.r2), ('R', R), ('T', T),])
            with open('./stereo_cmaera_caliParam.pkl','wb') as file:
                pickle.dump(stereo_camera_model, file)

        else:
            objpoints = np.array([self.objpoints], dtype=np.float64)
            objpoints = np.reshape(objpoints, (len(self.objpoints), 1, CHECKERBOARD[0]*CHECKERBOARD[1], 3))
            
            imgpoints_left = np.asarray(self.imgpoints_l, dtype=np.float64)
            imgpoints_left = np.reshape(imgpoints_left, (len(self.objpoints), 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))
            
            imgpoints_right = np.asarray(self.imgpoints_r, dtype=np.float64)
            imgpoints_right = np.reshape(imgpoints_right, (len(self.objpoints), 1, CHECKERBOARD[0]*CHECKERBOARD[1], 2))	

            if doPreCali:
                M1 = self.M1
                d1 = self.d1
                M2 = self.M2
                d2 = self.d2
                
            else:
                # M1 = np.zeros((3, 3))
                # d1 = np.zeros((4, 1))
                # M2 = np.zeros((3, 3))
                # d2 = np.zeros((4, 1))
                M1 = np.array([[563.7830564955816, 0.0, 934.0753795126369], [0.0, 561.4965497815822, 424.63229486280255], [0.0, 0.0, 1.0]])
                M2 = np.array([[563.7830564955816, 0.0, 934.0753795126369], [0.0, 561.4965497815822, 424.63229486280255], [0.0, 0.0, 1.0]])
                d1 = np.array([[-0.07815326673586208], [0.06614725743511764], [-0.05415647567553164], [0.013975752811922386]])
                d2 = np.array([[-0.07815326673586208], [0.06614725743511764], [-0.05415647567553164], [0.013975752811922386]])

            calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW 
            flags = 0
            flags |= cv2.fisheye.CALIB_FIX_INTRINSIC
            flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            # flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            flags |= cv2.fisheye.CALIB_FIX_SKEW
            flags |= cv2.fisheye.CALIB_CHECK_COND
            

            R = np.zeros((1, 1, 3), dtype=np.float64)
            T = np.zeros((1, 1, 3), dtype=np.float64)

            stereocalib_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            
            rms, M1, d1, M2, d2, R, T = cv2.fisheye.stereoCalibrate(objpoints,
                                                                    imgpoints_left,imgpoints_right,
                                                                    M1, d1, M2, d2,
                                                                    img_shape,
                                                                    R,
                                                                    T,
                                                                    flags = flags,
                                                                    criteria = stereocalib_criteria)

        print('Intrinsic_mtx_1', M1)
        print('distCoeffs1', d1)
        print('Intrinsic_mtx_2', M2)
        print('distCoeffs2', d2)
        print('R', R)
        print('T', T)
        print('')

        stereo_camera_model = dict([('Intrinsic_mtx_1', M1), 
                                    ('Intrinsic_mtx_2', M2), 
                                    ('dist1', d1),
                                    ('dist2', d2), 
                                    ('R', R), 
                                    ('T', T),])
        with open('./fisheye_stereo_cmaera_caliParam.pkl','wb') as file:
            pickle.dump(stereo_camera_model, file)


if __name__ == "__main__":
    sc = StereoCalibration(fisheyeCalibration=True)
    webcam = 'normal'

    """ use image file to calibrate """
    filePath = './tmp2/'
    if os.path.isdir(filePath):
        print (" Use saved images to do stero calibration.")
        sc.read_image_by_dir(filePath)
        
    else:
        if webcam == 'D435':
            left_camId = '832112073441'
            right_camId = '832112072526'
            d435CamInfo = [1920,1080,30] # [w,h,fps]
            sc.read_image_by_webcam(left_camId,right_camId, webcam, camInfo=d435CamInfo)   
        else: 
            leftCamId = 2
            rightCamId = 0
            sc.read_image_by_webcam(leftCamId, rightCamId, webcam)


# Intrinsic_mtx_1 [[555.0277735    0.         981.00183271]
#  [  0.         553.7873645  420.75245304]
#  [  0.           0.           1.        ]]
# distCoeffs1 [[-0.04264375]
#  [ 0.0006726 ]
#  [ 0.01743312]
#  [-0.01728765]]
# Intrinsic_mtx_2 [[562.46282876   0.         932.84605041]
#  [  0.         561.99000913 434.73559701]
#  [  0.           0.           1.        ]]
# distCoeffs2 [[-0.03944703]
#  [-0.02248328]
#  [ 0.02804656]
#  [-0.01161518]]
# R [[ 0.9995052  -0.00596516 -0.03088305]
#  [ 0.0048074   0.99928777 -0.03742786]
#  [ 0.03108431  0.03726088  0.998822  ]]
# T [[-1.99790454]
#  [-0.0365649 ]
#  [ 0.06187026]]