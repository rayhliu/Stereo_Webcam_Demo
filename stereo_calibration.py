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

CHECKERBOARD = (9,7)

class StereoCalibration(object):
    def __init__(self):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.
        
    def _getChessBoard(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            corners = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        return ret,corners
    
    def read_image_by_d435(self,leftCamId,rightCamId,camInfo):
        import pyrealsense2 as rs
        from realsense_device_manager import DeviceManager
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, camInfo[0], camInfo[1], rs.format.bgr8, camInfo[2])

        device_manager = DeviceManager(rs.context(), rs_config)
        device_manager.enable_device(leftCamId, enable_ir_emitter=False)
        device_manager.enable_device(rightCamId, enable_ir_emitter=False)
        print ('Use D435 to do stero calibration. ')
        imgL = None
        imgR = None
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames_devices = device_manager.poll_frames()
            
            for (device, frame) in frames_devices.items():
                image = np.rot90(np.asarray(frame[rs.stream.color].get_data()),3)
                if device == leftCamId:
                    imgL = image
                elif device == rightCamId:
                    imgR = image
            
            if imgL is not None and imgR is not None:
                retL,cornersL = self._getChessBoard(imgL)
                retR,cornersR = self._getChessBoard(imgR)
            
            if retL == True:
                cv2.drawChessboardCorners(imgL,CHECKERBOARD, cornersL, retL)
            if retR == True:
                cv2.drawChessboardCorners(imgR,CHECKERBOARD, cornersR, retR)
            if retL == True and retR == True:
                mode = '[O]'
            else:
                mode = '[X]'
            
            print (">>>",mode,len(self.imgpoints_l),len(self.imgpoints_r))
            
            mergeImg = np.hstack((imgL,imgR))
                
            cv2.namedWindow('demo',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('demo',(540,480))
            cv2.imshow('demo', mergeImg)
            
            if cv2.waitKey(50) & 0xff == ord('q'):
                break
            if cv2.waitKey(50) & 0xff == ord('s'):
                if retL == True and retR == True:
                    self.imgpoints_l.append(cornersL)
                    self.imgpoints_r.append(cornersR)
                    self.objpoints.append(self.objp)
                    
        device_manager.disable_streams()
        cv2.destroyAllWindows()
        
        if len(self.imgpoints_l) == 0 or len(self.imgpoints_r) == 0:
            print ('No detecting any chessboard corners, and finish it.')
        
        else:
            img_shape = imgL.shape[::-1][1:] 
            rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, None, None)
            rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, None, None)
            self.stereo_calibrate(img_shape)
        
    def read_images(self, cal_path):
        images_right = glob.glob(cal_path + 'RIGHT/*.JPG')
        images_left = glob.glob(cal_path + 'LEFT/*.JPG')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9, 6),
                                                  corners_l, ret_l)
                cv2.imshow(images_left[i], img_l)
                cv2.waitKey(500)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9, 6), corners_r, ret_r)
                cv2.imshow(images_right[i], img_r)
                cv2.waitKey(500)
            img_shape = gray_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, None, None)
        self.stereo_calibrate(img_shape)
        
            
    def stereo_calibrate(self, dims):
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
                                                              dims, criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('distCoeffs1', d1)
        print('Intrinsic_mtx_2', M2)
        print('distCoeffs2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        stereo_camera_model = dict([('Intrinsic_mtx_1', M1), ('Intrinsic_mtx_2', M2), ('dist1', d1),
                                        ('dist2', d2), ('rvecs1', self.r1),
                                        ('rvecs2', self.r2), ('R', R), ('T', T),
                                        ('E', E), ('F', F)])
        with open('./stereo_cmaera_caliParam.pkl','wb') as file:
            pickle.dump(stereo_camera_model, file)


if __name__ == "__main__":
    sc = StereoCalibration()
    
    """ use image file to calibrate """
    filePath = ''
    if os.path.isfile(filePath):
        print (" Use saved images to do stero calibration.")
        sc.read_images(filePath)
        
    else:
        """ use d435 webCam to calibrate """
        try:
            left_camId = '832112073441'
            right_camId = '832112072526'
            camInfo = [1920,1080,30] # [w,h,fps]
            sc.read_image_by_d435(left_camId,right_camId,camInfo)        
        except:
            print ('Can not do stero calibration')

