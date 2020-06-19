#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:22:13 2020

@author: rayhliu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time 
import pickle
import tkinter as tk

class BM_UI_Controller():
    def __init__(self):
        self.param_dict = {}
        self.init_controller()
        
    def oddVals(self,value):
        value = int(value)
        if value % 2==0: value = value+1
        return int(value)
        
    def x16(self,value):
        value = int(value)
        return int(value/16)*16
    
    def minDisparityCallBack(self,value):
        updateValue = self.oddVals(value)
        self.param_dict['MinDisparity'] = updateValue
    
    def numDisparitiesCallBack(self,value):
        updateValue = self.x16(value)
        self.param_dict['NumDisparities'] = updateValue
        
    def blockSizeCallBack(self,value):
        updateValue = self.oddVals(value)
        self.param_dict['BlockSize'] = updateValue
        
    def print_output(self):
        self.label.config(text=str(self.param_dict))
        
        
    def update(self):
        self.master.update()
        
        self.param_dict['MinDisparity'] = self.oddVals(self.MinDisparity.get())
        self.param_dict['NumDisparities'] = self.x16(self.NumDisparities.get())
        self.param_dict['BlockSize'] = self.oddVals(self.BlockSize.get())
        self.param_dict['TextureThreshold'] = self.TextureThreshold.get()
        self.param_dict['UniquenessRatio'] = self.UniquenessRatio.get()
        self.print_output()
#        self.param_dict['SpeckleRange'] = self.SpeckleRange.get()
#        self.param_dict['SpeckleWindowSize'] = self.SpeckleWindowSize.get()
    
    def init_controller(self):
        self.master = tk.Tk()
        self.master.title("StereoBM Settings")
        
        self.label = tk.Label(self.master, bg='yellow', width=100,height= 3, text='empty')
        self.label.pack()

        
        self.MinDisparity = tk.Scale(self.master, from_=-100, to=100, length=600, command=self.minDisparityCallBack, orient=tk.HORIZONTAL, label="Minimum Disparities")
        self.MinDisparity.pack()
        self.MinDisparity.set(0)
        
        self.NumDisparities = tk.Scale(self.master, from_=16, to=2048, length=600, command=self.numDisparitiesCallBack, orient=tk.HORIZONTAL, label="Number of Disparities")
        self.NumDisparities.pack()
        self.NumDisparities.set(0)
        
        self.BlockSize = tk.Scale(self.master, from_=5, to=255, length=600 ,orient=tk.HORIZONTAL, label="Block Size")
        self.BlockSize.pack()
        self.BlockSize.set(15)
        
        self.TextureThreshold = tk.Scale(self.master, from_=0, to=2500, length=600, orient=tk.HORIZONTAL, label="Texture Threshold")
        self.TextureThreshold.pack()
        self.TextureThreshold.set(0)
        
        self.UniquenessRatio = tk.Scale(self.master, from_=0, to=150, length=600, orient=tk.HORIZONTAL, label="Uniqueness Ratio")
        self.UniquenessRatio.pack()
        self.UniquenessRatio.set(15)
        
        self.SpeckleRange = tk.Scale(self.master, from_=0, to=60, length=600, orient=tk.HORIZONTAL, label="Speckle Range")
        self.SpeckleRange.pack()
        self.SpeckleRange.set(34)
        
        self.SpeckleWindowSize = tk.Scale(self.master, from_=60, to=150, length=600, orient=tk.HORIZONTAL, label="Speckle Window Size")
        self.SpeckleWindowSize.pack()
        self.SpeckleWindowSize.set(100)

with open('./fisheye_stereo_cmaera_caliParam.pkl' ,'rb') as file :
    stero_param = pickle.load(file)

def getRectifyTransform(height, width, config, fisheyeCali=False):
    """ load camera matrix and get stereo camera rectify param """
    left_K = config['Intrinsic_mtx_1']
    right_K = config['Intrinsic_mtx_2']
    left_distortion = config['dist1']
    right_distortion = config['dist2']
    R = config['R']
    T = config['T']
 
    
    height = int(height)
    width = int(width)

    if fisheyeCali:
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(left_K, left_distortion, 
                                                      right_K, right_distortion, 
                                                      (width, height), 
                                                      R, 
                                                      T, 
#                                                      flags=cv2.CALIB_ZERO_DISPARITY,
                                                      fov_scale=1)

        map1x, map1y = cv2.fisheye.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_16FC1)  
        map2x, map2y = cv2.fisheye.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_16FC1)
    else:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=0)
        map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
    
    return map1x, map1y, map2x, map2y, Q
 
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    """ rectify image """
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
 
    return rectifyed_img1, rectifyed_img2

def draw_line(image1, image2):
    """ check the left frame and right frame (horizontal pair) """
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
 
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
 
    for k in range(15):
        cv2.line(output, (0, 50 * (k + 1)), (2 * width, 50 * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
 
    return output

def disparity_SGBM(left_image, right_image, down_scale=False):
    # SGBM Param
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 11
    param = {'minDisparity': 0,
             'numDisparities': 16,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 31,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 32,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }
 

    sgbm = cv2.StereoSGBM_create(**param)
 
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = sgbm.compute(left_image, right_image)
        disparity_right = sgbm.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = size[0] / left_image_down.shape[1]
        disparity_left_half = sgbm.compute(left_image_down, right_image_down)
        disparity_right_half = sgbm.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA) 
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left *= factor 
        disparity_right *= factor
 
    return disparity_left, disparity_right


def preprocess(img1, img2):
    im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
    im1 = cv2.equalizeHist(im1)
    im2 = cv2.equalizeHist(im2)
 
    return im1, im2


def initBM():
    stereo = cv2.StereoBM_create(numDisparities=0, blockSize=0)
    stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
    
    """
#    stereo.setPreFilterSize(5)
#    stereo.setPreFilterCap(31)
    stereo.setBlockSize(9)
    stereo.setMinDisparity(-70)
    stereo.setNumDisparities(80)
    stereo.setTextureThreshold(0)
    stereo.setUniquenessRatio(0)
#    stereo.setSpeckleWindowSize()
#    stereo.setSpeckleRange(16)
    """
    
    return stereo

def update_BM(stereo,paramDict):
    stereo.setBlockSize(paramDict['BlockSize'])
    stereo.setMinDisparity(paramDict['MinDisparity'])
    stereo.setNumDisparities(paramDict['NumDisparities'])
    stereo.setTextureThreshold(paramDict['TextureThreshold'])
    stereo.setUniquenessRatio(paramDict['UniquenessRatio'])
    return stereo

def update_SGBM(stereo,paramDict):
    stereo.setBlockSize(paramDict['BlockSize'])
    stereo.setMinDisparity(paramDict['MinDisparity'])
    stereo.setNumDisparities(paramDict['NumDisparities'])
    stereo.setTextureThreshold(paramDict['TextureThreshold'])
    stereo.setUniquenessRatio(paramDict['UniquenessRatio'])
    return stereo
    

if __name__ == "__main__":
    detet_source = 'normal'
    genDepth = True
    bmUI = BM_UI_Controller()
    myBM = initBM()
    
    if detet_source != 'ip':
        if detet_source == 'd435':
            import pyrealsense2 as rs
            from realsense_device_manager import DeviceManager
            rs_config = rs.config()
            rs_config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            leftCamId = '832112073441'
            rightCamId = '832112072526'
            device_manager = DeviceManager(rs.context(), rs_config)
            device_manager.enable_device(leftCamId, enable_ir_emitter=False)
            device_manager.enable_device(rightCamId, enable_ir_emitter=False)
            print ('Use D435 to do stereo calibration. ')
        else:
            leftCamId = 2
            rightCamId = 0
            cap_r = cv2.VideoCapture(rightCamId)
            cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            cap_l = cv2.VideoCapture(leftCamId)
            cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print ('Use V2L to do stereo calibration. ')
        

        count = 0
        while True:

            count += 1
            # Wait for a coherent pair of frames: depth and color
            st = time.time()
            
            imgL = None
            imgR = None
            if detet_source == 'd435':
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
                    
            if imgL is not None and imgR is not None:
                h = imgL.shape[0]
                w = imgL.shape[1]
                
                """ stereo camera rectify """
                st_r = time.time()
                map1x, map1y, map2x, map2y, Q= getRectifyTransform(h, w, stero_param)  
                iml_rectified, imr_rectified = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)
#                print ('rect:',time.time() - st_r)
                
                """ show the merge frame """
                mergeRectifiedImg = draw_line(iml_rectified, imr_rectified)
                mergeImg = np.hstack((imgL,imgR))
#                cv2.imwrite("./stero_images/rectiImg_L.jpeg",iml_rectified)
#                cv2.imwrite("./stero_images/rectiImg_R.jpeg",imr_rectified)
#                cv2.imwrite("./stero_images/img_L.jpeg",imgL)
#                cv2.imwrite("./stero_images/img_R.jpeg",imgR)
#                cv2.imwrite("./stero_images/fisheye_rectImgL_"+str(count)+".jpg",iml_rectified)
#                cv2.imwrite("./stero_images/fisheye_rectImgR_"+str(count)+".jpg",imr_rectified)
                
                """ generate depth map by BM/SGBM """
                if genDepth:
                    bmUI.update()
                    myBM = update_BM(myBM,bmUI.param_dict)

                    # iml_, imr_ = preprocess(imgL, imgR) # orig_img
                    img_l = iml_rectified[0::6,0::6,:]
                    img_r = imr_rectified[0::6,0::6,:]
                    iml_, imr_ = preprocess(img_l, img_r) # calibrate img
            
                    disp = None
                    mode = 'bm'
                    if mode == 'sgbm':
                        """ sgbm """
                        disp, _ = disparity_SGBM(iml_,imr_)   
                    elif mode == 'bm':
                        """ bm """
                        disp = myBM.compute(iml_, imr_)
                    
                    print ("time:",time.time() - st)
                    
                    if disp is not None:
                        disp = np.divide(disp.astype(np.float32), 16.)
                        new_arr = ((disp - disp.min()) * (1/(disp.max() - disp.min()) * 255).astype('uint8'))
#                        cv2.imwrite('./depth/'+'fisheye_depth_'+str(count)+'.jpg',new_arr)
#                        cv2.imwrite('./depth/'+'fisheye_Recti_'+str(count)+'.jpg',mergeRectifiedImg)
#                        cv2.imwrite('./depth/'+'fisheye_'+str(count)+'.jpg',mergeImg)
                        plt.imshow(disp, 'gray')
                        plt.show()  
                    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=0.9), cv2.COLORMAP_BONE)
            
                cv2.namedWindow('demo',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('demo',(imgL.shape[1]*2//3,imgL.shape[0]//3))
                cv2.imshow('demo',mergeRectifiedImg )
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            else:
                continue
        if detet_source == 'd435':
            device_manager.disable_streams()
        cv2.destroyAllWindows()
    
    
    if detet_source == 'ip':
        img_l = cv2.imread('./stero_images/rectiImg_L.jpeg')
        img_r = cv2.imread('./stero_images/rectiImg_R.jpeg')
        
        img_l = img_l[0::5,0::5,:]
        img_r = img_r[0::5,0::5,:]
        line = draw_line(img_l, img_r)

        while True:
            if genDepth:
                bmUI.update()
                myBM = update_BM(myBM,bmUI.param_dict)

                iml_, imr_ = preprocess(img_l, img_r)
                mode = 'BM'
                st = time.time()
                if mode == 'BM':
                    myBM = BM()
                    disp = myBM.compute(iml_, imr_)
                elif mode == 'SGBM':
                    disp, _ = disparity_SGBM(iml_,imr_)  
                
                disp = np.divide(disp.astype(np.float32), 16.)  
                print (time.time()-st)
                plt.imshow(disp, 'gray')
                plt.show()  
        
            cv2.namedWindow('demo',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('demo',(imgL.shape[1]*2//3,imgL.shape[0]//3))
            cv2.imshow('demo',img_r )
            cv2.waitKey(0)
        

