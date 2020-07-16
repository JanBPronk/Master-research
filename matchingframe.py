# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:41:08 2019

@author: janbo
"""

import matplotlib.pyplot as plt
import gdal
from osgeo import gdal, osr
import numpy as np
import sys
import time
import cv2
from scipy import ndimage
from scipy import stats
import scipy.io as sio
import rasterio
import rasterio.plot
import rasterio.mask
import utm
import geopandas as gpd


np.set_printoptions(threshold=sys.maxsize)

from sidefunctions import ImgDisp, colfilt, bwareaopen, filtDisp


def loadsubimage(filepath, areapath):
    #filepath: path of image
    #areapath: path of shapefile
    #returns: img: cropped image, out_transform: utm coordinates cropped image

    # *** choose this when taking subselection by creation of square shapefile
#    shp=gpd.read_file(areapath + 'AreaSelection.shp')
#    shp_geometry=shp.geometry
     
    with rasterio.open(filepath) as src:
        print(filepath)       
        # ***
#        out_image, out_transform = rasterio.mask.mask(src, shp_geometry, crop=True, all_touched=True)
        # use this when masking a shapefile by glacier inventory
        out_image, out_transform = rasterio.mask.mask(src, areapath, crop=True, all_touched=True)
        img=out_image[0]
        img=img.astype(np.float32) 
       
    return img, out_transform


######## preprocessing

def orientation(I1, I2):
    #I1:        image used as chip
    #I2:        image used as reference
    #returns orientation image I1_f0 and I2_f0
    
    I1_d=(np.gradient(I1)[1] + np.gradient(I1)[0]*1j)
    I1_f0=I1_d/abs(I1_d)
    I1_f0[abs(I1_d)==0]=0+0j
       
    I2_d=(np.gradient(I2)[1] + np.gradient(I2)[0]*1j)
    I2_f0=I2_d/abs(I2_d)
    I2_f0[abs(I2_d)==0]=0+0j
     
    return I1_f0, I2_f0


#def wallis_filter(I1, I2):
#    # Used from Autorift module
#    #I1:        image used as chip
#    #I2:        image used as reference
#    
#    WallisFilterWidth = 21
#
#    I1 = np.abs(I1)
#    I2 = np.abs(I2)
#
#    kernel = np.ones((WallisFilterWidth,WallisFilterWidth), dtype=np.float32)
#    
#    m = cv2.filter2D(I1,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)
#    m2 = (I1)**2    
#    m2 = cv2.filter2D(m2,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)
#    s = np.sqrt(m2 - m**2) * np.sqrt(np.sum(kernel)/(np.sum(kernel)-1.0))
#    I1 = (I1 - m) / s
#    
#    m = cv2.filter2D(I2,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)    
#    m2 = (I2)**2   
#    m2 = cv2.filter2D(m2,-1,kernel,borderType=cv2.BORDER_CONSTANT)/np.sum(kernel)    
#    s = np.sqrt(m2 - m**2) * np.sqrt(np.sum(kernel)/(np.sum(kernel)-1.0))    
#    I2 = (I2 - m) / s
#         
#    return I1, I2
#

'''
Do the actual processing.
'''

def matchingframe(I1,I2, CoreType, areapath):
    #I1:        image used as chip
    #I2:        image used as reference
    #CoreType:  CNN or CFF-O
    #areapath:  directory to shapefile to crop out subselection
    #
    #returns:
    #Dx,Dy:     Arrays with displacement data in x- and y-direction
    #Interpmask:
    #ChipsizeX,Y: chipsize used for reliable calculations
    #sendout:   ??
    #I1, I2_out_transform: coordinate grid system information for Dx, Dy.
    
    
    #INPUTS
    xGrid = None
    yGrid = None
    Dx0 = 0
    Dy0 = 0

    ##Output file
    Dx = None
    Dy = None
    InterpMask = None
    ChipSizeX = None
    ChipSizeY = None
    
    ##Parameter list
    ChipSizeSmall = 16 ####
    ChipSizeLarge = 32 ####
    ScaleChipSizeY = 1
    SearchLimitX = 15
    SearchLimitY = 15
    SkipSampleX = 8 ####
    SkipSampleY = 8 ####
    fillFiltWidth = 3    
    minSearch = 6
    FracValid = 8/25
    FracSearch = 0.25
    FiltWidth = 5
    Iter = 3
    MadScalar = 4 #standart at 4
    OverSampleRatio = 16
    
    I1, I1_out_transform=loadsubimage(I1, areapath)
    I2, I2_out_transform=loadsubimage(I2, areapath)
 
    
    MI=(np.logical_not((I1==0))) & (np.logical_not(I2==0))
    
    #depending on CoreType, preprocess image
    if CoreType=='CNN':
        I1, I2=wallis_filter(I1, I2)
    elif CoreType=='CFF-O':
        I1, I2=orientation(I1, I2)
    else:
        sys.exit('No valid coretype selected')


    t1 = time.time()
    print("Create Grid Start")
    
    # create the grid as it does not yet exist
    m,n = I1.shape
    xGrid = np.arange(SkipSampleX+10,n-SkipSampleX,SkipSampleX)
    print(xGrid.shape)
    yGrid = np.arange(SkipSampleY+10,m-SkipSampleY,SkipSampleY)
    nd = len(xGrid)
    md = len(yGrid)
    xGrid = np.dot(np.ones((md,1)),np.reshape(xGrid,(1,len(xGrid))))
    yGrid = np.dot(np.reshape(yGrid,(len(yGrid),1)),np.ones((1,nd)))
    
    #mask for initial nodata
    MI=MI[SkipSampleY+10:m-SkipSampleY:SkipSampleY, SkipSampleX+10:n-SkipSampleX:SkipSampleX]

    origSize = xGrid.shape
    sendout=[xGrid, yGrid, origSize]
    
    print("Create Grid Done!!")
    

    ###Now the autorift part ###
    
    ChipSizeUniX = np.array([ChipSizeSmall,ChipSizeLarge], np.float32)
    ChipSize0=ChipSizeSmall
    
    Dx0 = np.ones(xGrid.shape, np.float32) * np.round(Dx0)
    Dy0 = np.ones(xGrid.shape, np.float32) * np.round(Dy0)
    SearchLimitX = np.ones(xGrid.shape, np.float32) * np.round(SearchLimitX)
    SearchLimitY = np.ones(xGrid.shape, np.float32) * np.round(SearchLimitY)
    ChipSizeSmall = np.ones(xGrid.shape, np.float32) * np.round(ChipSizeSmall)
    ChipSizeLarge = np.ones(xGrid.shape, np.float32) * np.round(ChipSizeLarge)
    
    ChipSizeX = np.zeros(xGrid.shape, np.float32)
    InterpMask = np.zeros(xGrid.shape, np.bool) #matrix of FALSE
    Dx = np.empty(xGrid.shape, dtype=np.float32)
    Dx.fill(np.nan)
    Dy = np.empty(xGrid.shape, dtype=np.float32)
    Dy.fill(np.nan)    


    for i in range(len(ChipSizeUniX)):
        
        if ChipSize0 != ChipSizeUniX[i]:
            Scale = ChipSize0 / ChipSizeUniX[i]
            dstShape = (int(xGrid.shape[0]*Scale),int(xGrid.shape[1]*Scale))
            xGrid0 = cv2.resize(xGrid.astype(np.float32),dstShape[::-1],interpolation=cv2.INTER_AREA)
            yGrid0 = cv2.resize(yGrid.astype(np.float32),dstShape[::-1],interpolation=cv2.INTER_AREA)
            
            if np.mod(ChipSizeUniX[i],2) == 0:
                xGrid0 = np.round(xGrid0+0.5)-0.5
                yGrid0 = np.round(yGrid0+0.5)-0.5
            else:
                xGrid0 = np.round(xGrid0)
                yGrid0 = np.round(yGrid0)
        
            M0 = (ChipSizeX == 0) & (ChipSize0 <= ChipSizeUniX[i]) & (ChipSizeLarge >= ChipSizeUniX[i])
            M0 = colfilt(M0, (int(1/Scale*6), int(1/Scale*6)), 0)
            M0 = cv2.resize(np.logical_not(M0).astype(np.uint8),dstShape[::-1],interpolation=cv2.INTER_NEAREST).astype(np.bool)
            MI0 = cv2.resize(MI.astype(np.uint8),dstShape[::-1],interpolation=cv2.INTER_NEAREST).astype(np.bool)
            
            SearchLimitX0 = colfilt(SearchLimitX, (int(1/Scale), int(1/Scale)), 0) + colfilt(Dx0, (int(1/Scale), int(1/Scale)), 4)
            SearchLimitY0 = colfilt(SearchLimitY, (int(1/Scale), int(1/Scale)), 0) + colfilt(Dy0, (int(1/Scale), int(1/Scale)), 4)
            Dx00 = colfilt(Dx0, (int(1/Scale), int(1/Scale)), 2)
            Dy00 = colfilt(Dy0, (int(1/Scale), int(1/Scale)), 2)
        
            SearchLimitX0 = np.ceil(cv2.resize(SearchLimitX0,dstShape[::-1]))
            SearchLimitY0 = np.ceil(cv2.resize(SearchLimitY0,dstShape[::-1]))
            SearchLimitX0[M0] = 0
            SearchLimitY0[M0] = 0
            Dx00 = np.round(cv2.resize(Dx00,dstShape[::-1],interpolation=cv2.INTER_NEAREST))
            Dy00 = np.round(cv2.resize(Dy00,dstShape[::-1],interpolation=cv2.INTER_NEAREST))
            error=Dy00.copy()
        
        else:
            SearchLimitX0 = SearchLimitX.copy()
            SearchLimitY0 = SearchLimitY.copy()
            Dx00 = Dx0.copy()
            Dy00 = Dy0.copy()
            error= Dy00.copy()
            xGrid0 = xGrid.copy()
            yGrid0 = yGrid.copy()
            MI0 = MI.copy()


        idxZero = (SearchLimitX0 <= 0) | (SearchLimitY0 <= 0)
        SearchLimitX0[idxZero] = 0
        SearchLimitY0[idxZero] = 0
        SearchLimitX0[(np.logical_not(idxZero)) & (SearchLimitX0 < minSearch)] = minSearch
        SearchLimitY0[(np.logical_not(idxZero)) & (SearchLimitY0 < minSearch)] = minSearch
        SearchLimitX0[np.logical_not(MI0)] = 0
        SearchLimitY0[np.logical_not(MI0)] = 0
        
#        plt.figure()
#        plt.imshow(SearchLimitY0)
#        plt.show()

        # Fine Search
        SubPixFlag = True
        ChipSizeXF = ChipSizeUniX[i]
        ChipSizeYF = np.float32(np.round(ChipSizeXF*ScaleChipSizeY/2)*2)

 

        DxF, DyF, error= ImgDisp(I2, I1, xGrid0, yGrid0, ChipSizeXF, ChipSizeYF, SearchLimitX0, SearchLimitY0, Dx00, Dy00, SubPixFlag, CoreType, OverSampleRatio)

#        print(error)
        if ChipSizeUniX[i] == ChipSizeUniX[0]:
            FracValid = 8/25
            FracSearch = 0.25
            FiltWidth = 5
            Iter = 3
            MadScalar = 4
        
        if ChipSize0 == ChipSizeUniX[i]:
            DxFF=DxF.copy()
        M0 = filtDisp(DxF, DyF, SearchLimitX0, SearchLimitY0, np.logical_not(np.isnan(DxF)), FracValid, FracSearch, FiltWidth, Iter, MadScalar)
        DxF[np.logical_not(M0)] = np.nan
        DyF[np.logical_not(M0)] = np.nan
#        error[np.logical_not(M0)] = np.nan
#        print(error)
        
        # Light interpolation with median filtered values: DxFM (filtered) and DxF (unfiltered)
        DxFM = colfilt(DxF, (fillFiltWidth, fillFiltWidth), 3)
        DyFM = colfilt(DyF, (fillFiltWidth, fillFiltWidth), 3)
        
        # M0 is mask for original valid estimates, MF is mask for filled ones, MM is mask where filtered ones exist for filling
        MF = np.zeros(M0.shape, dtype=np.bool)
        MM = np.logical_not(np.isnan(DxFM))
    
        for j in range(3):
            foo = MF | M0   # initial valid estimates
            foo1 = (cv2.filter2D(foo.astype(np.float32),-1,np.ones((3,3)),borderType=cv2.BORDER_CONSTANT) >= 6) | foo     # 1st area closing followed by the 2nd (part of the next line calling OpenCV)
            fillIdx = np.logical_not(bwareaopen(np.logical_not(foo1).astype(np.uint8), 5)) & np.logical_not(foo) & MM
            MF[fillIdx] = True
            DxF[fillIdx] = DxFM[fillIdx]
            DyF[fillIdx] = DyFM[fillIdx]
        
        
        # Below is for replacing the valid estimates with the bicubic filtered values for robust and accurate estimation

        if ChipSize0 == ChipSizeUniX[i]:
            Dx = DxF
            Dy = DyF
            ChipSizeX[M0|MF] = ChipSizeUniX[i]
            InterpMask[MF] = True
            errorf=error
            
        else:
            Scale = ChipSizeUniX[i] / ChipSize0
            dstShape = (int(Dx.shape[0]/Scale),int(Dx.shape[1]/Scale))
            
            # DxF0 (filtered) / Dx (unfiltered) is the result from earlier iterations, DxFM (filtered) / DxF (unfiltered) is that of the current iteration
            # first colfilt_CFF nans within 2-by-2 area (otherwise 1 nan will contaminate all 4 points)
            DxF0 = colfilt(Dx,(int(Scale+1),int(Scale+1)),2)
            # then resize to half size using area (similar to averaging) to match the current iteration
            DxF0 = cv2.resize(DxF0,dstShape[::-1],interpolation=cv2.INTER_AREA)
            DyF0 = colfilt(Dy,(int(Scale+1),int(Scale+1)),2)
            DyF0 = cv2.resize(DyF0,dstShape[::-1],interpolation=cv2.INTER_AREA)
            
            # Note this DxFM is almost the same as DxFM (same variable) in the light interpolation (only slightly better); however, only small portion of it will be used later at locations specified by M0 and MF that are determined in the light interpolation. So even without the following two lines, the final Dx and Dy result is still the same.
            # to fill out all of the missing values in DxF
            DxFM = colfilt(DxF, (5,5), 3)
            DyFM = colfilt(DyF, (5,5), 3)
            
            # fill the current-iteration result with previously determined reliable estimates that are not searched in the current iteration
            idx = np.isnan(DxF) & np.logical_not(np.isnan(DxF0))
            DxFM[idx] = DxF0[idx]
            DyFM[idx] = DyF0[idx]
            
            # Strong interpolation: use filtered estimates wherever the unfiltered estimates do not exist
            idx = np.isnan(DxF) & np.logical_not(np.isnan(DxFM))
            DxF[idx] = DxFM[idx]
            DyF[idx] = DyFM[idx]
            
            dstShape = (Dx.shape[0],Dx.shape[1])
            DxF = cv2.resize(DxF,dstShape[::-1],interpolation=cv2.INTER_CUBIC)
            DyF = cv2.resize(DyF,dstShape[::-1],interpolation=cv2.INTER_CUBIC)
            MF = cv2.resize(MF.astype(np.uint8),dstShape[::-1],interpolation=cv2.INTER_NEAREST).astype(np.bool)
            M0 = cv2.resize(M0.astype(np.uint8),dstShape[::-1],interpolation=cv2.INTER_NEAREST).astype(np.bool)
            
            idxRaw = M0 & (ChipSizeX == 0)
            idxFill = MF & (ChipSizeX == 0)
            ChipSizeX[idxRaw | idxFill] = ChipSizeUniX[i]
            InterpMask[idxFill] = True
            Dx[idxRaw | idxFill] = DxF[idxRaw | idxFill]
            Dy[idxRaw | idxFill] = DyF[idxRaw | idxFill]



    ChipSizeY = np.round(ChipSizeX * ScaleChipSizeY /2) * 2
    Dy[np.logical_not(MI)]=np.nan
    Dx[np.logical_not(MI)]=np.nan
    Dx = Dx
    Dy = Dy
    InterpMask = InterpMask
    ChipSizeX = ChipSizeX
    ChipSizeY = ChipSizeY
    print(time.time()-t1)
    
    return Dx, Dy, InterpMask, ChipSizeX, ChipSizeY, sendout, I1_out_transform, I2_out_transform


    
