# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:38:31 2019

@author: janbo
"""

'''
intermezzo functions.

'''

#involved mostly postprocessing: large parts adopted from Dehecq (2019)

import numpy as np
from skimage import measure
from skimage.util import view_as_windows as viewW
import matplotlib.pyplot as plt
import pdb
import subprocess
import re
import string
import sys

#import the following functions
from PixDispCore import PixDisp_CFF, PixDisp_CNN, SubPixDisp_CFF, SubPixDisp_CNN



def ImgDisp(I1, I2, xGrid, yGrid, ChipSizeX, ChipSizeY, SearchLimitX, SearchLimitY, Dx0, Dy0, SubPixFlag, CoreType, OverSampleRatio):
    
    print('Start Core')

    #not important at all        
    if np.size(SearchLimitX) == 1:
        if np.logical_not(isinstance(SearchLimitX,np.float32) & isinstance(SearchLimitY,np.float32)):
            sys.exit('SearchLimit must be float')
    else:
        if np.logical_not((SearchLimitX.dtype == np.float32) & (SearchLimitY.dtype == np.float32)):
            sys.exit('SearchLimit must be float')

    if np.size(Dx0) == 1:
        if np.logical_not(isinstance(Dx0,np.float32) & isinstance(Dy0,np.float32)):
            sys.exit('Search offsets must be float')
    else:
        if np.logical_not((Dx0.dtype == np.float32) & (Dy0.dtype == np.float32)):
            sys.exit('Search offsets must be float')

    if np.size(ChipSizeX) == 1:
        if np.logical_not(isinstance(ChipSizeX,np.float32) & isinstance(ChipSizeY,np.float32)):
            sys.exit('ChipSize must be float')
    else:
        if np.logical_not((ChipSizeX.dtype == np.float32) & (ChipSizeY.dtype == np.float32)):
            sys.exit('ChipSize must be float')

    if np.any(np.mod(ChipSizeX,2) != 0) | np.any(np.mod(ChipSizeY,2) != 0):
        sys.exit('it is better to have ChipSize = even number')
    
    if np.any(np.mod(SearchLimitX,1) != 0) | np.any(np.mod(SearchLimitY,1) != 0):
        sys.exit('SearchLimit must be an integar value')

    if np.any(SearchLimitX < 0) | np.any(SearchLimitY < 0):
        sys.exit('SearchLimit cannot be negative')
    
    if np.any(np.mod(ChipSizeX,4) != 0) | np.any(np.mod(ChipSizeY,4) != 0):
        sys.exit('ChipSize should be evenly divisible by 4')
    
    
    
    if np.size(Dx0) == 1:
        Dx0 = np.ones(xGrid.shape, dtype=np.float32) * Dx0
    
    if np.size(Dy0) == 1:
        Dy0 = np.ones(xGrid.shape, dtype=np.float32) * Dy0
    
    if np.size(SearchLimitX) == 1:
        SearchLimitX = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitX
    
    if np.size(SearchLimitY) == 1:
        SearchLimitY = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitY

    if np.size(ChipSizeX) == 1:
        ChipSizeX = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeX
    
    if np.size(ChipSizeY) == 1:
        ChipSizeY = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeY

    # convert from cartesian X-Y to matrix X-Y: X no change, Y from up being positive to down being positive
    Dy0 = -Dy0
    
    SLx_max = np.max(SearchLimitX + np.abs(Dx0))
    Px = np.int(np.max(ChipSizeX)/2 + SLx_max + 2) #default 37
    SLy_max = np.max(SearchLimitY + np.abs(Dy0))
    Py = np.int(np.max(ChipSizeY)/2 + SLy_max + 2) #default 37
    
    I1 = np.lib.pad(I1,((Py,Py),(Px,Px)),'constant') #originele matrix met extra nullen eromheen voor de leuk ofzo (nee voor het zoeken natuurlijk)
    I2 = np.lib.pad(I2,((Py,Py),(Px,Px)),'constant')

    
    # adjust center location by the padarray size and 0.5 is added because we need to extract the chip centered at X+1 with -chipsize/2:chipsize/2-1, which equivalently centers at X+0.5 (X is the original grid point location). So for even chipsize, always returns offset estimates at (X+0.5).
    xGrid += (Px + 0.5)
    yGrid += (Py + 0.5)

    #voor de zekerheid ofzo
    Dx = np.empty(xGrid.shape,dtype=np.float32)
    Dx.fill(np.nan)
    Dy = Dx.copy()
    error = Dy.copy()

       
    for jj in range(xGrid.shape[1]):
        if np.all(SearchLimitX[:,jj] == 0) & np.all(SearchLimitX[:,jj] == 0):
            continue
        Dx1 = Dx[:,jj]
        Dy1 = Dy[:,jj]
        error1 = error[:,jj]
        for ii in range(xGrid.shape[0]):
            if (SearchLimitX[ii,jj] == 0) & (SearchLimitX[ii,jj] == 0):
                continue
        
            # remember motion terms Dx and Dy correspond to I1 relative to I2 (reference)
            #hier worden de chips gemaakt: worden centraal op de chip heen gebouwd
            clx = np.floor(ChipSizeX[ii,jj]/2)
            ChipRangeX = slice(int(-clx - Dx0[ii,jj] + xGrid[ii,jj]) , int(clx - Dx0[ii,jj] + xGrid[ii,jj]))
            cly = np.floor(ChipSizeY[ii,jj]/2)
            ChipRangeY = slice(int(-cly - Dy0[ii,jj] + yGrid[ii,jj]) , int(cly - Dy0[ii,jj] + yGrid[ii,jj]))
            ChipI = I2[ChipRangeY,ChipRangeX]
            
            if CoreType=='CNN':
                SearchRangeX = slice(int(-clx - SearchLimitX[ii,jj] + xGrid[ii,jj]) , int(clx + SearchLimitX[ii,jj] - 1 + xGrid[ii,jj]))
                SearchRangeY = slice(int(-cly - SearchLimitY[ii,jj] + yGrid[ii,jj]) , int(cly + SearchLimitY[ii,jj] - 1 + yGrid[ii,jj]))
                RefI = I1[SearchRangeY,SearchRangeX]
            elif CoreType=='CFF-O':
#                RefI = I1[ChipRangeY,ChipRangeX]
                SearchRangeX = slice(int(-clx - SearchLimitX[ii,jj] + xGrid[ii,jj]) , int(clx + SearchLimitX[ii,jj] - 1 + xGrid[ii,jj]))
                SearchRangeY = slice(int(-cly - SearchLimitY[ii,jj] + yGrid[ii,jj]) , int(cly + SearchLimitY[ii,jj] - 1 + yGrid[ii,jj]))
                RefI = I1[SearchRangeY,SearchRangeX]
                ChipI = np.lib.pad(ChipI,((int(SearchLimitY[ii,jj]),int(SearchLimitY[ii,jj])-1),(int(SearchLimitY[ii,jj]),int(SearchLimitY[ii,jj])-1)),mode='constant',constant_values=0)
            else:
                sys.exit('No valid coretype selected')
                
#            if CoreType=='CNN':
            #schuift pixelwaarde op tot alles positief is
            minChipI = np.min(ChipI)
            if minChipI < 0:
                ChipI = ChipI - minChipI
            if np.all(ChipI == ChipI[0,0]):
                continue

            minRefI = np.min(RefI)
            if minRefI < 0:
                RefI = RefI - minRefI
            if np.all(RefI == RefI[0,0]):
                continue
            
            if SubPixFlag:
                # call Python
                if CoreType=='CNN':
                    Dx1[ii], Dy1[ii]= SubPixDisp_CNN(ChipI,RefI,OverSampleRatio)
                elif CoreType=='CFF-O':
                    Dx1[ii], Dy1[ii], error1[ii] = SubPixDisp_CFF(ChipI,RefI)
            else:
                # call Python
                if CoreType=='CNN':
                    Dx1[ii], Dy1[ii] = PixDisp_CNN(ChipI,RefI)
                elif CoreType=='CFF-O':
                    Dx1[ii], Dy1[ii] = PixDisp_CFF(ChipI,RefI)
    
    if CoreType=='CNN':
        # add back 1) I1 (RefI) relative to I2 (ChipI) initial offset Dx0 and Dy0, and
        #          2) RefI relative to ChipI has a left/top boundary offset of -SearchLimitX and -SearchLimitY
        idx = np.logical_not(np.isnan(Dx))
        Dx[idx] += (Dx0[idx] - SearchLimitX[idx])
        Dy[idx] += (Dy0[idx] - SearchLimitY[idx])


    
    Dy = -Dy

    print('stop core')

    
    return Dx, Dy, error

def colfilt(A, kernelSize, option):
    
    #form a band with width (int((kernelSize[0]-1)/2) with np.nan around Dx, Dy
    A = np.lib.pad(A,((int((kernelSize[0]-1)/2),int((kernelSize[0]-1)/2)),(int((kernelSize[1]-1)/2),int((kernelSize[1]-1)/2))),mode='constant',constant_values=np.nan)
    
    #makes a moving window view of all possible options array(number kernelelements, number of moved shifts)
    B = viewW(A, kernelSize).reshape(-1,kernelSize[0]*kernelSize[1]).T[:,::1]
    
    #same as Dx,Dy
    output_size = (A.shape[0]-kernelSize[0]+1,A.shape[1]-kernelSize[1]+1)

    
    C = np.zeros(output_size,dtype=A.dtype)
    if option == 0:#    max
        C = np.nanmax(B,axis=0).reshape(output_size)
    elif option == 1:#  min
        C = np.nanmin(B,axis=0).reshape(output_size)
    elif option == 2:#  mean
        C = np.nanmean(B,axis=0).reshape(output_size)
    elif option == 3:#  median
        C = np.nanmedian(B,axis=0).reshape(output_size)
    elif option == 4:#  range
        C = np.nanmax(B,axis=0).reshape(output_size) - np.nanmin(B,axis=0).reshape(output_size)
    elif option == 6:#  MAD (Median Absolute Deviation)
        m = B.shape[0]
        D = np.abs(B - np.dot(np.ones((m,1),dtype=A.dtype), np.array([np.nanmedian(B,axis=0)])))
        C = np.nanmedian(D,axis=0).reshape(output_size)
    elif option[0] == 5:#  displacement distance count with option[1] being the threshold
        m = B.shape[0]
        
#        print(m)
        c = int(np.round((m + 1) / 2)-1)
#        print(c)
#        D shows how different each element is compared to their 5x5 neighbour
        D = np.abs(B - np.dot(np.ones((m,1),dtype=A.dtype), np.array([B[c,:]])))
        print(D[:,0])
#        print(D.shape)
        #here we choose what to throw away! 
        C = np.sum(D<option[1],axis=0).reshape(output_size)
#        print(C)
    else:
        sys.exit('invalid option for columnwise neighborhood filtering')

    C = C.astype(A.dtype)

    return C

    
def filtDisp(Dx, Dy, SearchLimitX, SearchLimitY, M, FracValid, FracSearch, FiltWidth, Iter, MadScalar):
    
    
    dToleranceX = FracValid * FiltWidth**2 #8
    dToleranceY = FracValid * FiltWidth**2
#        pdb.set_trace()
    Dx = Dx / SearchLimitX
    Dy = Dy / SearchLimitY
    
    
    
    for i in range(Iter):
        Dx[np.logical_not(M)] = np.nan #sets all isnan values to false
        Dy[np.logical_not(M)] = np.nan
        #If a displacement elements is similar to its 5x5 neibours with less difference then 25percent, for at least 8 of these elements...> Mask is set at true
        M = (colfilt(Dx, (FiltWidth, FiltWidth), (5,FracSearch)) >= dToleranceX) & (colfilt(Dy, (FiltWidth, FiltWidth), (5,FracSearch)) >= dToleranceY)


    for i in range(np.max([Iter-1,1])):
        Dx[np.logical_not(M)] = np.nan
        Dy[np.logical_not(M)] = np.nan
        
        #calculates deviation from median of pixel group (gemiddelde afwijking van alle afwijking t.o.v. gemiddelde in pixelgroep)
        DxMad = colfilt(Dx, (FiltWidth, FiltWidth), 6)
        DyMad = colfilt(Dy, (FiltWidth, FiltWidth), 6)
        
        #calculates median of pixelgroup
        DxM = colfilt(Dx, (FiltWidth, FiltWidth), 3)
        DyM = colfilt(Dy, (FiltWidth, FiltWidth), 3)
        
        #Filter based on local deviation compared to local group deviation
        M = (np.abs(Dx - DxM) <= (MadScalar * DxMad)) & (np.abs(Dy - DyM) <= (MadScalar * DyMad)) & M
    return M




def bwareaopen(image,size1):
        
    # now identify the objects and remove those above a threshold
    labels, N = measure.label(image,connectivity=2,return_num=True)
    label_size = [(labels == label).sum() for label in range(N + 1)]
    
    # now remove the labels
    for label,size in enumerate(label_size):
        if size < size1:
            image[labels == label] = 0

    return image

