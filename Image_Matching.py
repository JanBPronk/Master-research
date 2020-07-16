# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:18:38 2019

@author: janbo
"""
import numpy as np
import cv2
from skimage.feature import register_translation



def SubPixDisp_CNN(ChipI,RefI,OverSampleRatio):
    lenC=np.shape(ChipI)[0]
    widC=np.shape(ChipI)[1]
    lenR=np.shape(RefI)[0]
    widR=np.shape(RefI)[1]
    result_cols=widR-widC + 1
    result_rows=lenR-lenC + 1
    result=cv2.matchTemplate(RefI, ChipI, cv2.TM_CCORR_NORMED)
    _, max_val,_, max_loc=cv2.minMaxLoc(result)
    maxloc_x=max_loc[0]
    maxloc_y=max_loc[1]
   
    #refine the offset at the sub-pixel level using image upsampling (pyramid algorithm): extract 5x5 small image at the coarse offset location
    x_start=np.maximum(maxloc_x-2, 0)
    x_start=np.minimum(x_start, result_cols-5)
    x_count=5
    
    y_start=np.maximum(maxloc_y-2, 0)
    y_start=np.minimum(y_start, result_rows-5)
    y_count=5
       
    result_small=result[int(y_start):int(y_start+y_count),int(x_start):int(x_start+x_count)].copy()
    cols=result_small[1]
    rows=result_small[0]
    overSampleFlag = 1
    
    predecessor_small = result_small.copy()
    
    while (overSampleFlag < OverSampleRatio):
        cols *= 2
        rows *= 2
        overSampleFlag *= 2
    
        matrix=cv2.pyrUp(predecessor_small)
        predecessor_small=matrix
        
    _, max_value,_, max_loc=cv2.minMaxLoc(matrix)

    
    maxlocsmall_x=max_loc[0]
    maxlocsmall_y=max_loc[1]
    

    Dx1=(maxlocsmall_x/OverSampleRatio+x_start).astype(np.float32)
    Dy1=(maxlocsmall_y/OverSampleRatio+y_start).astype(np.float32)
       
    return Dx1, Dy1

 
def PixDisp_CNN(ChipI,RefI):
    
    result=cv2.matchTemplate(RefI, ChipI, cv2.TM_CCORR_NORMED)
    _, max_val,_, max_loc=cv2.minMaxLoc(result)
    Dx1=max_loc[0]
    Dy1=max_loc[1]
    
    return Dx1, Dy1

def PixDisp_CFF(ChipI,RefI):
#    F0=np.fft.fft2(RefI)
#    
#    G0_cc=np.conj(np.fft.fft2(ChipI))
#    
#    P=np.real(np.fft.ifft2((F0*G0_cc)/(abs(F0*G0_cc)))).astype(np.float32)
#    
#    _, max_val,_, max_loc=cv2.minMaxLoc(P)
#    
#
#    Dx1=max_loc[0]
#    Dy1=max_loc[1]
#    
#    P0=P.copy()
#    P0[:,:int(len(P[0])/2)]=P[:,int(len(P[0])/2):]
#    P0[:,int(len(P[0])/2):]=P[:,:int(len(P[0])/2)]
#    P00=P0.copy()
#    P00[:int(len(P[0])/2),:]=P0[int(len(P[0])/2):,:]
#    P00[int(len(P[0])/2):,:]=P0[:int(len(P[0])/2),:]
#
#    _, max_val,_, max_loc=cv2.minMaxLoc(P00)
#
#    Dx1=max_loc[0]-len(P[0])/2
#    Dy1=max_loc[1]-len(P[0])/2
    
    shift, error, diffphase = register_translation(RefI, ChipI)  
    Dx1=shift[1]
    Dy1=shift[0]
    
    return Dx1, Dy1



def SubPixDisp_CFF(ChipI,RefI):
#    F0=np.fft.fft2(RefI)
#    G0_cc=np.conj(np.fft.fft2(ChipI))
#    P=np.real(np.fft.ifft2((F0*G0_cc)/(abs(F0*G0_cc)))).astype(np.float32)
#        
#    P0=P.copy()
#    P0[:,:int(len(P[0])/2)]=P[:,int(len(P[0])/2):]
#    P0[:,int(len(P[0])/2):]=P[:,:int(len(P[0])/2)]
#    P00=P0.copy()
#    P00[:int(len(P[0])/2),:]=P0[int(len(P[0])/2):,:]
#    P00[int(len(P[0])/2):,:]=P0[:int(len(P[0])/2),:]
#    
#    
#    _, max_val,_, max_loc=cv2.minMaxLoc(P00)
    
#    if max_loc[0]>=(len(P00[0])-1) or max_loc[1]>=(len(P00[0])-1):
#        Dx1=max_loc[0]-len(P[0])/2
#        Dy1=max_loc[1]-len(P[0])/2
#    else:
#        xsub=(P00[max_loc[1],max_loc[0]+1])-(P00[max_loc[1],max_loc[0]-1]) / 2*(2*P00[max_loc[1],max_loc[0]]-P00[max_loc[1],max_loc[0]+1]-P00[max_loc[1],max_loc[0]-1])
#        ysub=(P00[max_loc[1]+1,max_loc[0]])-(P00[max_loc[1]-1,max_loc[0]]) / 2*(2*P00[max_loc[1],max_loc[0]]-P00[max_loc[1]+1,max_loc[0]]-P00[max_loc[1]-1,max_loc[0]])
#        
#        Dx1=max_loc[0]-len(P[0])/2+xsub
#        Dy1=max_loc[1]-len(P[0])/2+ysub
    

##    shift, error1, diffphase = register_translation(RefI, ChipI)
    shift, error, diffphase = register_translation(RefI, ChipI, 16)
    Dx1=shift[1]
    Dy1=shift[0]
    
    return Dx1, Dy1, error





