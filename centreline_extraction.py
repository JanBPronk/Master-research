# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:47:46 2020

@author: janbo

Here I automate centreline pixel substraction....
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, Polygon
from fiona.crs import from_epsg
from math import isnan


#open the velocity results and other dataset to extract along centreline (TIF format)
with rasterio.open('C:/Users/janbo/OneDrive/Documenten/GitHub/GlacierVelocities/results_velocity/3km2glaciers/merge/utm45/SENTINEL_16__offset_CFF-O2017_2019.tif') as src:
    vel=src.read(1)
    vbounds=src.bounds
    src.close()  

with rasterio.open('C:/Users/janbo/OneDrive/Documenten/GitHub/GlacierVelocities/results_velocity/3km2glaciers/merge/utm45/SENTINEL_16__err95CFF-O2017_2019.tif') as src:
    uncertainty=src.read(1)
    src.close()

with rasterio.open('E:/personal_shapefiles/merge/ALOS_30m_merge_UTM45.tif') as src:
    dem=src.read(1)
    src.close()
    
with rasterio.open('C:/Users/janbo/OneDrive/Documenten/GitHub/GlacierVelocities/results_velocity/3km2glaciers/merge/utm45/SENTINEL_16__err95_DyCFF-O2017_2019.tif') as src:
    thickness=src.read(1)
    src.close()
   
with rasterio.open('C:/Users/janbo/OneDrive/Documenten/GitHub/GlacierVelocities/results_velocity/3km2glaciers/merge/utm45/SENTINEL_16__err95_DxCFF-O2017_2019.tif') as src:
    dh=src.read(1)
    src.close()    
 
   
dem1 = rasterio.open('E:/personal_shapefiles/merge/ALOS_30m_merge_UTM45.tif')
thickness1 = rasterio.open('C:/Users/janbo/OneDrive/Documenten/GitHub/GlacierVelocities/results_velocity/3km2glaciers/merge/utm45/SENTINEL_16__err95_DyCFF-O2017_2019.tif')
dh1=rasterio.open('C:/Users/janbo/OneDrive/Documenten/GitHub/GlacierVelocities/results_velocity/3km2glaciers/merge/utm45/SENTINEL_16__err95_DxCFF-O2017_2019.tif')
vel1 = rasterio.open('C:/Users/janbo/OneDrive/Documenten/GitHub/GlacierVelocities/results_velocity/3km2glaciers/merge/utm45/SENTINEL_16__offset_CFF-O2017_2019.tif')
uncertainty1 = rasterio.open('C:/Users/janbo/OneDrive/Documenten/GitHub/GlacierVelocities/results_velocity/3km2glaciers/merge/utm45/SENTINEL_16__err95CFF-O2017_2019.tif')




shapefile_centrelines = gpd.read_file('E:/personal_shapefiles/merge/glacier_centrelines_3km2.shp')
shapefile_centrelines=shapefile_centrelines.cx[vbounds[0]:vbounds[2],vbounds[3]:vbounds[1]]


#%%
shp_cl = shapefile_centrelines.loc[shapefile_centrelines['MAIN'] == 1]

#%%
#here a gaussian kernel is produced
def gkern(l=3, sig=0.7):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    print(xx)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    
    print(np.square(xx) + np.square(yy))

    return kernel 

#%%
    
### here the data gets extracted.
kernel=gkern(3,0.7)

Stepsize=80 #m
Idlist0=[]
Idlist1=[]
labels=[]
data=[]

for i in range(shp_cl.shape[0]): #select a centreline
    line=shp_cl.iloc[i]
    
    x_utm=[]
    y_utm=[]
    height=[]
    velocity=[]
    error=[] #ERROR
    thick=[]
    mbalance=[]
    
    xx=[np.nan]*400 #create empty arrays
    yy=[np.nan]*400
    zz=[np.nan]*400
    vv=[np.nan]*400
    rr=[np.nan]*400 ##error
    hh=[np.nan]*400
    mb=[np.nan]*400
    
    
    
    for j in range(len(line['geometry'].coords)-1): #move along centreline

        if j==0:
            r1=0
            dx1=0
            dy1=0
            Ne1=0
                        
        #update values
        dx0=dx1
        dy0=dy1
        r0=r1
        Ne0=Ne1
                
        coord=list(line['geometry'].coords)
        x0=line['geometry'].coords[j][0]
        y0=line['geometry'].coords[j][1]
        x1=line['geometry'].coords[j+1][0]
        y1=line['geometry'].coords[j+1][1]
        dx1=x1-x0
        dy1=y1-y0
        Ls1=np.sqrt((dx1)**2 + (dy1)**2) #length segment between two geonometry points
        Ls1=Ls1+r0*Stepsize #add leftover part from preveous length segment
        Ne1=Ls1/Stepsize #cacluate number of points were're calculating
        N=int(Ne1)
        r1=(Ne1)-N

        
        for k in range(N): 
            if k==0:
                x=x0
                y=y0
             
            #coordinates to extract from               
            if j==0: #when first item along glacier line Ne0==0 per definition and has to be avoided
                x=x+dx1/Ne1
                y=y+dy1/Ne1
            else:    
                x=x+dx1/Ne1-(dx0*r0/Ne0)
                y=y+dy1/Ne1-(dy0*r0/Ne0)
            
            row0, col0 = vel1.index(x, y)
            row1, col1 = dem1.index(x, y)
            row2, col2 = thickness1.index(x, y)
            row3, col3 = uncertainty1.index(x, y)
            row4, col4 = dh1.index(x, y)
                        
            v=np.zeros
            v=vel[row0-1:row0+2, col0-1:col0+2]
            v[abs(v)>1000]=np.nan
            er=uncertainty[row3-1:row3+2,col3-1:col3+2]
            er[abs(er)>1000]=np.nan
            v=np.nansum(v*kernel/np.square(er))/(np.nansum(kernel/np.square(er))) #spatial filter using gaussian kernel and uncertainty
            er=np.nansum(er*kernel/np.square(er))/(np.nansum(kernel/np.square(er)))
            

            
            z=dem[row1, col1]
            h=thickness[row2, col2]
            
            
            #
            msbl=dh[row4-1:row4+2,col4-1:col4+2]
            msbl[abs(msbl)>1000]=np.nan
            msbl=np.nansum(msbl*kernel)/(np.nansum(kernel))
 
           
            if z<0:
                z=np.nan
                       
            x_utm.append(x)
            y_utm.append(y)
            velocity.append(v)
            height.append(z)
            thick.append(h)
            error.append(er)
            mbalance.append(msbl)
            
            
    #reverses the list in case the list is generated in the wrong order        
    if next((x for x in height if not isnan(x)), 1) < next((x for x in height[::-1] if not isnan(x)), 0):
       print(line['RGIID']) 
       height=height[::-1]
       velocity=velocity[::-1]
       error=error[::-1]
       mbalance=mbalance[::-1]
       thick=thick[::-1]
       x_utm=x_utm[::-1]
       y_utm=y_utm[::-1]         

            
    xx[:len(x_utm)]=x_utm
    yy[:len(y_utm)]=y_utm
    zz[:len(height)]=height
    vv[:len(velocity)]=velocity
    rr[:len(error)]=error
    hh[:len(thick)]=thick
    mb[:len(mbalance)]=mbalance
    
    
        
    Idlist0.append(line['RGIID'])
    Idlist1.extend([line['RGIID'],line['RGIID'],line['RGIID'],line['RGIID'],line['RGIID'], line['RGIID'], line['RGIID']])
    labels.extend(['x', 'y', 'velocity (m/year)','conf_95 (m/year)', 'elevation (m.a.s.l)', 'ice_thickness (m)', 'elevation change (dh)'])
    data.extend([xx,yy,vv,rr,zz,hh,mb])
    arrays=[Idlist1, labels]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['ID', 'second'])

data=np.asarray(data).T   
df = pd.DataFrame(data, columns=index)
#df.to_csv('E:/personal_shapefiles/merge/ITS_LIVE_240_centreline_profiles_weighted_3km2.csv') Safe data

