# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:46:02 2020

@author: janbo

Here I call the matchingframe function. Currently this code might not be working as I found out that a scikit learn package (register translation) is depreciated.
"""

from matchingframe import matchingframe
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import fiona
from fiona.crs import from_epsg
import utm
import gdal
import osr

from Read_Sattelites import Everest_S

#%%   


#choose paths to store and load from
#paths, satelite, region etc.
#region='Poiqu Basin'
#directory='E:/sentinel/'
#area='west_poiqu/'
subdirectory='offset/3km2glaciers/Bhutan_South/sentinel/'
#subdirectoryvel='results_velocity/3km2glaciers/Poiqu_East/'
#NW=[85.63, 28.50] #kangjiaruo
#SE=[85.75, 28.36] #kangjiaruo


pixelsize=10
satellite='SENTINEL_16_'
year=[ '2017', '2018', '2019']
CoreType='CFF-O'
areapath='E:/' #external disk...
chipsize=16 #should be same as minimal chipsize in matchingframe

years, utm0=Everest_S()

##CREATE SHAPEFILE from choosen coordinates.
#utm_NW=utm.from_latlon(NW[1], NW[0])
#utm_SE=utm.from_latlon(SE[1], SE[0])
#newdata=gpd.GeoDataFrame()
#newdata['geometry']=None
#newdata
#coordinates=[(utm_NW[0], utm_SE[1]), (utm_SE[0], utm_SE[1]), (utm_SE[0], utm_NW[1]), (utm_NW[0],utm_NW[1])]
#poly = Polygon(coordinates)
#newdata.loc[0, 'geometry'] = poly
#newdata.loc[0, 'Location'] = region
#newdata.crs = from_epsg(32600+utm0)
#outfp = areapath + "AreaSelection.shp"
#newdata.to_file(outfp)


#SHAPEFILE
shapefile_himalaya = gpd.read_file('E:/glacier_polygon/marge_gandam_UTM45N.shp')
#poiqu=shapefile_himalaya.cx[utm_NW[0]:utm_SE[0],utm_NW[1]:utm_SE[1]]
poiqu=shapefile_himalaya.cx[399960:509760,3100020:2990220] #alternative for large scale, simply take the bounds of the satellite imgage!
poiqu3=poiqu.loc[poiqu['Area'] >= 1] #ditis  heel belangrijk for out_transform bounds
poiqugeom=poiqu3.geometry



#Running the matchingframe script


for k in range(len(years)-1):
#    if k>1:
#        continue
    for i in range(len(years[k])): 
        for j in range(len(years[k+1])):
            Dx,  Dy, InterpMask, ChipSizeX, ChipSizeY, outframe, I1_out_transform, I2_out_transform=matchingframe(years[k][i], years[k+1][j], CoreType, poiqugeom) #areapath instead of poiqugeom....
#            np.savez(subdirectory + satellite + '_offset_' + CoreType + year[k] + str(i) + str(j) + '.npz', Dx=Dx, Dy=Dy, InterpMask=InterpMask, ChipSizeX=ChipSizeX)



#safe the velocity fields in numpy files

##%%
#            
#for k in range(len(years)-1):
#    Dx=[]
#    Dy=[]
#    for i in range(len(years[k])):
#        for j in range(len(years[k+1])):                        
#            npz=np.load(subdirectory + satellite + '_offset_' + CoreType + year[k] + str(i) + str(j) + '.npz')
#            Dx0=npz['Dx'] 
#            Dy0=npz['Dy']
#            Dx.append(Dx0)
#            Dy.append(Dy0)
#            npz.close()
#    np.savez(subdirectoryvel + satellite + '_offset_' + CoreType + year[k] + '.npz', Dx=np.nanmean(Dx, axis=0), Dy=np.nanmean(Dy, axis=0))
#    Dx.clear()
#    Dy.clear()
#
#
##%%
#    
#for k in range(len(years)-1):    
#    Disp=np.load(subdirectoryvel + satellite + '_offset_' + CoreType + year[k] + '.npz')
#    Dx=Disp['Dx'] 
#    Dy=Disp['Dy'] 
#    
#    Dx=Dx#-np.nanmean(Dx)
#    Dy=Dy#-np.nanmean(Dy)
#    D=(np.sqrt(np.square(Dx)+np.square(Dy)))*pixelsize
#    
#    #create TIF file
#    driver = gdal.GetDriverByName( 'GTiff' )
#    dst_filename = subdirectoryvel + satellite + '_offset_' + CoreType + year[k] + '.tif'
#    dst_ds=driver.Create(dst_filename, D.shape[1], D.shape[0],1, gdal.GDT_Float32)
#    dst_ds.SetGeoTransform([I1_out_transform[2] + ((chipsize/2 + 10)*pixelsize - (chipsize/4)*pixelsize), (chipsize*pixelsize)/2 , 0, I1_out_transform[5] - ((chipsize/2 +10 )*pixelsize - (chipsize/4)*pixelsize), 0, -(chipsize*pixelsize)/2 ])
#    srs = osr.SpatialReference()
#    srs.SetUTM( utm0, 1 )
#    srs.SetWellKnownGeogCS( 'EPSG:326'+'utm0' )
#    dst_ds.SetProjection( srs.ExportToWkt() )
#    raster = D
#    dst_ds.GetRasterBand(1).WriteArray(raster)
#    dst_ds = None
#     
#    ###   DX    ####   
#    Disp=np.load(subdirectoryvel + satellite + '_offset_' + CoreType + year[k] + '.npz')
#    Dx=Disp['Dx'] 
#    Dy=Disp['Dy'] 
#    
#    Dx=(Dx-np.nanmean(Dx))*pixelsize
#    
#    #create TIF file
#    driver = gdal.GetDriverByName( 'GTiff' )
#    dst_filename = subdirectoryvel + satellite + '_offset_Dx' + CoreType + year[k] + '.tif'
#    dst_ds=driver.Create(dst_filename, D.shape[1], D.shape[0],1, gdal.GDT_Float32)
#    #dst_ds.SetGeoTransform([I1_out_transform[2]+ ( (chipsize+10)*pixelsize -(chipsize/2)*pixelsize), (chipsize*pixelsize) , 0, I1_out_transform[5]- (chipsize*pixelsize + 10*pixelsize-(chipsize/2)*pixelsize), 0, -(chipsize*pixelsize) ])
#    dst_ds.SetGeoTransform([I1_out_transform[2] + ((chipsize/2 + 10)*pixelsize - (chipsize/4)*pixelsize), (chipsize*pixelsize)/2 , 0, I1_out_transform[5] - ((chipsize/2 +10 )*pixelsize - (chipsize/4)*pixelsize), 0, -(chipsize*pixelsize)/2 ])
#    srs = osr.SpatialReference()
#    srs.SetUTM( utm0, 1 )
#    srs.SetWellKnownGeogCS( 'EPSG:326'+'utm0' )
#    dst_ds.SetProjection( srs.ExportToWkt() )
#    raster = Dx
#    dst_ds.GetRasterBand(1).WriteArray(raster)
#    dst_ds = None
#     
#    #### Dy #########    
#    Disp=np.load(subdirectoryvel + satellite + '_offset_' + CoreType + year[k] + '.npz')
#    Dx=Disp['Dx'] 
#    Dy=Disp['Dy'] 
#    
#    Dy=(Dy-np.nanmean(Dy))*pixelsize
#    
#    #create TIF file
#    driver = gdal.GetDriverByName( 'GTiff' )
#    dst_filename = subdirectoryvel + satellite + '_offset_Dy' + CoreType + year[k] + '.tif'
#    dst_ds=driver.Create(dst_filename, D.shape[1], D.shape[0],1, gdal.GDT_Float32)
#    dst_ds.SetGeoTransform([I1_out_transform[2] + ((chipsize/2 + 10)*pixelsize - (chipsize/4)*pixelsize), (chipsize*pixelsize)/2 , 0, I1_out_transform[5] - ((chipsize/2 +10 )*pixelsize - (chipsize/4)*pixelsize), 0, -(chipsize*pixelsize)/2 ])
#    srs = osr.SpatialReference()
#    srs.SetUTM( utm0, 1 )
#    srs.SetWellKnownGeogCS( 'EPSG:326'+'utm0' )
#    dst_ds.SetProjection( srs.ExportToWkt() )
#    raster = Dy
#    dst_ds.GetRasterBand(1).WriteArray(raster)
#    dst_ds = None
