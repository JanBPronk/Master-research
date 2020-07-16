# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:14:49 2020

@author: janbo
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys


stats=pd.read_csv('E:/personal_shapefiles/merge/statistics_3km2.csv', index_col=2)


def plotfunction(data, subdata, region, glacierrange, plottype, glaciertype=None, seperate_supra=False):
#   data:        centrelinedatasets, velocity, dh/dt, DEMs etc..
#   subdata:     choose variable you want to plot
#   region:      choose subregion
#   glacierrange: Choose size range
#   plottype:     subselection depending on terminus type and surface cover is all covered in 'stats' variable
#   glaciertype: 
#   seperate_supra: If True, glaciers with supra glacial lakes will be plotted seperately  
    
#   returns:
#       v_50_df, norm_lake_df, norm_land_df, valid_lake_df, valid_land_df, area_lake, area_land, area_df, length_df, length_land_df, length_lake_df, ELA_lake_df, ELA_land_df, norm_supra_df, length_supra_df
#   v_50_df: velocity data ablation zone(all upscaled/downscaled to 50 points)
#   norm_lake_df: same for lakes
#   norm_land_df: same for land
#   valid_xxx : count valid estimates
#   area_land: Return glacier area of land-terminating glacier
#   area_lake: Same for lake-t
#   length:   Length ablation zone




    
    idx = pd.IndexSlice
    n_columns=7

    
    def linterp(item, length):
        y = item.interpolate()
        x = np.linspace(0, len(y)-1, len(y))
        xvals=np.linspace(0, len(y)-1, length)
        f = np.interp(xvals, x, y)
        return f

    
    v_50_df=pd. DataFrame() 
    norm_lake_df=pd. DataFrame() 
    norm_land_df=pd. DataFrame() 
    valid_lake_df=pd. DataFrame() 
    valid_land_df=pd. DataFrame()
    area_lake=[]
    area_land=[]
    area_df=pd. DataFrame()
    length_land_df=pd. DataFrame() 
    length_lake_df=pd. DataFrame() 
    length_df=pd. DataFrame() 
    ELA_lake_df=pd. DataFrame() 
    ELA_land_df=pd. DataFrame() 

    norm_supra_df=pd. DataFrame() 
    area_supra=[]
    length_supra_df=pd. DataFrame()     
    valid_supra_df=pd. DataFrame()  
    ELA_supra_df=pd. DataFrame()  
    
    for j in range(int(data.shape[1]/n_columns)):  
        
        glacierID=data.columns[j*n_columns][0]
        
        v_item=data.loc[idx[:], idx[glacierID, subdata]]
        vel_item=data.loc[idx[:], idx[glacierID, 'velocity (m/year)']]
        conf_item=data.loc[idx[:], idx[glacierID, 'conf_95 (m/year)']]
        z_item=data.loc[idx[:], idx[glacierID, 'elevation (m.a.s.l)']]
        
        if plottype=='lakevsland' and (glaciertype=='land' or glaciertype=='lake'):
            sys.exit('Cannot create lakevsland plot when choosing lake or land glaciers')
            
        if plottype=='debrisvsclean' and (glaciertype=='debris' or glaciertype=='clean'):
            sys.exit('Cannot create debrisvsclean plot when choosing debris or clean glaciers')

            
        #filter glacier selection
        if stats.loc[idx[glacierID, 'region']] not in region:
#            print('fuck it')
            continue

        if glaciertype=='clean':            
            if stats.loc[idx[glacierID, 'debris_jb']]==1: 
                continue
        if glaciertype=='debris':            
            if stats.loc[idx[glacierID, 'debris_jb']]==0: 
                continue
        if glaciertype=='lake':            
            if stats.loc[idx[glacierID, 'lake_type']]!=1: #!=1
                continue
        if glaciertype=='land':            
            if stats.loc[idx[glacierID, 'lake_type']]==1: #==1
                continue
        
#        #filter out glaciers with small area/length ratio
#        if stats.loc[idx[glacierID, 'rgi_area_km2']]/stats.loc[idx[glacierID, 'longuest_centerline_km']]>1 and stats.loc[idx[glacierID, 'lake_type']]!=1:
#            continue
#    
    
        #filter for glacier size
        if len(glacierrange)==2:            
            if glacierrange[0]>stats.loc[idx[glacierID, 'rgi_area_km2']] or glacierrange[1]<stats.loc[idx[glacierID, 'rgi_area_km2']]:
                continue
        else:
            if glacierrange>stats.loc[idx[glacierID, 'rgi_area_km2']]:
                continue


    
        #iterate along glacier    
        for i in range(len(v_item)):
            
            if np.isnan(v_item[len(v_item)-i-1]):    
                continue
            else:
                conf_item=conf_item[:(len(v_item)-i)]
                vel_item=vel_item[:(len(v_item)-i)]
                v_item=v_item[:(len(v_item)-i)]

                
                
                med_elev=stats.loc[idx[glacierID, 'dem_med_elev']]
                
                v_item[z_item>med_elev]=np.nan
                conf_item[z_item>med_elev]=np.nan
                vel_item[z_item>med_elev]=np.nan

                if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                    v_item[abs(v_item)>500]=np.nan  #to get rid of this strange stuff 
                    conf_item[abs(conf_item)>500]=np.nan
                    vel_item[abs(vel_item)>500]=np.nan

                
                for k in range(len(v_item)):
                    if np.isnan(v_item[k]):
                        continue
                    else:
                        conf_item=conf_item[k:] 
                        v_item=v_item[k:] 
                        vel_item=vel_item[k:] 
                        break  
                if subdata=='velocity (m/year)':
                    v_item[conf_item>5]=np.nan #delete estimetes with error>5myear
                    vel_item[conf_item>5]=np.nan
                    
                if subdata=='elevation (m.a.s.l)':
                    elev_item=v_item.copy()
                    v_item=pd.Series(np.gradient(v_item, 80, edge_order=1))
                    if glacierID== 'RGI60-15.05181':
                        v_item[:]=np.nan

                length_df[str(glacierID)]=[(len(v_item)*80)]
       
                f=linterp(v_item, 50)
                
                fvel=linterp(vel_item, 50)
                
                v_50_df[str(glacierID)]=f
                area_df[str(glacierID)]=np.array([(stats.loc[idx[glacierID, 'rgi_area_km2']])])    
                

                if plottype=='lakevsland':
                    if seperate_supra==False: 
                    
                        if stats.loc[idx[glacierID, 'lake_type']]==1:
    #                        if glacierID not in goodglaciers:
    #                            break
    #                        print('nobreak')
        #                if stats.loc[idx[glacierID, 'debris_jb']]==1:
            #                plt.plot(np.linspace(0,1, len(v_item)), v_item, color='C0', alpha=0.2)
#                            if np.mean(np.asarray(fvel)[40:45])>np.mean(np.asarray(fvel)[47:50]):
#                                print(glacierID)
#                                break

            
                            f=linterp(v_item, 50)
                            
                            norm_lake_df[str(glacierID)]=f
    
                            area_lake.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                            length_lake_df[str(glacierID)]=[(len(v_item)*80)]
            
                            #quantify non valid values:
                            if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                                valid=v_item.copy()
                                valid[~np.isnan(v_item[:])]=1
                                valid[valid!=1]=0                
                                f=linterp(valid, 50)
                                valid_lake_df[str(glacierID)]=f
    #                        if subdata=='elevation (m.a.s.l)':
                                ELA_lake_df[str(glacierID)]=[np.array(med_elev)]
                        elif stats.loc[idx[glacierID, 'lake_type']]==0 or stats.loc[idx[glacierID, 'lake_type']]==2:
    #                    elif stats.loc[idx[glacierID, 'lake_type']]==2:
    
                            f=linterp(v_item, 50)
                            norm_land_df[str(glacierID)]=f
                            area_land.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                            length_land_df[str(glacierID)]=[(len(v_item)*80)]
        
                            #quantify non valid values:
                            if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                                valid=v_item.copy()
                                valid[~np.isnan(v_item[:])]=1
                                valid[valid!=1]=0                
                                f=linterp(valid, 50)
                                valid_land_df[str(glacierID)]=f 
    #                        if subdata=='elevation (m.a.s.l)':
                                ELA_land_df[str(glacierID)]=[np.array(med_elev)]
                    if seperate_supra==True: 
                        if stats.loc[idx[glacierID, 'lake_type']]==1:
    #                        if glacierID not in goodglaciers:
    #                            break
    #                        print('nobreak')
        #                if stats.loc[idx[glacierID, 'debris_jb']]==1:
            #                plt.plot(np.linspace(0,1, len(v_item)), v_item, color='C0', alpha=0.2)
#                            if np.mean(np.asarray(fvel)[40:45])>np.mean(np.asarray(fvel)[47:50]):
#                                break
                            
#                            print(glacierID)
            
                            f=linterp(v_item, 50)
                            
                            norm_lake_df[str(glacierID)]=f
    
                            area_lake.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                            length_lake_df[str(glacierID)]=[(len(v_item)*80)]
            
                            #quantify non valid values:
                            if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                                valid=v_item.copy()
                                valid[~np.isnan(v_item[:])]=1
                                valid[valid!=1]=0                
                                f=linterp(valid, 50)
                                valid_lake_df[str(glacierID)]=f
    #                        if subdata=='elevation (m.a.s.l)':
                                ELA_lake_df[str(glacierID)]=[np.array(med_elev)]
                        elif stats.loc[idx[glacierID, 'lake_type']]==0:
    #                    elif stats.loc[idx[glacierID, 'lake_type']]==2:
    
                            f=linterp(v_item, 50)
                            norm_land_df[str(glacierID)]=f
                            area_land.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                            length_land_df[str(glacierID)]=[(len(v_item)*80)]
        
                            #quantify non valid values:
                            if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                                valid=v_item.copy()
                                valid[~np.isnan(v_item[:])]=1
                                valid[valid!=1]=0                
                                f=linterp(valid, 50)
                                valid_land_df[str(glacierID)]=f 
    #                        if subdata=='elevation (m.a.s.l)':
                                ELA_land_df[str(glacierID)]=[np.array(med_elev)]
                        elif stats.loc[idx[glacierID, 'lake_type']]==2:
    #                    elif stats.loc[idx[glacierID, 'lake_type']]==2:
#                            print(glacierID)
                            f=linterp(v_item, 50)
                            norm_supra_df[str(glacierID)]=f
                            area_supra.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                            length_supra_df[str(glacierID)]=[(len(v_item)*80)]
        
                            #quantify non valid values:
                            if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                                valid=v_item.copy()
                                valid[~np.isnan(v_item[:])]=1
                                valid[valid!=1]=0                
                                f=linterp(valid, 50)
                                valid_supra_df[str(glacierID)]=f 
    #                        if subdata=='elevation (m.a.s.l)':
                                ELA_supra_df[str(glacierID)]=[np.array(med_elev)]
    
                if plottype=='debrisvsclean':
                    if stats.loc[idx[glacierID, 'debris_jb']]==1:
        #                plt.plot(np.linspace(0,1, len(v_item)), v_item, color='C0', alpha=0.2)

                        f=linterp(v_item, 50)
                        
                        norm_lake_df[str(glacierID)]=f
                        area_lake.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                        length_lake_df[str(glacierID)]=[(len(v_item)*80)]
        
                        #quantify non valid values:
                        if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                            valid=v_item.copy()
                            valid[~np.isnan(v_item[:])]=1
                            valid[valid!=1]=0                
                            f=linterp(valid, 50)
                            valid_lake_df[str(glacierID)]=f
                        
                    elif stats.loc[idx[glacierID, 'debris_jb']]==0:
        #                plt.plot(np.linspace(0,1, len(v_item)), v_item, color='C3', alpha=0.2)

                        f=linterp(v_item, 50)
                        norm_land_df[str(glacierID)]=f
                        area_land.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                        length_land_df[str(glacierID)]=[(len(v_item)*80)]

                        #quantify non valid values:
                        if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                            valid=v_item.copy()
                            valid[~np.isnan(v_item[:])]=1
                            valid[valid!=1]=0                
                            f=linterp(valid, 50)
                            valid_land_df[str(glacierID)]=f 
                break
            
    return v_50_df, norm_lake_df, norm_land_df, valid_lake_df, valid_land_df, area_lake, area_land, area_df, length_df, length_land_df, length_lake_df, ELA_lake_df, ELA_land_df, norm_supra_df, length_supra_df


stats=pd.read_csv('E:/personal_shapefiles/merge/statistics_3km2.csv', index_col=2)


def normplotfunction(data, subdata, region, glacierrange, plottype, glaciertype=None):

    idx = pd.IndexSlice
    n_columns=7

    
    def linterp(item, length):
        y = item.interpolate()
        x = np.linspace(0, len(y)-1, len(y))
        xvals=np.linspace(0, len(y)-1, length)
        f = np.interp(xvals, x, y)
        return f

    
    v_50_df=pd. DataFrame() 
    norm_lake_df=pd. DataFrame() 
    norm_land_df=pd. DataFrame() 
    valid_lake_df=pd. DataFrame() 
    valid_land_df=pd. DataFrame()
    area_lake=[]
    area_land=[]
    area_df=pd. DataFrame()
    length_land_df=pd. DataFrame() 
    length_lake_df=pd. DataFrame() 
    length_df=pd. DataFrame() 
    median_z_abl_lake_df=pd. DataFrame() 
    median_z_abl_land_df=pd. DataFrame() 

    
    for j in range(int(data.shape[1]/n_columns)):  
        
        glacierID=data.columns[j*n_columns][0]
        v_item=data.loc[idx[:], idx[glacierID, subdata]]
        conf_item=data.loc[idx[:], idx[glacierID, 'conf_95 (m/year)']]
        z_item=data.loc[idx[:], idx[glacierID, 'elevation (m.a.s.l)']]
        
        if plottype=='lakevsland' and (glaciertype=='land' or glaciertype=='lake'):
            sys.exit('Cannot create lakevsland plot when choosing lake or land glaciers')
            
        if plottype=='debrisvsclean' and (glaciertype=='debris' or glaciertype=='clean'):
            sys.exit('Cannot create debrisvsclean plot when choosing debris or clean glaciers')
        
#        if glacierID=='RGI60-15.02371' or glacierID=='RGI60-15.10286':
#            continue
            
        #filter glacier selection
        if stats.loc[idx[glacierID, 'region']] not in region:
#            print('fuck it')
            continue

        if glaciertype=='clean':            
            if stats.loc[idx[glacierID, 'debris_jb']]==1: 
                continue
        if glaciertype=='debris':            
            if stats.loc[idx[glacierID, 'debris_jb']]==0: 
                continue
        if glaciertype=='lake':            
            if stats.loc[idx[glacierID, 'lake_type']]!=1: #!=1
                continue
        if glaciertype=='land':            
            if stats.loc[idx[glacierID, 'lake_type']]==1: #==1
                continue
        
#        #filter out glaciers with small area/length ratio
#        if stats.loc[idx[glacierID, 'rgi_area_km2']]/stats.loc[idx[glacierID, 'longuest_centerline_km']]>1 and stats.loc[idx[glacierID, 'lake_type']]!=1:
#            continue
#    
    
        #filter for glacier size
        if len(glacierrange)==2:            
            if glacierrange[0]>stats.loc[idx[glacierID, 'rgi_area_km2']] or glacierrange[1]<stats.loc[idx[glacierID, 'rgi_area_km2']]:
                continue
        else:
            if glacierrange>stats.loc[idx[glacierID, 'rgi_area_km2']]:
                continue


    
        #iterate along glacier    
        for i in range(len(v_item)):
            
            if np.isnan(v_item[len(v_item)-i-1]):    
                continue
            else:
                conf_item=conf_item[:(len(v_item)-i)]
                v_item=v_item[:(len(v_item)-i)]
                
                med_elev=stats.loc[idx[glacierID, 'dem_med_elev']]
                
                v_item[z_item>med_elev]=np.nan
                conf_item[z_item>med_elev]=np.nan
                
                if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                    v_item[abs(v_item)>500]=np.nan  #to get rid of this strange stuff 
                    conf_item[abs(conf_item)>500]=np.nan
                
                for k in range(len(v_item)):
                    if np.isnan(v_item[k]):
                        continue
                    else:
                        conf_item=conf_item[k:] 
                        v_item=v_item[k:] 
                        break  
                if subdata=='velocity (m/year)':
                    v_item[conf_item>5]=np.nan #delete estimetes with error>5myear
                if subdata=='elevation (m.a.s.l)':
                    elev_item=v_item.copy()
                    v_item=pd.Series(np.gradient(v_item, 80, edge_order=1))
                    if glacierID== 'RGI60-15.05181':
                        v_item[:]=np.nan


                length_df[str(glacierID)]=[(len(v_item)*80)]
       
                f=linterp(v_item, 50)
                f=pd.Series(f)
                f=np.asarray(f.sub(0).div((f.max() - 0)))
                
                v_50_df[str(glacierID)]=f
                area_df[str(glacierID)]=np.array([(stats.loc[idx[glacierID, 'rgi_area_km2']])])    
                

                if plottype=='lakevsland':
                    if stats.loc[idx[glacierID, 'lake_type']]==1: 
    #                if stats.loc[idx[glacierID, 'debris_jb']]==1:
        #                plt.plot(np.linspace(0,1, len(v_item)), v_item, color='C0', alpha=0.2)
                        f=linterp(v_item, 50)
                        f=pd.Series(f)
                        f=np.asarray(f.sub(0).div((f.max() - 0)))
                        
                        
                        
#                        if np.mean(np.asarray(f)[40:45])>np.mean(np.asarray(f)[47:50]):
#                            break
                        norm_lake_df[str(glacierID)]=f
                        area_lake.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                        length_lake_df[str(glacierID)]=[(len(v_item)*80)]
        
                        #quantify non valid values:
                        if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                            valid=v_item.copy()
                            valid[~np.isnan(v_item[:])]=1
                            valid[valid!=1]=0                
                            f=linterp(valid, 50)
                            valid_lake_df[str(glacierID)]=f
                        if subdata=='elevation (m.a.s.l)':
                            median_z_abl_lake_df[str(glacierID)]=[np.nanmedian(elev_item)]
                        
                    elif stats.loc[idx[glacierID, 'lake_type']]==0 or stats.loc[idx[glacierID, 'lake_type']]==2:
    #                elif stats.loc[idx[glacierID, 'debris_jb']]==0:
        #                plt.plot(np.linspace(0,1, len(v_item)), v_item, color='C3', alpha=0.2)

                        f=linterp(v_item, 50)
                        f=pd.Series(f)
                        f=np.asarray(f.sub(0).div((f.max() - 0)))
                        
                        norm_land_df[str(glacierID)]=f
                        area_land.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                        length_land_df[str(glacierID)]=[(len(v_item)*80)]
    
                        #quantify non valid values:
                        if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                            valid=v_item.copy()
                            valid[~np.isnan(v_item[:])]=1
                            valid[valid!=1]=0                
                            f=linterp(valid, 50)
                            valid_land_df[str(glacierID)]=f 
                        if subdata=='elevation (m.a.s.l)':
                            median_z_abl_land_df[str(glacierID)]=[np.nanmedian(elev_item)]

                if plottype=='debrisvsclean':
                    if stats.loc[idx[glacierID, 'debris_jb']]==1:
        #                plt.plot(np.linspace(0,1, len(v_item)), v_item, color='C0', alpha=0.2)

                        f=linterp(v_item, 50)
                        f=pd.Series(f)
                        f=np.asarray(f.sub(0).div((f.max() - 0)))
                        
                        norm_lake_df[str(glacierID)]=f
                        area_lake.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                        length_lake_df[str(glacierID)]=[(len(v_item)*80)]
        
                        #quantify non valid values:
                        if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                            valid=v_item.copy()
                            valid[~np.isnan(v_item[:])]=1
                            valid[valid!=1]=0                
                            f=linterp(valid, 50)
                            valid_lake_df[str(glacierID)]=f
                        
                    elif stats.loc[idx[glacierID, 'debris_jb']]==0:
        #                plt.plot(np.linspace(0,1, len(v_item)), v_item, color='C3', alpha=0.2)

                        f=linterp(v_item, 50)
                        f=pd.Series(f)
                        f=np.asarray(f.sub(0).div((f.max() - 0)))
                        norm_land_df[str(glacierID)]=f
                        area_land.append(stats.loc[idx[glacierID, 'rgi_area_km2']])
                        length_land_df[str(glacierID)]=[(len(v_item)*80)]

                        #quantify non valid values:
                        if subdata=='velocity (m/year)' or subdata=='elevation change (dh)':
                            valid=v_item.copy()
                            valid[~np.isnan(v_item[:])]=1
                            valid[valid!=1]=0                
                            f=linterp(valid, 50)
                            valid_land_df[str(glacierID)]=f 
                        
                break
            
    return v_50_df, norm_lake_df, norm_land_df, valid_lake_df, valid_land_df, area_lake, area_land, area_df, length_df, length_land_df, length_lake_df, median_z_abl_lake_df, median_z_abl_land_df
