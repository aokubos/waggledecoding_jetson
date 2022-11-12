import argparse
import sys
import time
import datetime
import math
from scipy import signal
import numpy as np
import cupy as cp
import cupyx
import cv2
from tqdm import trange
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
import astropy.units as u
from osgeo import osr, gdal
import folium

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("duration", type=int, help="recording duration in minute")
    parser.add_argument("--loop", type=int, default=1, help="number of loop")
    parser.add_argument("--iw", type=int, default=1440, help="width pixel of the video input")
    parser.add_argument("--ih", type=int, default=1080, help="height pixel of the video input")
    parser.add_argument("--rs", type=float, default=7.5, help="resize ratio of input to analyzing resolution")
    parser.add_argument("--lat", type=float, default=36.020635, help="latitude of the recording site in degree")
    parser.add_argument("--lon", type=float, default=140.120258, help="longititude of the recording site in degree")
    parser.add_argument("--prefix", type=str, default="result", help="prefix output result file name")
    parser.add_argument("--peakcut", type=float, default=8.0, help="peak cut threshold for filtering waggle candidates")

    return parser.parse_args()


def gst_pipeline():
    return(
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=%d, height=%d, framerate=120/1, format=GRAY8 ! "
        "queue ! videoscale ! video/x-raw, width=%d, height=%d ! appsink"
    % (
        width, height, rswidth, rsheight
    ))


#Pixcel-based FFT and detect waggle candidates
def calc_fft():
    res[:] = 0
    dat_gpu = cp.asarray(dat)
    F = cupyx.scipy.fftpack.fft(dat_gpu, axis = 0, plan = plan)
    F_abs = cp.abs(F)
    F_abs = F_abs / (N/2)
    F_abs_cpu = cp.asnumpy(F_abs)
    maximal_idx, maximal_idy = signal.argrelmax(F_abs_cpu, axis=0, order=1)
    temp = np.zeros((N,rswidth*rsheight))
    temp[maximal_idx,maximal_idy] = 1
    temp = np.where(F_abs_cpu>peak_cut,1,0)*temp
    temp2 = temp[(minfreq_idx-1):(maxfreq_idx+1),:]
    res[np.sum(temp2,axis=0) > 0] = 255


# Clump candidate pixcels
def clump_DancePix(nframe):
    img = res.reshape([rsheight,rswidth])
    img = cv2.medianBlur(img,3) #remove noise
    nLabel, labelImage, stats, centroids = cv2.connectedComponentsWithStats(img)
    listNLabels[nframe] = nLabel
    cent_x[nframe,0:nLabel] = centroids[:,0]
    cent_y[nframe,0:nLabel] = centroids[:,1]
    return img, labelImage, stats


# Dance continuity ckeck
def check_DanceContinuity(nframe,count,labelimg0,labelimg1):
    for label in range(1, listNLabels[nframe-1]):
        tmp = np.max(labelimg1[labelimg0 == label])
        if(tmp > 0): 
            if(connectF[nframe-1,label] < 1):
                count += 1
                connectF[nframe,tmp] = count
                connectF[nframe-1,label] = count
            else:
                connectF[nframe,tmp] = connectF[nframe-1,label]
    return count


# Reserve start and end waggle point
def calc_DanceInfo(count):
    for i in range(1,count+1):
        crow,ccol = np.where(connectF == i)
        beginF[i-1] = np.min(crow)
        endF[i-1] = np.max(crow)
        beginLabel = np.argmax(connectF[beginF[i-1],:] == i)
        endLabel =  np.argmax(connectF[endF[i-1],:] == i)
        begin_centx[i-1] = cent_x[beginF[i-1],beginLabel]
        begin_centy[i-1] = cent_y[beginF[i-1],beginLabel]
        end_centx[i-1] = cent_x[endF[i-1],endLabel]
        end_centy[i-1] = cent_y[endF[i-1],endLabel]


# Connect segmented waggle run
def connect_DanceGap(count):
    for i in range(0,count-1):
        for j in range(1,count-i):
            if((int(beginF[i+j]) - int(endF[i])) < (CONNECTMXFRAM/N)):
                dist = (begin_centx[i+j]-end_centx[i])**2 + (begin_centy[i+j]-end_centy[i])**2
                if(dist < CONNECTMXDIST**2):
                    endF[i] = endF[i+j]
                    end_centx[i] = end_centx[i+j]
                    end_centy[i] = end_centy[i+j]
                    beginF[i+j] = 0
                    endF[i+j] = 0
                    begin_centx[i+j] = 0
                    begin_centy[i+j] = 0
                    end_centx[i+j] = 0
                    end_centy[i+j] = 0


# Sun azimuth calculation
def SunAzimuth(sec):
    obloc = EarthLocation(lat=oblat*u.deg,lon=oblon*u.deg)
    ttemp = datetime.datetime.fromtimestamp(sec,datetime.timezone.utc)
    t = Time(ttemp)
    sunloc = get_sun(t).transform_to(AltAz(obstime=t,location=obloc))
    return sunloc.az.degree


# Output
def output_DanceInfo(count,fcsvname,fhtmlname):
    f = open(fcsvname, 'w')
    f.writelines('danceID,Frame.start,Frame.end,Duration,X.start,Y.start,X.end,Y.end,Angle2Sun,SunAzimuth,DirectionfromN,Distance,Easting_UTM,Northing_UTM,Longitude,Latitude\n')
    # Foraging map preparation
    m = folium.Map(
        location=[oblat, oblon],
        zoom_start=13
        )
    folium.TileLayer(
        tiles='https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png',
        name='Japan Topo map',
        attr="<a href='https://maps.gsi.go.jp/development/ichiran.html' target='_blank'>GSI-J tiles</a>"
        ).add_to(m)
    folium.TileLayer(
        tiles='https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg',
        name='Japan latest aerial photos',
        attr="<a href='https://maps.gsi.go.jp/development/ichiran.html' target='_blank'>GSI-J tiles</a>"
        ).add_to(m)
    folium.Marker(
        location=[oblat,oblon],
        popup="hive",
        icon=folium.Icon(color='red', icon='asterisk')
        ).add_to(m)
    
    fincount = 0
    for i in trange(0,count):
        tmp_durt = endF[i] - beginF[i] + 1
        if((endF[i] != 0) and (tmp_durt > 2)):
            dancedurt = tmp_durt * N / fps
            direction = np.arctan2((end_centx[i]-begin_centx[i]),-(end_centy[i]-begin_centy[i]))
            azimuth = np.radians(SunAzimuth(starttime + beginF[i]*N/fps))
            dancedirc = azimuth + direction
            if dancedurt <= 1.7130:
                dist = 1000 * (np.log(1 - (dancedurt - a) / b) / c)
            else:
                dist = 1000 * ((dancedurt - d) / e)
            easting = dist * np.sin(dancedirc)
            northing = dist * np.cos(dancedirc)
            # Convert coordination
            forage_e_utm = easting + obloc_utm[0]
            forage_n_utm = northing + obloc_utm[1]
            forage_e_geo, forage_n_geo, z = trans_utm2geo.TransformPoint(forage_e_utm,forage_n_utm)
            danceinfo = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}\n'.format(i,beginF[i],endF[i],dancedurt,begin_centx[i],begin_centy[i],end_centx[i],end_centy[i],direction,azimuth,dancedirc,dist,forage_e_utm,forage_n_utm,forage_e_geo,forage_n_geo)
            f.writelines(danceinfo)
            folium.Marker(location=[forage_n_geo,forage_e_geo]).add_to(m)
            fincount += 1
    
    f.close()
    folium.LayerControl().add_to(m)
    m.save(fhtmlname)
    return fincount


# Main procedure
def main():
    nframe = 0
    count = 0
    fcsvname = fprefix + now.strftime('_%Y%m%d_%H%M%S') + '.csv'
    fhtmlname = fprefix + now.strftime('_%Y%m%d_%H%M%S') + ".html"

    video = cv2.VideoCapture(gst_pipeline(), cv2.CAP_GSTREAMER)
    if not video.isOpened():
        sys.exit(1)
    
    for skip in trange(0, frame_count, N):
        for i in range(N):
            ret, frame = video.read()
            dat[i,:] = frame.reshape(-1)

        calc_fft()

        img, labelImage, stats = clump_DancePix(nframe)

        if nframe < 1:
            labelimg0 = labelImage
        else:
            labelimg1 = labelImage
            count = check_DanceContinuity(nframe,count,labelimg0,labelimg1)
            labelimg0 = labelimg1

        nframe += 1

    video.release()

    calc_DanceInfo(count)

    connect_DanceGap(count)
    
    fincount = output_DanceInfo(count,fcsvname,fhtmlname)
    print('Number of waggle run detected: '+str(fincount))


if __name__ == '__main__':
    args = get_args()
    duration = args.duration
    nloop = args.loop
    width = args.iw
    height = args.ih
    rsratio = args.rs
    oblat = args.lat
    oblon = args.lon
    fprefix = args.prefix
    peak_cut = args.peakcut
    
    rswidth = int(width/rsratio)
    rsheight = int(height/rsratio)

    fps = 120/1 # Frame per second
    N = 20 # Num of frames for FFT
    frame_count = int(fps*60*duration)

    CONNECTMXDIST = 3.0 # Distance threshold for connecting waggle run gap (pixcel value in reshape image)
    CONNECTMXFRAM = 120 # Frame threshold for connecting waggle run gap (frame number)

    # Calculating EPSG to convert UTM coordination in WGS84
    utmzone = int((oblon+180)/6)+1
    if oblat > 0:
        utmepsg = 32600 + utmzone
    else:
        utmepsg = 32700 + utmzone
    geoepsg = 4326 # WGS84 geographical

    geo_srs = osr.SpatialReference()
    utm_srs = osr.SpatialReference()
    geo_srs.ImportFromEPSG(geoepsg)
    utm_srs.ImportFromEPSG(utmepsg)
    trans_geo2utm = osr.CoordinateTransformation(geo_srs, utm_srs)
    trans_utm2geo = osr.CoordinateTransformation(utm_srs, geo_srs)
    obloc_utm = trans_geo2utm.TransformPoint(oblon, oblat) # Be careful! the order lon/lat changed from gdal v3.
    
    # Distance estimation parameters from von Frisch and Jander (1957) after recalculation by Kohl and Rutschmann (2021)
    # waggle duration [s] <= 1.7130:
    #     distance [km] = np.log(1 - (t - a) / b) / c
    # waggle duration [s] > 1.7130:
    #     distance [km] = (t - d) / e
    a = 0.1096
    b = 1.7208 / 0.6272
    c = -0.6272
    d = 0.7198
    e = 0.70947

    freq = np.linspace(0, fps, N)
    minfreq, maxfreq = 10, 16 # Waggling frequency
    targetfreq = np.where((freq > minfreq) & (freq < maxfreq))
    minfreq_idx = np.amin(targetfreq)
    maxfreq_idx = np.amax(targetfreq)    

    dat = np.zeros((N,rswidth*rsheight),dtype = np.uint8)
    res = np.zeros((rswidth*rsheight),dtype = np.uint8)
    
    # Template for GPU-accelerated FFT
    plan = cupyx.scipy.fftpack.get_fft_plan(cp.asarray(dat),axes=0)

    for rp in range(0,nloop):
        maxLabel = 1024
        connectF = np.zeros((math.ceil(frame_count/N),maxLabel))
        listNLabels = np.zeros(math.ceil(frame_count/N), dtype=np.uint8)
        cent_x = np.zeros((math.ceil(frame_count/N),maxLabel))
        cent_y = np.zeros((math.ceil(frame_count/N),maxLabel))
        
        maxrun = 1024
        beginF = np.zeros(maxrun,dtype = np.uint16)
        endF = np.zeros(maxrun,dtype = np.uint16)
        begin_centx = np.zeros(maxrun)
        begin_centy = np.zeros(maxrun)
        end_centx = np.zeros(maxrun)
        end_centy = np.zeros(maxrun)

        starttime = time.time()
        now = datetime.datetime.now()

        main()
