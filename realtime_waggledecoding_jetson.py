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
    parser.add_argument("--video", type=str, default="video", help="prefix of output video file name")
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


def gst_out(fvideoname):
    return(
        "appsrc ! "
        "nvvidconv ! " 
        "queue ! nvv4l2h264enc bitrate=8000000 ! h264parse ! qtmux ! "
        "filesink location=%s"
    % (
        fvideoname
    ))


#ピクセル単位でフーリエ変換してダンス周波数帯に極大値があるピクセルを抽出
def calc_fft():
    res[:] = 0
    dat_gpu = cp.asarray(dat) # GPUにデータアップ
    F = cupyx.scipy.fftpack.fft(dat_gpu, axis = 0, plan = plan) # テンプレートを使ってフーリエ変換
    F_abs = cp.abs(F) # 複素数 ->絶対値に変換
    F_abs = F_abs / (N/2) # 振幅を元の信号のスケールに揃える
    # FFTデータからピークを自動検出
    F_abs_cpu = cp.asnumpy(F_abs) # CPUへデータダウン
    maximal_idx, maximal_idy = signal.argrelmax(F_abs_cpu, axis=0, order=1) # ピーク（極大値）のインデックス取得
    temp = np.zeros((N,rswidth*rsheight))
    temp[maximal_idx,maximal_idy] = 1 # ピークのある部分にフラグ
    temp = np.where(F_abs_cpu>peak_cut,1,0)*temp # 基準値（peakcut値）より大きい部分のみ抽出
    temp2 = temp[(minfreq_idx-1):(maxfreq_idx+1),:] # ダンスの周波数帯のみ抜き出し
    res[np.sum(temp2,axis=0) > 0] = 255


# ピクセル単位のダンス候補からノイズを取り除いてオブジェクト（クランプ）化
def clump_DancePix(nframe):
    # ダンス候補マップを二元配列に戻してノイズ除去
    img = res.reshape([rsheight,rswidth])
    img = cv2.medianBlur(img,3) #ごま塩ノイズの除去
    # フレーム分割ごとにダンスピクセルをオブジェクト（クランプ）化
    nLabel, labelImage, stats, centroids = cv2.connectedComponentsWithStats(img)
    listNLabels[nframe] = nLabel
    cent_x[nframe,0:nLabel] = centroids[:,0]
    cent_y[nframe,0:nLabel] = centroids[:,1]
    return img, labelImage, stats


# 前後のフレーム分割で連続したダンスがあるか確認し、連続している場合には同一のダンス番号を付与
def check_DanceContinuity(nframe,count,labelimg0,labelimg1):
    for label in range(1, listNLabels[nframe-1]):
        tmp = np.max(labelimg1[labelimg0 == label]) # 一つ前のオブジェクト内にある次のオブジェクト番号（複数あることもあるので一番大きいオブジェクト番号を割り当て）
        if(tmp > 0): 
            # 一つ前のフレーム分割でダンス番号がついていない場合、新たなダンス番号を付与
            if(connectF[nframe-1,label] < 1):
                count += 1
                connectF[nframe,tmp] = count
                connectF[nframe-1,label] = count
            # 一つ前のフレーム分割でダンス番号がついている場合、その番号を継承
            else:
                connectF[nframe,tmp] = connectF[nframe-1,label]
    return count


# ダンス候補ごとに継続時間、始点と終点を計算
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


# 許容距離とフレーム数に応じて一つのダンスが細切れになっている場合を修復
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
    obloc = EarthLocation(lat=oblat*u.deg,lon=oblon*u.deg) # 観測地点の座標をAstropyで使う変数型に格納
    ttemp = datetime.datetime.fromtimestamp(sec,datetime.timezone.utc) # 汎プラットフォームな方法？
    t = Time(ttemp)
    sunloc = get_sun(t).transform_to(AltAz(obstime=t,location=obloc))
    return sunloc.az.degree


# 最終出力
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
        name='国土地理院 標準地図',
        attr="<a href='https://maps.gsi.go.jp/development/ichiran.html' target='_blank'>国土地理院 地理院タイル</a>"
        ).add_to(m)
    folium.TileLayer(
        tiles='https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg',
        name='国土地理院 全国最新写真（シームレス）',
        attr="<a href='https://maps.gsi.go.jp/development/ichiran.html' target='_blank'>国土地理院 地理院タイル</a>"
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


# メイン
def main():
    nframe = 0
    count = 0
    fvideoname = videoprefix + now.strftime('_%Y%m%d_%H%M%S') + '.mp4'
    fcsvname = fprefix + now.strftime('_%Y%m%d_%H%M%S') + '.csv'
    fhtmlname = fprefix + now.strftime('_%Y%m%d_%H%M%S') + ".html"

    video = cv2.VideoCapture(gst_pipeline(), cv2.CAP_GSTREAMER)
    if not video.isOpened():
        sys.exit(1)
    
    out = cv2.VideoWriter(gst_out(fvideoname), cv2.CAP_GSTREAMER, 0, fps, (rswidth, rsheight), isColor=False)

    for skip in trange(0, frame_count, N):
        #フーリエ変換するNフレームを読込
        for i in range(N):
            ret, frame = video.read()
            if not ret:
                frame = np.zeros((rswidth,rsheight),dtype = np.uint8)
            dat[i,:] = frame.reshape(-1)
            out.write(frame)

        # フーリエ変換してダンスピクセルの抽出
        calc_fft()

        # ピクセル単位のダンス候補からノイズを取り除いてオブジェクト（クランプ）化
        img, labelImage, stats = clump_DancePix(nframe)

        if nframe < 1: # 最初のフレーム分割はオブジェクト化したダンス候補マップを退避
            labelimg0 = labelImage
        else: # ２番目以降のフレーム分割から連続してダンス候補が現れるかチェック
            labelimg1 = labelImage
            # 連続する2フレーム分割でダンス候補が継続して確認できる場合、同一のダンスIDを付与
            count = check_DanceContinuity(nframe,count,labelimg0,labelimg1)
            labelimg0 = labelimg1

        # フーリエ変換した分割フレーム数の増加
        nframe += 1

    video.release()
    out.release()

    # ダンス候補ごとに継続時間、始点と終点を計算
    calc_DanceInfo(count)

    # １フレーム分割欠損していて一つのダンスが細切れになっている場合を修復
    connect_DanceGap(count)
    
    # 最終のダンス情報を出力
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
    videoprefix = args.video
    fprefix = args.prefix
    peak_cut = args.peakcut
    
    rswidth = int(width/rsratio)
    rsheight = int(height/rsratio)

    fps = 120/1 # Frame per second
    N = 20 # FFTのフレーム数
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
    # Preparation for geocoordination
    geo_srs = osr.SpatialReference()
    utm_srs = osr.SpatialReference()
    geo_srs.ImportFromEPSG(geoepsg)
    utm_srs.ImportFromEPSG(utmepsg)
    trans_geo2utm = osr.CoordinateTransformation(geo_srs, utm_srs)
    trans_utm2geo = osr.CoordinateTransformation(utm_srs, geo_srs)
    # UTM location of the hive
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

    # ミツバチダンスのターゲット周波数帯
    freq = np.linspace(0, fps, N) # 周波数軸
    minfreq, maxfreq = 10, 16 # ミツバチダンスの周波数帯(Hz)
    targetfreq = np.where((freq > minfreq) & (freq < maxfreq))
    minfreq_idx = np.amin(targetfreq)
    maxfreq_idx = np.amax(targetfreq)    

    dat = np.zeros((N,rswidth*rsheight),dtype = np.uint8) #フーリエ変換に使うフレームを格納する配列
    res = np.zeros((rswidth*rsheight),dtype = np.uint8) #ダンス周波数帯にあるピクセルを格納する配列
    
    # GPUでFFTするためのテンプレート確保（これを使い回すと計算速度が増す）
    plan = cupyx.scipy.fftpack.get_fft_plan(cp.asarray(dat),axes=0)

    for rp in range(0,nloop):
        # ダンス候補の一時格納変数
        maxLabel = 512 #ダンス候補ピクセルをオブジェクト化したときの最大オブジェクト数（本当は可変にできるといいけど）
        connectF = np.zeros((math.ceil(frame_count/N),maxLabel)) # ダンスIDを格納する配列
        listNLabels = np.zeros(math.ceil(frame_count/N), dtype=np.uint8) # オブジェクト数を格納する配列
        cent_x = np.zeros((math.ceil(frame_count/N),maxLabel)) # オブジェクトの重心座標xを格納する配列
        cent_y = np.zeros((math.ceil(frame_count/N),maxLabel)) # オブジェクトの重心座標yを格納する配列
        
        # arrays for final waggle run info
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
