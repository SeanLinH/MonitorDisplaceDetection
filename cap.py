import cv2
import os
from datetime import datetime
import threading
import time
import argparse
import numpy as np
import pandas as pd
import main

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"

class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
        self.count = 0
        self.capture = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

    def start(self):
        self.isstop = False
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        self.isstop = True
        # self.capture.release()
        #print('ipcam stopped!')
   
    def getframe(self):
        return self.Frame.copy()
        
    def queryframe(self):
        self.count += 1
        if self.count > 5:
            self.stop()
        for i in range(5):     
            self.status, self.Frame = self.capture.read()
        self.capture.release()
    
def shift(img, x, y):
    size = img.shape
    M = np.float32([[1,0,x], [0,1,y]])
    shifted = cv2.warpAffine(img, M, (size[1],size[0]))
    return shifted
    
def zoom(img, zoom_factor=1):
    oh, ow = (img.shape[0], img.shape[1])
    zoomed = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    h, w = ((img.shape[0] - oh) // 2, (img.shape[1] - ow) //2)
    return zoomed[h:oh-h, w:ow-w]
    
    

def fetch(cam_ip,channel):
    
    URL = "rtsp://account:password@" + cam_ip + ":554/cam/realmonitor?channel="+str(channel)
    ipcam = ipcamCapture(URL)
    ipcam.start()
    time.sleep(1)
    
    try:
        img_lst = []
        
        for i in range(10):
            img = ipcam.getframe()
            img = img.astype('int64')
            img_lst.append(img)
        img_lst = np.array(img_lst, dtype='uint8')
        print(cam_ip, channel)
        ipcam.stop()
        return img_lst, 'OK'
    
    except AttributeError as e:
        URL = "rtsp://admin:lc2-admin@" + cam_ip + ":554/chID=" + str(channel) + "&streamtype=main&linkType=tcp"
        ipcam = ipcamCapture(URL)
        ipcam.start()
        time.sleep(1)
        
        try:
            img_lst = []
            for i in range(10):
                img = ipcam.getframe()
                img = img.astype('int64')
                img_lst.append(img)
            img_lst = np.array(img_lst, dtype='uint8')
            print(cam_ip, channel)
            ipcam.stop()
            return img_lst, 'OK'
        
        except AttributeError as e:
            print(f'chanell {channel} 收圖失敗')
            ipcam.stop()
            return None, '監視器未連接'
    
    except:
        print(f'chanell {channel} 收圖失敗')
        ipcam.stop()
        
        return None, '無影像輸出'

def continue_fetch(cam_ip,channel, path):
    URL = "rtsp://admin:lc2-admin@" + cam_ip + ":554/cam/realmonitor?channel="+str(channel)+"&subtype=0"
    ipcam = ipcamCapture(URL)
    ipcam.start()
    time.sleep(1)
    time_check = ''
    foleder_name_2 = ''
    test = input("是否測試異常圖片, 無=0, 位移=1, 放大=2: ")
    lst = pd.read_csv('ip_list.txt', encoding='ANSI')
    depart = lst[lst.IP位址 == cam_ip].課別.values[0]
    past_feature = np.load(f'{path}/datasets/{depart}({cam_ip})/{channel}.npy')
    while True:
        
        img_org, error = fetch(cam_ip, channel)
        ng_count = 0
        max_score = 0
        for img in img_org:   
            if test == '1':
                img = shift(img, 100,100)
            if test == '2':
                img = zoom(img, 1.5)
            feature = main.get_feature(img, num=4)
            result, score = main.move_detect(past_feature,feature, 2)
            max_score = max(score, max_score)
            if result == 'ng':
                ng_count += 1
        if ng_count > 5: 
            judge = 'NG'
        else:
            judge = 'OK'
        cv2.rectangle(img, (50,50), (400,100), [255,255,255], -1)
        cv2.putText(img, f'score:{max_score*100:.2f}%...{judge}', (75,75), cv2.FONT_HERSHEY_SIMPLEX,1, 0, 3)
        # cv2.imwrite(file,img)
        img = cv2.resize(img, (640,360))
        cv2.imshow('rtsp', img)
        cv2.waitKey(1)
    
    
def get_parser():
    parser = argparse.ArgumentParser(description='設定ip')
    parser.add_argument('ip', type=str)
    parser.add_argument('channel',type=int)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    continue_fetch(args.ip, args.channel)
