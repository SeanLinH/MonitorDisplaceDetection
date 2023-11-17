import cv2
import os
from datetime import datetime, timedelta
import cap
import pandas as pd
import numpy as np
import time
import argparse
import threading

def check_folder(folder_name):
    if not os.path.exists(folder_name):
            os.makedirs(folder_name)

def mask(img):
    time_mask = [0.041666666666666664, 0.65625, 0.09375, 0.9791666666666666]
    channel_mask = [0.8541666666666666, 0.020833333333333332, 0.9583333333333334, 0.15625]
    y1 = int(img.shape[0] * time_mask[2])
    y2 = int(img.shape[0] * channel_mask[0])
    img = img[y1:y2, :]
    return img


def get_feature(img, num=3, t=40):
    '''定義圖像特徵
    img: 影像資料 ndarray
    num: 圖像判定矩陣 num=3 --> 3x3 9宮格
    t: 矩陣判斷點的大小，t=40 --> 40x40 的點位大小
    '''
    feature = []
    h = (img.shape[0] - num * t) // (num-1)
    w = (img.shape[1] - num * t) // (num-1)
    for i in range(num):
        y = (h + t)*i
        for j in range(num):
            x = (w + t) * j
            gray = cv2.cvtColor(img[y:y+t, x:x+t], cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray, 30, 150)
            canny = canny.reshape(canny[...,np.newaxis].shape)
            canny = np.concatenate((canny,canny,canny),axis=2)
            dot  = img[y:y+t, x:x+t].astype('float32')           
            dot1 = dot[:,:,0] / (dot[:,:,1] + 1e-3)
            dot2 = dot[:,:,1] / (dot[:,:,2] + 1e-3)
            dot3 = dot[:,:,2] / (dot[:,:,0] + 1e-3)
            dot1 = dot1.reshape(dot1[..., np.newaxis].shape)
            dot2 = dot2.reshape(dot2[..., np.newaxis].shape)
            dot3 = dot3.reshape(dot3[..., np.newaxis].shape)
            rgb_dot = np.concatenate((dot1,dot2,dot3), axis=2)
            
            feature.append([dot, rgb_dot, canny])
    feature = np.array(feature, dtype='float32')
    return feature

def move_detect(past, now, allow=3):
    '''
    設定允收值allow門檻
    allow 越大越敏感，容易跳NG
    當allow 超過特徵矩陣必會跳NG
    建議從0開始試錯
    '''
    
    a = abs(now - past)
    total = a.shape[0]
    rgb = [False, False, False] # 判斷像素RGB [差值, 顏色, 形狀]
    score = 0
    for i in range(a.shape[0]):
        for j in range(3):
            ## 判斷差值
            dot = a[i,0,:,:,j]
            cnt = dot[dot > 5].size
            if cnt > int(a.shape[2]**2 * 0.8):
                rgb[0] = True

            ## 判斷顏色變化
            rgb_dot = a[i,1,:,:,j]
            cnt = rgb_dot[rgb_dot > 0.005].size
            if cnt > int(a.shape[2]**2 * 0.4):
                rgb[1] = True
        ## 判斷形狀
        canny_score = (a[i,2,:,:,0].sum() + 1e-3) / (past[i,2,:,:,0].sum() + 1e-3)
        if canny_score > 0.9:
            rgb[2] = True
        if rgb == [True, True, True]:
            score += 1
        rgb = [False, False, False]
    
    if (score+allow) >= total:
        return 'ng', 1.
    score = score / total
    return 'ok', score

def savebase(ip, channel, path):
    '''
    若要更新基準照片
    請在CMD 字元命令下使用參數 --save
    將會自動更新
    '''
    lst = pd.read_csv('ip_list.txt')
    depart = lst[lst.IP位址 == ip].課別.values[0]
    path = os.path.join(path, 'datasets')
    file = f'{path}/{depart}({ip})/{channel}.jpg'
    npy_file = f'{path}/{depart}({ip})/{channel}.npy'
    img_org, error = cap.fetch(ip, str(channel)) # 抓圖片出來
    feature = get_feature(img_org[-1], num=4)
    np.save(npy_file, feature)
    cv2.imencode('.jpg', img_org[-1])[1].tofile(f'{file}') #儲存原圖
    print(f'儲存基準圖片: {file}')
    return 1

def to_record(now, ip, channel, text, path):
    with open(os.path.join(path, 'record.csv'), 'a') as f:
        f.write(f'{now},{ip},{channel},{text}\n')
        f.close
    
def run(ip_list, path):
    ### 新增未建立的檔案
    if os.path.exists(ip_list) == False:
        with open(ip_list, 'w') as f:
            f.write('課別,IP位址,通道數')
            f.close
    result_dir = os.path.join(path,'results')
    datasets_dir = os.path.join(path, 'datasets')
    print(result_dir)
    check_folder(result_dir)
    check_folder(datasets_dir)
    if not os.path.exists(f'{result_dir}/record.csv'):
        with open(f'{result_dir}/record.csv', 'w') as re:
            re.write(f'time,ip,channel,judge\n')
            re.close
    white = pd.read_csv('white_list.txt')
        
    while True:
        lst = pd.read_csv(ip_list)
        
        allday = datetime.now().strftime('%Y%m%d')
        if len(lst) != 0:
            try:
                for ip in lst.IP位址:
                    depart = lst[lst.IP位址 == ip].課別.values[0] # 抓當下IP 的課別
                    check_folder(f'{datasets_dir}/{depart}({ip})') # 確定有無資料夾

                    for channel in range(1, lst[lst.IP位址 == ip].通道數.values[0]+1):
                        img_org, error = cap.fetch(ip, str(channel)) # 抓圖片出來

                        # img = img_lst[-1]
                        file = f'{datasets_dir}/{depart}({ip})/{channel}.jpg'
                        # time.sleep(5)
                        now = datetime.now().strftime('%Y%m%d')
                        npy_file = f'{datasets_dir}/{depart}({ip})/{channel}.npy'

                        ### 判斷是否為影像資料
                        if (type(img_org) != np.ndarray) and (error != 'OK') and (os.path.exists(npy_file) == True):
                            print(f'{ip} 通道{channel} 未連接')
                            to_record(now, ip, channel, error, result_dir)

                        elif type(img_org) == np.ndarray and error == 'OK':
                            # img = cap.shift(img, 50,50) # 測試用
                            im = mask(img_org[-1])
                            ### 判定是否有黑屏
                            if (im.max() < 20) and (os.path.exists(npy_file) == True):
                                print(f'IP:{ip} -- channel {channel} 黑屏')
                                cv2.imencode('.jpg', img_org[-1])[1].tofile(f'{datasets_dir}/{depart}({ip})/moved/{now}-{channel}.jpg') #儲存原圖
                                to_record(now, ip, channel, '黑屏', result_dir)
                            elif (im.max() < 20) and (os.path.exists(npy_file) == False):
                                print(f'IP:{ip} -- channel {channel} 黑屏尚未開通')

                            ### 若沒有基準圖片會自動儲存
                            elif (im.max() > 20) and (os.path.exists(npy_file) == False): 
                                feature = get_feature(img_org[-1], num=4)
                                np.save(npy_file, feature)
                                cv2.imencode('.jpg', img_org[-1])[1].tofile(f'{file}') #儲存原圖
                                print(f'儲存基準圖片: {file}')

                            ### 圖像邊緣偵測 判斷 ### 
                            elif os.path.exists(npy_file) == True:
                                past_feature = np.load(npy_file)

                                '''拿10張影像來比較，若超過5張認定NG才算是NG'''
                                ng_count = 0
                                for img in img_org:
                                    feature = get_feature(img, num=4)
                                    result, score = move_detect(past_feature,feature, 2)
                                    if result == 'ng':
                                        ng_count += 1

                                if ng_count > 5: #連續十張超過一半判斷NG即確定被移動過
                                    print(f'IP:{ip} -- channel {channel} 被移動過')
                                    check_folder(f'{datasets_dir}/{depart}({ip})/moved/')
                                    cv2.imencode('.jpg', img_org[-1])[1].tofile(f'{datasets_dir}/{depart}({ip})/moved/{now}-{channel}.jpg') #儲存原圖
                                    

                                    # 將NG結果存到record 
                                    to_record(now, ip, channel,'NG', result_dir)
                                    if channel in white[(white.IP位址 == ip)].通道.values:
                                        to_record(now, ip, channel, 'OK', result_dir)                                  
                                    print(f'之前的圖片存放於: {datasets_dir}/{depart}({ip})/moved/{now}-{channel}.jpg')
                                else:
                                    # 將OK結果存到record 
                                    to_record(now, ip, channel, 'OK', result_dir)
                                    print(f'{depart} {ip} -- {channel}...OK')                            
            except:
                time.sleep(10)
                pass
        else:
            print('請在ip_lis.txt 中新增監視器IP通道')
            break
            
        # 跨夜計算結果
        now = datetime.strptime(now,'%Y%m%d' )
        now = (now - timedelta(days=1)).strftime('%Y%m%d')
        if now == allday:
            record = pd.read_csv(f'{result_dir}/record.csv')
            for ip in lst.IP位址:
                ip_df = record[record['ip'] == ip]
                ch_list = ip_df['channel'].unique()
                depart = lst[lst.IP位址 == ip].課別.values[0]
                for ch in ch_list:
                    judge = ip_df[ip_df['channel'] == ch]['judge'].value_counts()
                    if len(judge) == 0:
                        pass
                    elif ('監視器未連接' in judge) == True:
                        img_org, error = cap.fetch(ip, str(ch))
                        if error == 'OK':
                            state = '目前已恢復'
                        else:
                            state = ''
                        with open(f'{result_dir}/{depart}({ip}).txt', 'a') as f:
                            f.write(f'{now},{depart}({ip}),{ch},監視器未連接{state}\n')
                            f.close
                            
                    elif ('OK' in judge) == True:
                        rm = f'{datasets_dir}/{depart}({ip})/moved/'
                        check_folder(rm)
                        rm_list = os.listdir(rm)
                        for i in rm_list:
                            if (i.find(allday) == 0) and (float(i[9:11]) == float(ch)):
                                try:
                                    rm_file = os.path.join(rm, i)
                                    os.remove(rm_file)
                                except:
                                    pass
                    elif ('黑屏' in judge) == True:
                        with open(f'{result_dir}/{depart}({ip}).txt', 'a') as f:
                            f.write(f'{now},{depart}({ip}),{ch},黑屏\n')
                            f.close
                    elif ('無影像輸出' in judge) == True:
                        with open(f'{result_dir}/{depart}({ip}).txt', 'a') as f:
                            f.write(f'{now},{depart}({ip}),{ch},無影像輸出\n')
                            f.close
                    else:
                        with open(f'{result_dir}/{depart}({ip}).txt', 'a') as f:
                            f.write(f'{now},{depart}({ip}),{ch},位置更動\n')
                            f.close
            with open(f'{result_dir}/record.csv', 'w') as re:
                re.write(f'time,ip,channel,judge\n')
                re.close

            
def realtime(ip, channel, path):
    if ip == None:
        return 0
    cap.continue_fetch(ip, channel, path)
        

def get_parser():
    print("上線運作：python main.py\n"
          "測試即時畫面：python main.py --ip [ip] --ch [channel]\n"
          "更新基準畫面: python main.py --ip [ip] --ch [channel] --save\n")
    parser = argparse.ArgumentParser(description="", usage='歡迎使用監視器位移偵測，使用帶參數來了解快速測試')
    parser.add_argument('--ip', type=str, default=None, help='input CCTV ip address' )
    parser.add_argument('--ch',type=int, default=None, help='input CCTV monitor channel number')
    parser.add_argument('--save', action='store_true', help='renew CCTV benchmark image  ')
    parser.add_argument('--path', type=str, default='',help='監視器畫面的儲存位置')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.save:
        a = savebase(args.ip, args.ch, args.path)
    else:
        a = realtime(args.ip, args.ch, args.path)
        
    if a == 0:
        # threading.Thread(target=run, daemon=True, args=()).start()
        run('ip_list.txt', args.path)