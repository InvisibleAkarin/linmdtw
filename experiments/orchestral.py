import linmdtw
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import warnings
import json
import subprocess
import os
import glob
import traceback
# import youtube_dl
from scipy.spatial.distance import euclidean

def my_hook(d):
    print(d['status'])

def download_corpus(foldername):
    """
    从 YouTube 下载音频语料库
    参数
    ----------
    foldername: string
        下载音频的文件夹路径。它必须包含一个 "info.json" 文件，
        其中包含所有片段的 URL、开始时间和结束时间
    """
    infofile = "{}/info.json".format(foldername)
    pieces = json.load(open(infofile, "r"))
    for i, pair in enumerate(pieces):
        for j, piece in enumerate(pair):
            path = "{}/{}_{}.mp3".format(foldername, i, j)
            if os.path.exists(path):
                print("已经下载 ", path)
                continue
            url = piece['url']

            if os.path.exists('temp.mp3'):
                os.remove('temp.mp3')
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'progress_hooks': [my_hook],
                'outtmpl':'temp.mp3'
            }
            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except:
                warnings.warn("下载出错 {}".format(path))
                continue
            
            command = ["ffmpeg", "-i", "temp.mp3"]
            start = 0
            converting = False
            if 'start' in piece:
                converting = True
                start = piece['start']
                command.append("-ss")
                command.append("{}".format(start))
            if 'end' in piece:
                converting = True
                time = piece['end'] - start
                command.append("-t")
                command.append("{}".format(time))
            if converting:
                command.append(path)
                subprocess.call(command)
            else:
                subprocess.call(["mv", "temp.mp3", path])

def align_pieces(filename1, filename2, sr, hop_length, do_mfcc, compare_cpu, do_stretch=False, delta=30, do_stretch_approx=False):
    """
    使用各种技术对齐两段音频，并将结果保存到 .mat 文件中
    参数
    ----------
    filename1: string
        第一个音频文件的路径
    filename2: string
        第二个音频文件的路径
    sr: int
        两个音频文件使用的采样率
    hop_length: int
        特征窗口之间的跳跃长度
    do_mfcc: boolean
        如果为真，使用 mfcc_mod 特征。否则，使用 DLNC0 特征
    compare_cpu: boolean
        如果为真，与暴力 CPU DTW 进行比较
    do_stretch: boolean
        如果为真，根据 GPU 扭曲路径拉伸音频，并保存到文件
    delta: int
        用于 fastdtw 的半径
    do_stretch_approx: boolean
        如果为真，根据 fastdtw 和 mrmsdtw 的近似扭曲路径拉伸音频
    """
    if not os.path.exists(filename1):
        warnings.warn("跳过 "+ filename1)
        return
    if not os.path.exists(filename2):
        warnings.warn("跳过 "+ filename2)
        return
    prefix = "mfcc"
    if not do_mfcc:
        prefix = "chroma"
    pathfilename = "{}_{}_path.mat".format(filename1, prefix)
    approx_pathfilename = "{}_{}_approx_path.mat".format(filename1, prefix)
    if os.path.exists(pathfilename) and os.path.exists(approx_pathfilename):
        print("已经计算了所有对齐 ", filename1, filename2)
        return
    
    x1, sr = linmdtw.load_audio(filename1, sr)
    x2, sr = linmdtw.load_audio(filename2, sr)
    if do_mfcc:
        X1 = linmdtw.get_mfcc_mod(x1, sr, hop_length)
        X2 = linmdtw.get_mfcc_mod(x2, sr, hop_length)
    else:
        X1 = linmdtw.get_mixed_DLNC0_CENS(x1, sr, hop_length)
        X2 = linmdtw.get_mixed_DLNC0_CENS(x2, sr, hop_length)

    X1 = np.ascontiguousarray(X1, dtype=np.float32)
    X2 = np.ascontiguousarray(X2, dtype=np.float32)

    if os.path.exists(pathfilename):
        print("已经计算了所有完整的", prefix, "对齐", filename1, filename2)
    else:
        tic = time.time()
        metadata = {'totalCells':0, 'M':X1.shape[0], 'N':X2.shape[0], 'timeStart':tic}
        print("开始 GPU 对齐...")
        path_gpu = linmdtw.linmdtw(X1, X2, do_gpu=True, metadata=metadata)
        metadata['time_gpu'] = time.time() - metadata['timeStart']
        print("GPU 时间", metadata['time_gpu'])
        # print(path_gpu)
        path_gpu = np.array(path_gpu[1])
        paths = {"path_gpu":path_gpu}
        if compare_cpu:
            tic = time.time()
            print("进行 CPU 对齐...")
            path_cpu = linmdtw.dtw_brute_backtrace(X1, X2)
            elapsed = time.time() - tic
            print("CPU 时间", elapsed)
            metadata["time_cpu"] = elapsed
            path_cpu = np.array(path_cpu[1])
            paths["path_cpu"] = path_cpu

        for f in ['totalCells', 'M', 'N']:
            metadata[f] = int(metadata[f])
        for f in ['XGPU', 'YGPU', 'timeStart']:
            if f in metadata:
                del metadata[f]
        json.dump(metadata, open("{}_{}_stats.json".format(filename1, prefix), "w"))
        path_gpu_arr = path_gpu.copy()
        sio.savemat(pathfilename, paths)

        if do_stretch:
            print("拉伸中...")
            x = linmdtw.stretch_audio(x1, x2, sr, path_gpu_arr, hop_length)
            linmdtw.audiotools.save_audio(x, sr, "{}_{}_sync".format(filename1, prefix))
            print("拉伸完成")

    # 进行近似对齐
    if os.path.exists(approx_pathfilename):
        print("已经计算了所有近似", prefix, "对齐", filename1, filename2)
    else:
        print("进行 fastdtw...")
        tic = time.time()
        path_fastdtw = linmdtw.fastdtw(X1, X2, radius = delta)
        elapsed = time.time()-tic
        print("fastdtw 耗时", elapsed)
        path_fastdtw = np.array([[p[0], p[1]] for p in path_fastdtw[1]])
        res = {"path_fastdtw":path_fastdtw, "elapsed_fastdtw":elapsed}
        if do_stretch_approx:
            x = linmdtw.stretch_audio(x1, x2, sr, path_fastdtw, hop_length)
            linmdtw.audiotools.save_audio(x, sr, "{}_{}_fastdtw_sync".format(filename1, prefix))
        # 现在使用不同的内存限制进行 mrmsdtw
        for tauexp in [3, 4, 5, 6, 7]:
            print("进行 mrmsdtw 10^%i"%tauexp)
            tic = time.time()
            path = linmdtw.mrmsdtw(X1, X2, 10**tauexp)
            elapsed = time.time()-tic
            print("mrmsdtw 10^%i 耗时: %.3g"%(tauexp, elapsed))
            res['path_mrmsdtw%i'%tauexp] = path[1]
            res['elapsed_mrmsdtw%i'%tauexp] = elapsed
        # print("res ",res)
        sio.savemat(approx_pathfilename, res)



def align_corpus(foldername, compare_cpu, do_stretch):
    """
    对特定语料库进行所有对齐操作
    参数
    ----------
    foldername: string
        下载音频的文件夹路径。它必须包含一个 "info.json" 文件，
        其中包含所有片段的 URL、开始时间和结束时间
    compare_cpu: boolean
        如果为真，与暴力 CPU DTW 进行比较
    do_stretch: boolean
        如果为真，根据 GPU 扭曲路径拉伸音频，并保存到文件
    """
    hop_length = 512
    sr = 22050
    infofile = "{}/info.json".format(foldername)
    pieces = json.load(open(infofile, "r"))
    for do_mfcc in [False, True]:
        for i, pair in enumerate(pieces):
            # try:
                filename1 = "{}/{}_0.mp3".format(foldername, i)
                filename2 = "{}/{}_1.mp3".format(foldername, i)
                print("正在对齐 ", filename1, filename2)
                align_pieces(filename1, filename2, sr, hop_length, do_mfcc=do_mfcc, compare_cpu=compare_cpu, do_stretch=do_stretch)
            # except Exception as e:
                # print("发生错误：", e)
                # traceback.print_exc()  # 打印错误的 traceback

if __name__ == '__main__':
    # download_corpus("OrchestralPieces/Short")
    align_corpus("OrchestralPieces/Short", compare_cpu=True, do_stretch=False)
    # download_corpus("OrchestralPieces/Long")
    # align_corpus("OrchestralPieces/Long", compare_cpu=False, do_stretch=False)