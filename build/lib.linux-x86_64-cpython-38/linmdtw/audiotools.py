import numpy as np
import matplotlib.pyplot as plt
import warnings

def load_audio(filename, sr = 44100):
    """
    从文件中加载音频波形。尝试使用 ffmpeg 将其转换为 .wav 文件，
    以便 scipy 的快速 wavfile 加载器可以工作。否则，回退到较慢的 librosa。

    参数
    ----------
    filename: string
        要加载的音频文件路径
    sr: int
        要使用的采样率
    
    返回
    -------
    y: ndarray(N)
        音频样本
    sr: int
        实际使用的采样率
    """
    try:
        # 首先，尝试使用更快的方式加载音频
        from scipy.io import wavfile
        import subprocess
        import os
        FFMPEG_BINARY = "ffmpeg"
        wavfilename = "%s.wav"%filename
        if os.path.exists(wavfilename):
            os.remove(wavfilename)
        subprocess.call([FFMPEG_BINARY, "-i", filename, "-ar", "%i"%sr, "-ac", "1", wavfilename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        _, y = wavfile.read(wavfilename)
        y = y/2.0**15
        os.remove(wavfilename)
        return y, sr
    except:
        # 否则，回退到使用 librosa 加载音频
        warnings.warn("回退到使用 librosa 加载音频，对于长音频文件可能会比较慢")
        import librosa
        return librosa.load(filename, sr=sr)

    
def save_audio(x, sr, outprefix):
    """
    将音频保存到文件

    参数
    ----------
    x: ndarray(N, 2)
        要保存的立体声音频
    sr: int
        要保存的音频的采样率
    outprefix: string
        用作保存音频文件的前缀
    """
    from scipy.io import wavfile
    import subprocess
    import os
    wavfilename = "{}.wav".format(outprefix)
    mp3filename = "{}.mp3".format(outprefix)
    if os.path.exists(wavfilename):
        os.remove(wavfilename)
    if os.path.exists(mp3filename):
        os.remove(mp3filename)
    wavfile.write(wavfilename, sr, x)
    subprocess.call(["ffmpeg", "-i", wavfilename, mp3filename])
    os.remove(wavfilename)

def get_DLNC0(x, sr, hop_length, lag=10, do_plot=False):
    """
    计算衰减的局部自适应归一化 C0 (DLNC0) 特征

    参数
    ----------
    x: ndarray(N)
        音频样本
    sr: int
        采样率
    hop_length: int
        窗口之间的跳跃大小
    lag: int
        包含的滞后数
    
    返回
    -------
    X: ndarray(n_win, 12)
        DLNC0 特征
    """
    from scipy.ndimage.filters import gaussian_filter1d as gf1d
    from scipy.ndimage.filters import maximum_filter1d
    import librosa
    X = np.abs(librosa.cqt(x, sr=sr, hop_length=hop_length, bins_per_octave=12))
    # 半波整流离散导数
    #X = librosa.amplitude_to_db(X, ref=np.max)
    #X[:, 0:-1] = X[:, 1::] - X[:, 0:-1]
    X = gf1d(X, 5, axis=1, order = 1)
    X[X < 0] = 0
    # 保留峰值
    XLeft = X[:, 0:-2]
    XRight = X[:, 2::]
    mask = np.zeros_like(X)
    mask[:, 1:-1] = (X[:, 1:-1] > XLeft)*(X[:, 1:-1] > XRight)
    X[mask < 1] = 0
    # 折叠成八度
    n_octaves = int(X.shape[0]/12)
    X2 = np.zeros((12, X.shape[1]), dtype=X.dtype)
    for i in range(n_octaves):
        X2 += X[i*12:(i+1)*12, :]
    X = X2
    # 计算范数
    if do_plot:
        import librosa.display
        plt.subplot(411)
        librosa.display.specshow(X, sr=sr, x_axis='time', y_axis='chroma')
    norms = np.sqrt(np.sum(X**2, 0))
    if do_plot:
        plt.subplot(412)
        plt.plot(norms)
    norms = maximum_filter1d(norms, size=int(2*sr/hop_length))
    if do_plot:
        import librosa.display
        plt.plot(norms)
        plt.subplot(413)
        X = X/norms[None, :]
        librosa.display.specshow(X, sr=sr, x_axis='time', y_axis='chroma')
    # 应用LNCO
    decays = np.linspace(0, 1, lag+1)[1::]
    decays = np.sqrt(decays[::-1])
    XRet = np.zeros_like(X)
    M = X.shape[1]-lag+1
    for i in range(lag):
        XRet[:, i:i+M] += X[:, 0:M]*decays[i]
    if do_plot:
        plt.subplot(414)
        librosa.display.specshow(XRet, sr=sr, x_axis='time', y_axis='chroma')
        plt.show()
    return XRet

def get_mixed_DLNC0_CENS(x, sr, hop_length, lam=0.1):
    """
    将 DLNC0 与 CENS 连接起来

    参数
    ----------
    x: ndarray(N)
        音频样本
    sr: int
        采样率
    hop_length: int
        窗口之间的跳跃大小
    lam: float
        CENS 特征前的系数
    
    返回
    -------
    X: ndarray(n_win, 24)
        前 12 列是 DLNC0 特征，
        后 12 列是加权的 CENS 特征
    """
    import librosa
    X1 = get_DLNC0(x, sr, hop_length).T
    X2 = lam*librosa.feature.chroma_cens(y=x, sr=sr, hop_length=hop_length).T
    return np.concatenate((X1, X2), axis=1)

def get_mfcc_mod(x, sr, hop_length, n_mfcc=120, drop=20, n_fft=2048):
    """
    计算 mfcc_mod 特征，如 Gadermaier 2019 中所述

    参数
    ----------
    x: ndarray(N)
        音频样本
    sr: int
        采样率
    hop_length: int
        窗口之间的跳跃大小
    n_mfcc: int
        要计算的 MFCC 系数数量
    drop: int
        忽略系数的索引下限
    n_fft: int
        每个窗口使用的 FFT 点数
    
    返回
    -------
    X: ndarray(n_win, n_mfcc-drop)
        MFCC-mod 特征
    """
    import skimage.transform
    import librosa
    X = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length, n_mfcc = n_mfcc, n_fft=n_fft, htk=True)
    X = X[drop::, :].T
    return X

def stretch_audio(x1, x2, sr, path, hop_length, refine = True):
    """
    使用 pyrubberband 将一个音频流变形为另一个音频流，
    根据某个变形路径

    参数
    ----------
    x1: ndarray(M)
        第一个音频流
    x2: ndarray(N)
        第二个音频流
    sr: int
        采样率
    path: ndarray(P, 2)
        变形路径，以窗口为单位
    hop_length: int
        窗口之间的跳跃长度
    refine: boolean
        是否在对齐之前优化变形路径
    
    返回
    -------
    x3: ndarray(N, 2)
        同步后的音频。x2 在右声道，
        x1 变形为 x2 在左声道
    """
    from .alignmenttools import refine_warping_path
    import pyrubberband as pyrb
    print("拉伸中...")
    path_final = path.copy()
    if refine:
        path_final = refine_warping_path(path_final)
    path_final *= hop_length
    path_final = [(row[0], row[1]) for row in path_final if row[0] < x1.size and row[1] < x2.size]
    path_final.append((x1.size, x2.size))
    x3 = np.zeros((x2.size, 2))
    x3[:, 1] = x2
    x1_stretch = pyrb.timemap_stretch(x1, sr, path_final)
    x1_stretch = x1_stretch[0:min(x1_stretch.size, x3.shape[0])]
    x3 = x3[0:min(x3.shape[0], x1_stretch.size), :]
    x3[:, 0] = x1_stretch
    return x3