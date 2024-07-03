import time
import scipy.io as sio
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

def get_cdf(mat, times):
    '''
    函数接收一个误差矩阵（mat）和一组时间阈值（times），
    对于每个阈值，计算误差矩阵中小于或等于该阈值的元素比例，
    这个比例即为累积分布函数（CDF）的值。
    '''
    cdf = np.zeros(len(times))
    for i, time in enumerate(times):
        cdf[i] = np.sum(mat <= time)/mat.size
    return cdf

def plot_err_distributions(short = True):
    '''
    用于绘制音乐片段对齐误差分布
    '''
    # 根据输入参数short决定分析的是短片段还是长片段音乐，然后加载相应的音乐信息文件。
    from linmdtw import get_alignment_row_col_dists
    foldername = "OrchestralPieces/Short"
    if not short:
        foldername = "OrchestralPieces/Long"
    infofile = "{}/info.json".format(foldername)
    pieces = json.load(open(infofile, "r"))

    # 初始化一系列用于存储不同对齐方法误差分布的矩阵。
    N = len(pieces)
    hop_size = 43
    times = np.array([1, 2, 22, 43])
    #tauexp = [3, 4, 5, 6, 7]
    # MRMSDTW 的内存限制指数值
    tauexp = [5, 7]

    distfn = get_alignment_row_col_dists
    XCPUCPU = np.zeros((N, len(times)))
    XGPUCPU = np.zeros((N, len(times)))
    XChromaFastDTW = np.zeros((N, len(times)))
    XChromaMRMSDTW = [np.zeros((N, len(times))) for i in range(len(tauexp))]
    XMFCCFastDTW = np.zeros((N, len(times)))
    XMFCCMRMSDTW = [np.zeros((N, len(times))) for i in range(len(tauexp))]
    XChromaMFCC = np.zeros((N, len(times)))

    for i in range(N):
        if not ( os.path.exists("{}/{}_0.mp3_chroma_path.mat".format(foldername, i)) ):
            continue
        # 在循环中，对于每个音乐片段，脚本加载不同对齐方法产生的路径（如GPU对齐路径、CPU对齐路径等），
        # 并使用distfn（get_alignment_row_col_dists）函数计算这些路径与标准路径之间的距离
        # 对于每种对齐方法，脚本使用get_cdf函数计算在不同时间阈值下，误差小于或等于该阈值的比例，这些比例反映了对齐准确性的分布。get_cdf 函数返回一个与 times 长度相同的数组。
        res = sio.loadmat("{}/{}_0.mp3_chroma_path.mat".format(foldername, i))
        chroma_path_gpu = res['path_gpu']
        if short:
            chroma_path_cpu = res['path_cpu']

        # if short:
        #     chroma_path_cpu64_diag = sio.loadmat("{}/{}_0.mp3_chroma_cpudiag_path.mat".format(foldername, i))['path_cpu']
        #     chroma_path_cpu64_left = sio.loadmat("{}/{}_0.mp3_chroma_path.mat".format(foldername, i))['path_cpu']
        #     d = distfn(chroma_path_cpu64_diag, chroma_path_cpu64_left)
        #     mfcc_path_cpu64_diag = sio.loadmat("{}/{}_0.mp3_mfcc_cpudiag_path.mat".format(foldername, i))['path_cpu']
        #     mfcc_path_cpu64_left = sio.loadmat("{}/{}_0.mp3_mfcc_path.mat".format(foldername, i))['path_cpu']
        #     d = np.concatenate((d, distfn(mfcc_path_cpu64_diag, mfcc_path_cpu64_left)))
        #     XCPUCPU[i, :] = get_cdf(d, times)


        res = sio.loadmat("{}/{}_0.mp3_mfcc_path.mat".format(foldername, i))
        mfcc_path_gpu = res['path_gpu']
        if short:
            mfcc_path_cpu = res['path_cpu']
        
        if short:
            d = np.concatenate((distfn(chroma_path_gpu, chroma_path_cpu), 
                            distfn(mfcc_path_gpu, mfcc_path_cpu)))
            XGPUCPU[i, :] = get_cdf(d, times)

        # 加载近似对齐结果
        res = sio.loadmat("{}/{}_0.mp3_chroma_approx_path.mat".format(foldername, i))
        XChromaFastDTW[i, :] = get_cdf(distfn(chroma_path_gpu, res['path_fastdtw']), times)
        for k, exp in enumerate(tauexp):
            XChromaMRMSDTW[k][i, :] = get_cdf(distfn(chroma_path_gpu, res['path_mrmsdtw%i'%exp]),  times)

        res = sio.loadmat("{}/{}_0.mp3_mfcc_approx_path.mat".format(foldername, i))
        XMFCCFastDTW[i, :] = get_cdf(distfn(mfcc_path_gpu, res['path_fastdtw']), times)
        for k, exp in enumerate(tauexp):
            XMFCCMRMSDTW[k][i, :] = get_cdf(distfn(mfcc_path_gpu, res['path_mrmsdtw%i'%exp]), times)

        XChromaMFCC[i, :] = get_cdf(distfn(chroma_path_gpu, mfcc_path_gpu), times)
    
    for i in range(N-1, -1, -1):
        if not ( os.path.exists("{}/{}_0.mp3_chroma_path.mat".format(foldername, i)) ):
            # 删除对应行
            XCPUCPU = np.delete(XCPUCPU, i, axis=0)
            XGPUCPU = np.delete(XGPUCPU, i, axis=0)
            XChromaFastDTW = np.delete(XChromaFastDTW, i, axis=0)
            for k in range(len(tauexp)):
                XChromaMRMSDTW[k] = np.delete(XChromaMRMSDTW[k], i, axis=0)
            XMFCCFastDTW = np.delete(XMFCCFastDTW, i, axis=0)
            for k in range(len(tauexp)):
                XMFCCMRMSDTW[k] = np.delete(XMFCCMRMSDTW[k], i, axis=0)
            XChromaMFCC = np.delete(XChromaMFCC, i, axis=0)

    times = times/hop_size
    # names 和 results 列表包含了不同的 DTW（动态时间弯曲）类型和对应的结果数据。
    # results 列表中的每个元素是一个列表，包含了在不同误差阈值（times 数组定义）下的比例数据。
    names = ["DLNC0\nFastDTW"] + ["DLNC0\nMRMSDTW\n$10^%i$"%exp for exp in tauexp] + ["mfcc-mod\nFastDTW"] + ["mfcc-mod\nMRMSDTW\n$10^%i$"%exp for exp in tauexp] + ["DLNC0\nvs\nmfcc-mod"]
    results = [XChromaFastDTW] + XChromaMRMSDTW + [XMFCCFastDTW] + XMFCCMRMSDTW + [XChromaMFCC]
    if short:
        # names = ["CPU vs CPU\n64-bit", "GPU vs CPU\n32-bit"] + names
        names = ["GPU vs CPU\n32-bit"] + names
        # results = [XCPUCPU, XGPUCPU] + results
        results = [XGPUCPU] + results
    p = 1
    approxtype = []
    cdfthresh = []
    cdf = []
    # 通过遍历 results 列表，对于每个 DTW 类型和对应的误差阈值，计算落在该误差阈值内的比例。
    for name, X in zip(names, results):
        print(name)
        print(X)
        for k, t in enumerate(times):
            approxtype += [name]*X.shape[0]
            cdfthresh += ["<= %.2g"%(times[k])]*X.shape[0]
            # X列表中所有行的第k列的值，形成一个一维列表
            cdf += (X[:, k]**p).tolist()
            print([name])
            print((X[:, k]**p).tolist())
    plt.figure(figsize=(6, 4))
    # # 设置 matplotlib 使用支持中文的字体
    # matplotlib.rcParams['font.family'] = 'SimHei'
    # matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    palette = sns.color_palette("cubehelix", len(times))
    # 横轴：DTW 类型；纵轴：误差容限内的比例；颜色：误差（秒）
    df = pd.DataFrame({"DTW 类型":approxtype, "误差（秒）":cdfthresh, "误差容限内的比例":cdf})
    ax = plt.gca()
    g = sns.swarmplot(x="DTW 类型", y="误差容限内的比例", hue="误差（秒）", data=df, palette=palette)
    # 一维数组ticks。这个数组包含从0到1（包括两端点）的11个等间隔的数值。
    ticks = np.linspace(0, 1, 11)
    ax.set_yticks(ticks)
    # 设置纵轴的刻度标签。这里，每个刻度标签是 ticks 数组中的每个元素。
    ax.set_yticklabels(["%.2g"%(t**(1.0/p)) for t in ticks])
    ax.set_xticklabels(g.get_xticklabels(), rotation=90)
    if short:
        plt.title("较短片段的对齐误差")
        plt.savefig("Shorter.svg", bbox_inches='tight')
    else:
        plt.title("较长片段的对齐误差")
        plt.savefig("Longer.svg", bbox_inches='tight')
    

def draw_systolic_array():
    plt.figure(figsize=(5, 5))
    AW = 0.2
    N = 6
    ax = plt.gca()
    for i in range(N):
        for j in range(N):
            if i > 0:
                ax.arrow(i-0.1, j, -0.6, 0, head_width = AW, head_length = AW, fc = 'k', ec = 'k')
            if j > 0:
                ax.arrow(i, j-0.15, 0, -0.52, head_width = AW, head_length = AW, fc = 'k', ec = 'k')
            if i > 0 and j > 0:
                ax.arrow(i-0.08, j-0.08, -0.67, -0.67, head_width = AW, head_length = AW, fc = 'k', ec = 'k')
    for i in range(N):
        for j in range(N):
            plt.scatter(i, j, 200, c='C0', facecolors='none', zorder=10)

    c = plt.get_cmap('afmhot')
    C = c(np.array(np.round(np.linspace(0, 255, 2*N+1)), dtype=np.int32))
    C = C[:, 0:3]
    for i in range(1, N+1):
        x = np.array([i-0.5, -0.5])
        y = np.array([-0.5, i-0.5])
        plt.plot(x, y, c=C[i-1, :], linewidth=3)
        print(i-1)
        plt.plot(N-x-1, N-y-1, c=C[2*N-i, :], linewidth=3)
        print(2*N-i)
    
    #plt.axis('off')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    f = 0.8
    ax.set_facecolor((f, f, f))
    plt.savefig("LinearSystolic.svg", bbox_inches = 'tight')

def get_length_distributions():
    fac = 0.8
    plt.figure(figsize=(fac*12, fac*3))
    # for k, s in enumerate(["Short", "Long"]):
    for k, s in enumerate(["Short"]):
        plt.subplot(1, 2, k+1)
        foldername = "OrchestralPieces/{}".format(s)
        infofile = "{}/info.json".format(foldername)
        pieces = json.load(open(infofile, "r"))
        N = len(pieces)
        hop_length = 43
        lengths = []
        for i in range(N):
            if not ( os.path.exists("{}/{}_0.mp3_chroma_path.mat".format(foldername, i)) ):
                continue
            res = json.load(open("{}/{}_0.mp3_chroma_stats.json".format(foldername, i), "r"))
            M = res['M']
            N = res['N']
            lengths.append(M/(hop_length*60))
            lengths.append(N/(hop_length*60))
        sns.distplot(np.array(lengths), kde=False, bins=20, rug=True)
        plt.xlabel("持续时间（分钟）")
        plt.ylabel("计数")
        plt.title("{} 收集".format(s))
    plt.savefig("Counts.svg", bbox_inches='tight')



def get_cell_usage_distributions():
    ratios = []
    for s in ["Short", "Long"]:
        foldername = "OrchestralPieces/{}".format(s)
        infofile = "{}/info.json".format(foldername)
        pieces = json.load(open(infofile, "r"))
        N = len(pieces)
        for f in ["chroma", "mfcc"]:
            for i in range(N):
                if not ( os.path.exists("{}/{}_0.mp3_chroma_path.mat".format(foldername, i)) ):
                    continue
                res = json.load(open("{}/{}_0.mp3_{}_stats.json".format(foldername, i, f), "r"))
                denom = res['M']*res['N']
                total = res['totalCells']
                ratios.append(total/denom)
    plt.figure(figsize=(5, 3))
    sns.distplot(ratios, kde=False)
    plt.title("处理的单元与总单元的比率")
    plt.xlabel("比率")
    plt.ylabel("计数")
    plt.savefig("Cell.svg", bbox_inches='tight')


def get_memory_table():
    fac = 0.8
    delta = 30
    plt.figure(figsize=(fac*12, fac*3))
    for k, s in enumerate(["Short", "Long"]):
        plt.subplot(1, 2, k+1)
        foldername = "OrchestralPieces/{}".format(s)
        infofile = "{}/info.json".format(foldername)
        pieces = json.load(open(infofile, "r"))
        N = len(pieces)
        hop_length = 43
        lengths = []
        for i in range(N):
            if not ( os.path.exists("{}/{}_0.mp3_chroma_path.mat".format(foldername, i)) ):
                continue
            res = json.load(open("{}/{}_0.mp3_chroma_stats.json".format(foldername, i), "r"))
            M = res['M']
            N = res['N']
            print(M/hop_length, pieces[i][0]['info'])
            print(N/hop_length, pieces[i][1]['info'])

            dtw = M*N*4/(1024**2)
            if dtw < 1024:
                print("DTW: ", dtw, "MB")
            else:
                print("DTW: ", dtw/1024, "GB")
            print("Ours: ", min(M, N)*4*3/(1024**2), "MB")
            print("FastDTW: ", 4*min(M, N)*(4*delta+5)/(1024**2), "MB" )



# 设置 matplotlib 使用支持中文的字体
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# plot_err_distributions(short=True)
# plot_err_distributions(short=False)
draw_systolic_array()
# get_length_distributions()
# get_cell_usage_distributions()
# get_memory_table()