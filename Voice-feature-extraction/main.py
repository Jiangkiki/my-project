import librosa
import librosa.display as dd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
# 1.加载音频文件
audio_path = '/py project/Voice-feature-extraction/dlm.wav'
info, sr = librosa.load(audio_path,sr=None,duration=6)
print('数据x类型和采样率sr类型', type(info), type(sr))
print('数据x尺寸和采样率', info.shape, sr)
# 设置梅尔滤波器组参数，并设置分帧参数n_fft--帧长，hp_length--帧移
S = librosa.feature.melspectrogram(y=info, sr=sr, n_mels=60,n_fft=1024, hop_length=512,fmax=16000)
mfcc=librosa.feature.mfcc(info, sr, S=librosa.power_to_db(S),n_mfcc=40) # 提取mfcc系数
stft_coff=abs(librosa.stft(info,1024,512,1024)) #分帧然后求短时傅里叶变换，分帧参数与对数能量梅尔滤波器组参数设置要相同
energy = np.sum(np.square(stft_coff),0) #求每一帧的平均能量
MFCC_Energy = np.vstack((mfcc,energy)) # 将每一帧的MFCC与短时能量拼接在一起
print(MFCC_Energy.shape)




# # 2.可视化音频
# plt.figure(figsize=(14,5))
# dd.waveplot(info, sr=sr)
# plt.savefig('波形.jpg')
#
# # 3.声谱图
# X = librosa.stft(info)
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14,5))
# dd.specshow(Xdb,sr=sr,x_axis='time',y_axis='hz')
# plt.colorbar()
# plt.savefig('声谱图.jpg')
#
# # 频谱中心
#
# def spec_center():
#     x=info[:80000] # 取80000/8000=10秒的数据
#     spec_centroids=librosa.feature.spectral_centroid(x, sr=sr)[0]
#     frames=range(len(spec_centroids))
#     t=librosa.frames_to_time(frames, sr=8000) # 时间轴
#
#     # 归一化处理
#     def normalize(x, axis=0):
#         return sklearn.preprocessing.minmax_scale(x, axis=axis)
#     dd.waveplot(x, sr=sr, alpha=0.4, label='wave')
#     plt.figure(figsize=(14, 5))
#     plt.plot(t, normalize(spec_centroids), color='r', linewidth=1, linestyle=':', label='spec_center')
#     plt.legend()
#     plt.show()
#
# spec_center()

