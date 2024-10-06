# 버튼 누를시 바로 count가 되게끔 해주려고 맨위로 start_time = time.perf_counter()을 올리려고 했는데 오류가 나옴.
import sounddevice as sd
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
import time  # 정밀 타임스탬프를 위해 추가
import os

# Butterworth 밴드패스 필터 생성 함수 (200Hz ~ 800Hz)
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs  # 나이퀴스트 주파수
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# 필터 적용 함수
def apply_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 설정값 정의
duration = 10  # 녹음 시간 (초)
sample_rate = 44100  # 원본 샘플링 레이트 (Hz)
target_sample_rate = 1600  # 다운샘플링할 샘플링 레이트 (Hz)
lowcut = 200.0  # 밴드패스 필터의 하한 주파수 (Hz)
highcut = 800.0  # 밴드패스 필터의 상한 주파수 (Hz)

# 1. 실시간 녹음
print("녹음 시작...")
start_time = time.perf_counter()  # 정밀 타임스탬프 시작 시간 기록
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
sd.wait()  # 녹음 완료 대기
print("녹음 완료!")

# 2. 오디오 데이터를 1D 배열로 변환
audio_data = audio_data.flatten()

# 3. 200Hz ~ 800Hz 필터 적용 (밴드패스 필터)
filtered_audio = apply_bandpass_filter(audio_data, lowcut, highcut, sample_rate, order=3)

# 4. 44100Hz -> 1600Hz로 다운샘플링
downsampled_audio = resample(filtered_audio, int(duration * target_sample_rate))

# 5. 타임스탬프 생성 (0초를 시작으로 정밀한 타임스탬프 적용)
timestamps = [i / target_sample_rate for i in range(len(downsampled_audio))]

# 6. 다운샘플링된 오디오 데이터를 엑셀 파일로 저장
save_path = 'C:/Users/user/Desktop/sampling_exel data/10.03 호흡음/gs_breath.xlsx'
df = pd.DataFrame({
    'Timestamp (s)': timestamps,  # 0초를 기준으로 한 타임스탬프
    'Amplitude': downsampled_audio
})

# 디렉토리 확인 및 생성
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 엑셀 파일로 저장
df.to_excel(save_path, index=False)

print(f"다운샘플링된 오디오 데이터를 엑셀 파일로 저장했습니다: {save_path}")
