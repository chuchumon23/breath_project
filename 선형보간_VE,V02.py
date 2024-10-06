

import pandas as pd
import numpy as np

# 파일 읽기
file_path = 'C:/Users/user/Desktop/sampling_exel data/10.03protocol/RAW DATA/cjy/cjy_cosmedK5.xlsx'
data = pd.read_excel(file_path)

# t 값을 초 단위로 변환하는 함수
def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second

# t 열을 초 단위로 변환
data['t_seconds'] = data['t'].apply(time_to_seconds)

# 새로운 시간 축 생성 (100Hz = 0.01초 간격)
t_new = np.arange(data['t_seconds'].min(), data['t_seconds'].max(), 0.01)

# VE와 VO2에 대해 100Hz로 선형 보간
ve_interp = np.interp(t_new, data['t_seconds'], data['VE'])
vo2_interp = np.interp(t_new, data['t_seconds'], data['VO2'])

# 새로운 DataFrame으로 정리
data_interp = pd.DataFrame({'t_seconds': t_new, 'VE': ve_interp, 'VO2': vo2_interp})

# 결과를 저장
output_path = 'C:/Users/user/Desktop/sampling_exel data/10.03protocol/MAKE DATA.xlsx'
data_interp.to_excel(output_path, index=False)
