

import numpy as np
import pandas as pd

# 파일 읽기
file_path_hr ="C:/Users/user/Desktop/sampling_exel data/10.03protocol/RAW DATA/추정연/cjy_heartrate.xlsx"
data_hr = pd.read_excel(file_path_hr)

# 새로운 시간 축 생성 (100Hz = 0.01초 간격)
t_new = np.arange(data_hr['Timestamp (s)'].min(), data_hr['Timestamp (s)'].max(), 0.01)

# Heart Rate에 대해 100Hz로 선형 보간
hr_interp = np.interp(t_new, data_hr['Timestamp (s)'], data_hr['Heart Rate'])

# 새로운 DataFrame으로 정리
data_hr_interp = pd.DataFrame({'Timestamp (s)': t_new, 'Heart Rate': hr_interp})

# 타임스탬프를 소숫점 두 번째 자리까지만 나타냄
data_hr_interp['Timestamp (s)'] = data_hr_interp['Timestamp (s)'].round(2)

# 결과를 저장
output_path_hr_rounded = 'C:/Users/user/Desktop/sampling_exel data/10.03protocol/MAKE DATA/추정연/resampling_100hz_HR.xlsx'
data_hr_interp.to_excel(output_path_hr_rounded, index=False)
