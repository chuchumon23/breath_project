import numpy as np
import pandas as pd

# 파일 읽기
file_path_acc = "C:/Users/user/Desktop/sampling_exel data/10.03protocol/RAW DATA/추정연/cjy_skeleton_ACC.xlsx"
data_acc = pd.read_excel(file_path_acc)

# 새로운 시간 축 생성 (100Hz = 0.01초 간격)
t_new = np.arange(data_acc['time'].min(), data_acc['time'].max(), 0.01)

# 각 열에 대해 100Hz로 선형 보간
interpolated_data = {
    'time': t_new,
    'R_knee_angle_vel': np.interp(t_new, data_acc['time'], data_acc['R_knee_angle_vel']),
    'L_knee_angle_vel': np.interp(t_new, data_acc['time'], data_acc['L_knee_angle_vel']),
    'R_knee_x_acc': np.interp(t_new, data_acc['time'], data_acc['R_knee_x_acc']),
    'L_knee_x_acc': np.interp(t_new, data_acc['time'], data_acc['L_knee_x_acc']),
    'R_knee_y_acc': np.interp(t_new, data_acc['time'], data_acc['R_knee_y_acc']),
    'L_knee_y_acc': np.interp(t_new, data_acc['time'], data_acc['L_knee_y_acc']),
    'R_knee_z_acc': np.interp(t_new, data_acc['time'], data_acc['R_knee_z_acc']),
    'L_knee_z_acc': np.interp(t_new, data_acc['time'], data_acc['L_knee_z_acc']),
}

# 새로운 DataFrame으로 정리
data_acc_interp = pd.DataFrame(interpolated_data)

# 타임스탬프를 소숫점 두 번째 자리까지만 나타냄
data_acc_interp['time'] = data_acc_interp['time'].round(2)

# 결과를 저장
output_path_acc = 'C:/Users/user/Desktop/sampling_exel data/10.03protocol/MAKE DATA/추정연/resampling_skeleton.xlsx'
data_acc_interp.to_excel(output_path_acc, index=False)

output_path_acc
