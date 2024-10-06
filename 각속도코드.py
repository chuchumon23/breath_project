import pandas as pd
import numpy as np

# 엑셀 파일 불러오기
file_path = "C:/Users/user/Desktop/sampling_exel data/10.03protocol/RAW DATA/추정연/cjy_skeleton.xlsx"
data = pd.read_excel(file_path)

# 오른쪽, 왼쪽 무릎 각속도 및 x, y, z 가속도 계산
time = data['time']
r_knee_angle = data['R_knee_angle']
l_knee_angle = data['L_knee_angle']

r_knee_x = data['R_knee_x']
l_knee_x = data['L_knee_x']
r_knee_y = data['R_knee_y']
l_knee_y = data['L_knee_y']
r_knee_z = data['R_knee_z']
l_knee_z = data['L_knee_z']

# 각속도 계산 (각도 미분)
r_knee_angle_vel = np.gradient(r_knee_angle, time)
l_knee_angle_vel = np.gradient(l_knee_angle, time)

# x, y, z 가속도 계산 (위치 2번 미분)
r_knee_x_acc = np.gradient(np.gradient(r_knee_x, time), time)
l_knee_x_acc = np.gradient(np.gradient(l_knee_x, time), time)

r_knee_y_acc = np.gradient(np.gradient(r_knee_y, time), time)
l_knee_y_acc = np.gradient(np.gradient(l_knee_y, time), time)

r_knee_z_acc = np.gradient(np.gradient(r_knee_z, time), time)
l_knee_z_acc = np.gradient(np.gradient(l_knee_z, time), time)

# 결과를 새로운 데이터프레임으로 저장
result = pd.DataFrame({
    'time': time,
    'R_knee_angle_vel': r_knee_angle_vel,
    'L_knee_angle_vel': l_knee_angle_vel,
    'R_knee_x_acc': r_knee_x_acc,
    'L_knee_x_acc': l_knee_x_acc,
    'R_knee_y_acc': r_knee_y_acc,
    'L_knee_y_acc': l_knee_y_acc,
    'R_knee_z_acc': r_knee_z_acc,
    'L_knee_z_acc': l_knee_z_acc
})

# 결과를 ep2 엑셀 파일로 저장
output_path = "C:/Users/user/Desktop/sampling_exel data/10.03protocol/RAW DATA/추정연/cjy_skeleton_ACC.xlsx"
result.to_excel(output_path, index=False)

print(f"결과가 '{output_path}'로 저장되었습니다.")
