import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from protocol_data_collection.utils_3d_skeleton import DLT, get_projection_matrix, save_2d_pose, extract_joint_angles
import time
from vispy import app, scene
from vispy.scene import visuals
import os
from openpyxl import Workbook
import threading

# 사용자로부터 라벨 입력받기
foldername = input("Enter the folder name for the Excel file: ")
label = input("Enter the label for the Excel file: ")

# 파일 경로 설정
folder_path = f'C:/Users/user/Documents/Chu_Breath_project/Skeleton_excel_data/{foldername}'
file_path = f'{folder_path}/{label}.xlsx'

# 폴더가 존재하지 않으면 생성
os.makedirs(folder_path, exist_ok=True)

# 워크북 생성
wb = Workbook()
ws = wb.active

# 첫 행에 라벨 입력
headers = ["time", "R_knee_angle", "L_knee_angle", "R_knee_x", "L_knee_x", "R_knee_y", "L_knee_y", "R_knee_z",
           "L_knee_z"]
ws.append(headers)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

frame_shape = [1280, 720]

# 추가적으로 필요한 키포인트 설정
pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28, 29, 31, 30, 32]


# VisPy에서 사용할 CustomAxis 클래스 정의
class CustomAxis:
    def __init__(self, parent):
        self._xaxis = visuals.Line(color='red', width=4, method='gl')
        self._yaxis = visuals.Line(color='green', width=4, method='gl')
        self._zaxis = visuals.Line(color='blue', width=4, method='gl')
        self._update()

        parent.add(self._xaxis)
        parent.add(self._yaxis)
        parent.add(self._zaxis)

    def _update(self):
        x_data = np.array([[0, 0, 0], [10000, 0, 0]], dtype=np.float32)
        y_data = np.array([[0, 0, 0], [0, 10000, 0]], dtype=np.float32)
        z_data = np.array([[0, 0, 0], [0, 0, 10000]], dtype=np.float32)
        self._xaxis.set_data(pos=x_data)
        self._yaxis.set_data(pos=y_data)
        self._zaxis.set_data(pos=z_data)


# 비디오 캡처 초기화 함수
def init_video_capture(input_stream, cap_list, index, frame_shape):
    cap = cv.VideoCapture(input_stream, cv.CAP_DSHOW)  # DirectShow 사용 (Windows 전용)
    if not cap.isOpened():
        print(f"Error: Camera {input_stream} could not be opened.")
        cap_list[index] = None
        return

    # 비디오 캡처 속성 미리 설정
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'XVID'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_shape[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_shape[1])
    cap.set(cv.CAP_PROP_FPS, 60)

    cap_list[index] = cap


# 3D 좌표 변환 및 평행이동 함수
def rotate_and_translate_3d_points(points, rotation_matrix, translation_vector):
    rotated_points = np.dot(points, rotation_matrix.T)
    translated_points = rotated_points + translation_vector
    return translated_points


def change_origin(points, new_origin):
    return points - new_origin


def distance(cor1, cor2):
    return np.linalg.norm(np.array(cor1) - np.array(cor2))


# 스켈레톤 분석을 통해 3D 포인트를 얻고, 데이터를 수집하는 메인 함수
def run_mp(input_stream1, input_stream2, P0, P1):
    start_time = time.perf_counter()
    time.sleep(5)
    torso = [[0, 1], [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legl = [[6, 8], [8, 10], [10, 12], [10, 14], [12, 14]]
    legr = [[7, 9], [9, 11], [11, 13], [11, 15], [13, 15]]

    body = [torso, arml, armr, legr, legl]
    # colors = ['red', 'blue', 'green', 'black', 'orange']
    initial_foot_position = np.zeros(3)
    lines = [
        visuals.Line(color='red', width=5),
        visuals.Line(color='blue', width=5),
        visuals.Line(color='green', width=5),
        visuals.Line(color='black', width=5),
        visuals.Line(color='darkorange', width=5)
    ]

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    for line in lines:
        view.add(line)
    # Set the camera (viewing angle)
    view.camera = scene.cameras.TurntableCamera(up='y',
                                                azimuth=-90,
                                                elevation=0,
                                                fov=60,
                                                distance=1000)
    # Add a custom axis to the view
    axis = CustomAxis(parent=view)

    caps = [None, None]
    threads = [
        threading.Thread(target=init_video_capture, args=(input_stream1, caps, 0, [1280, 720])),
        threading.Thread(target=init_video_capture, args=(input_stream2, caps, 1, [1280, 720]))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    cap0, cap1 = caps
    # cap1, cap0 = caps
    # create body keypoints detector objects.
    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # containers for detected keypoints for each camera. These are filled at each frame.
    # This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []

    # 흰색 이미지 생성 (700x700 크기, 3채널 BGR)
    image = np.ones((1200, 1200, 3), dtype=np.uint8) * 255

    # 글자 속성 정의
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 0, 0)  # 검은색 (BGR)
    thickness = 1


    def update_vispy(ev):
        for bodypart, line in zip(body, lines):
            pos = np.array([frame_p3ds[pt] for pair in bodypart for pt in pair], dtype=np.float32)
            connect = np.array([[i, i + 1] for i in range(0, len(pos), 2)], dtype=np.uint32)
            line.set_data(pos=pos, connect=connect)

    timer = app.Timer()
    timer.connect(update_vispy)
    timer.start(0.02)

    current_time = 0.0
    #start_time = None
    last_recorded_time = 0
    while True:

        current_time = time.perf_counter()

        # read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        frame0 = cv.resize(frame0, (960, 720))
        frame1 = cv.resize(frame1, (960, 720))

        if not ret0 or not ret1: break

        # crop to 720x720.
        # Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[0] != 720:
            frame0 = frame0[:, frame_shape[0] // 2 - frame_shape[1] // 2:frame_shape[0] // 2 + frame_shape[1] // 2]
            frame1 = frame1[:, frame_shape[0] // 2 - frame_shape[1] // 2:frame_shape[0] // 2 + frame_shape[1] // 2]

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)

        # reverse changes
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        # 얼굴과 손 랜드마크 제외
        exclude_indices = list(range(0, 11)) + list(range(17, 23))

        # 2D mediapipe 저장(frame0)
        frame0_keypoints = []
        transformed_2d_frame0 = save_2d_pose(frame0, results0, exclude_indices)
        for i, landmark in transformed_2d_frame0.items():
            pxl_x = int(landmark[0])
            pxl_y = int(landmark[1])
            kpts = [pxl_x, pxl_y]
            frame0_keypoints.append(kpts)

        # 2D mediapipe 저장(frame1)
        frame1_keypoints = []
        transformed_2d_frame1 = save_2d_pose(frame1, results1, exclude_indices)
        for i, landmark in transformed_2d_frame1.items():
            pxl_x = int(landmark[0])
            pxl_y = int(landmark[1])
            kpts = [pxl_x, pxl_y]
            frame1_keypoints.append(kpts)

        # kpts_cam0.append(frame0_keypoints)
        # kpts_cam1.append(frame1_keypoints)

        # calculate 3d position
        frame_p3ds = []  # [[-1,-1,-1],[-1,-1,-1],....,[-1,-1,-1]]
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2)  # calculate 3d position of keypoint
            frame_p3ds.append(_p3d)

        theta = np.radians(19)
        rotation_matrix_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        foot_y = np.array([-740, -450, -127, -30])
        hip_y = np.array([220, 750, 1100, 1500])

        # 기울기 계산
        # foot_slope = np.polyfit(range(len(foot_y)), foot_y, 1)[0]
        # hip_slope = np.polyfit(range(len(hip_y)), hip_y, 1)[0]

        # 기울기 비율 계산
        # slope_ratio = foot_slope/hip_slope

        # 발목 좌표값 조정
        # frame_p3ds[10][1] = frame_p3ds[10][1] * slope_ratio
        # frame_p3ds[11][1] = frame_p3ds[11][1] * slope_ratio

        # 평행이동 벡터
        translation_vector = np.array([0, 0, 0])
        origin = np.array([1000, 0, 0])

        #frame_p3ds = change_origin(frame_p3ds, origin)
        #frame_p3ds = rotate_and_translate_3d_points(frame_p3ds, rotation_matrix_y, translation_vector)


        angles = extract_joint_angles(np.array(frame_p3ds))

        # 이미지 초기화 (흰색)
        image[:] = 255

        # 무릎 각도 및 좌표 값 저장
        angle_right_knee = float(format(angles['right_knee'], ".2f"))
        angle_left_knee = float(format(angles['left_knee'], ".2f"))
        right_knee_x = float(format(frame_p3ds[8][0], ".2f"))
        left_knee_x = float(format(frame_p3ds[9][0], ".2f"))
        right_knee_y = float(format(frame_p3ds[8][1], ".2f"))
        left_knee_y = float(format(frame_p3ds[9][1], ".2f"))
        right_knee_z = float(format(frame_p3ds[8][2], ".2f"))
        left_knee_z = float(format(frame_p3ds[9][2], ".2f"))

        if start_time is not None:
            # 경과 시간(타임스탬프)을 계산하여 기록
            elapsed_time = float(format(current_time - start_time, ".2f"))
            ws.append([elapsed_time, angle_right_knee, angle_left_knee,
                       right_knee_x, left_knee_x, right_knee_y, left_knee_y, right_knee_z, left_knee_z])

        # 이미지에 숫자 그리기

        cv.putText(image, "left_knee_angle: " + str(angle_left_knee), (15, 100), font, font_scale, font_color,
                   thickness, lineType=cv.LINE_AA)
        cv.putText(image, "right_knee_angle: " + str(angle_right_knee), (15, 135), font, font_scale, font_color,
                   thickness, lineType=cv.LINE_AA)

        cv.putText(image, "left_knee_x: " + str(left_knee_x), (15, 170), font, font_scale, font_color, thickness,
                   lineType=cv.LINE_AA)
        cv.putText(image, "right_knee_x: " + str(right_knee_x), (15, 205), font, font_scale, font_color, thickness,
                   lineType=cv.LINE_AA)

        cv.putText(image, "left_knee_y: " + str(left_knee_y), (15, 240), font, font_scale, font_color, thickness,
                   lineType=cv.LINE_AA)
        cv.putText(image, "right_knee_y: " + str(right_knee_y), (15, 275), font, font_scale, font_color, thickness,
                   lineType=cv.LINE_AA)

        cv.putText(image, "left_knee_z: " + str(left_knee_z), (15, 310), font, font_scale, font_color, thickness,
                   lineType=cv.LINE_AA)
        cv.putText(image, "right_knee_z: " + str(right_knee_z), (15, 345), font, font_scale, font_color, thickness,
                   lineType=cv.LINE_AA)

        #frame_p3ds = np.array(frame_p3ds).reshape((16, 3))
        # kpts_3d.append(frame_p3ds)

        # 이미지 표시
        cv.imshow("Angle Parameter", image)
        # cv.imshow("cam0", frame0)
        # cv.imshow("cam1", frame1)

        # ------------------------------------------------------------------------

        # ESC 키 입력 확인 (OpenCV 4.5.4 이상)
        if cv.pollKey() & 0xFF == 27:  # ESC 키를 누르면 종료
            break

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    # 엑셀 파일로 저장
    wb.save(file_path)

    print(f'Excel file created at {file_path}')

    #return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)


if __name__ == '__main__':
    input_stream1 = 0
    input_stream2 = 1

    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    # 카메라 투영 행렬 가져오기
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    # 스켈레톤 데이터 수집 실행
    run_mp(input_stream1, input_stream2, P0, P1)

    # VisPy 실행
    app.run()
