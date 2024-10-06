# 심박수 샘플링의 경우
import asyncio
from bleak import BleakClient
import pandas as pd
import time  # 정밀 타임스탬프를 위한 모듈 추가
import os  # 추가


program_start_time = time.perf_counter()
# Polar H10의 Bluetooth MAC 주소
address = "F7:D2:3E:6D:22:41"

# Polar H10의 심박수 서비스 UUID
HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# 데이터를 저장할 리스트 초기화
heart_rate_data = []

# 프로그램 실행(시작) 시점의 정밀 타임스탬프 기록


async def run(address):
    try:
        async with BleakClient(address) as client:
            print(f"Connected: {client.is_connected}")

            def handle_heart_rate(sender: int, data: bytearray):
                heart_rate = data[1]
                # 프로그램 시작 시점부터의 경과 시간을 타임스탬프로 기록
                timestamp = time.perf_counter() - program_start_time
                print(f"{timestamp:.6f} - Heart Rate: {heart_rate}")
                heart_rate_data.append([timestamp, heart_rate])

            # 심박수 데이터를 받기 위한 알림 시작
            await client.start_notify(HR_UUID, handle_heart_rate)

            # 1분(60초) 동안 데이터 수신
            await asyncio.sleep(200)
            await client.stop_notify(HR_UUID)

            # 데이터프레임 생성 및 엑셀로 저장
            df = pd.DataFrame(heart_rate_data, columns=["Timestamp (s)", "Heart Rate"])

            # 엑셀 파일을 지정된 경로에 저장
            save_path = r"C:\Users\user\Desktop\샘플링 예시\heart_rate_data_with_timestamps.xlsx"
            df.to_excel(save_path, index=False)
            print(f"Data saved to {save_path}")

            # 현재 작업 디렉터리 출력
            print("Current working directory:", os.getcwd())  # 추가

    except Exception as e:
        print(f"Failed to connect: {e}")


# asyncio를 사용하여 실행
asyncio.run(run(address))
