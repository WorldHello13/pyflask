from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# 카메라 캡처 설정
camera = cv2.VideoCapture(0)

# ArUco 딕셔너리 및 파라미터 설정
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# 카메라 매트릭스 및 왜곡 계수 (예시 값)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

def draw_cylinder(image, rvec, tvec, camera_matrix, dist_coeffs, base_length=200, base_radius=5, color=(0, 255, 0)):
    # 카메라와 마커 사이의 거리 계산
    distance = np.linalg.norm(tvec)
    
    # 거리에 비례하여 원기둥의 두께 조절 (스케일 인자 조정)
    scale_factor = 2  # 스케일링 인자 조정
    radius = int(base_radius / distance * scale_factor)
    
    # 원기둥의 시작점과 끝점 3D 좌표 정의
    start_point_3d = np.array([[0, 0, 0]], dtype=np.float32)
    end_point_3d = np.array([[0, base_length, 0]], dtype=np.float32)
    
    # 원기둥의 3D 좌표를 이미지 좌표로 투영
    start_point_2d, _ = cv2.projectPoints(start_point_3d, rvec, tvec, camera_matrix, dist_coeffs)
    end_point_2d, _ = cv2.projectPoints(end_point_3d, rvec, tvec, camera_matrix, dist_coeffs)
    
    # 원기둥 그리기
    start_point_2d = tuple(start_point_2d[0][0].astype(int))
    end_point_2d = tuple(end_point_2d[0][0].astype(int))
    cv2.line(image, start_point_2d, end_point_2d, color, radius)

def generate_frames():
    while True:
        success, frame = camera.read()  # 카메라에서 프레임 읽기
        if not success:
            break
        else:
            # ArUco 마커 탐지
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                # 각 마커의 위치와 자세 추정
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
                
                for i in range(len(ids)):
                    # 마커 경계선 그리기
                    cv2.aruco.drawDetectedMarkers(frame, corners)
                    
                    # 마커의 위치와 자세에 맞춰 원기둥 그리기
                    rvec = rvecs[i][0]
                    tvec = tvecs[i][0]
                    draw_cylinder(frame, rvec, tvec, camera_matrix, dist_coeffs, base_length=300, base_radius=13, color=(0, 255, 0))

            frame = cv2.flip(frame, 1)  # 최종 프레임 좌우 반전

            ret, buffer = cv2.imencode('.jpg', frame)  # 프레임을 JPEG 형식으로 인코딩
            frame = buffer.tobytes()  # 프레임을 바이트로 변환
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 멀티파트 형식으로 프레임 전달

@app.route('/')
def index():
    return render_template('index.html')  # index.html 페이지 렌더링

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
