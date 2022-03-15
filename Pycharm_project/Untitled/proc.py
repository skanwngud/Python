import json
import cv2
import math
import numpy as np
import torch

# cuda device 사용 여부
print(f"{torch.cuda.is_available()}")

model = torch.hub.load(
    repo_or_dir="D:/pycharmProject/OT-OD/yolov5/",
    model="custom", source="local",
    path="D:/pycharmProject/OT-OD/yolov5/02_07_1920/02_07_1920/weights/best.pt"
)

model.conf = 0.7
model.iou = 0.3


# 좌표, confidence 값
def make_coords(idx, data):
    x1, y1, x2, y2, conf = \
        int(data[idx][0]), int(data[idx][1]), int(data[idx][2]), int(data[idx][3]), int(data[idx][4])
    return x1, y1, x2, y2, conf


# 중심좌표
def make_center(x1, y1, x2, y2):
    w = np.abs(x1-x2)
    h = np.abs(y1-y2)

    center_x = int(x1+w/2)
    center_y = int(y1+h/2)
    return center_x, center_y


# VideoWriter 인스턴스
def make_video_writer(vidcap):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = round(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    return fourcc, w, h , fps


# 두 점 사이의 각도 구하기
def make_angle(dx, dy):
    if dx == 0:
        dx = 1

    angle = round(math.atan(dy/dx) * (180.0 / math.pi))

    if dx < 0:
        angle += 180

    return angle


# 두 점 사이의 거리 구하기
def make_distance(x1, y1, x2, y2):
    dist = int(math.sqrt(math.pow(x1-x2, 2) +math.pow(y1-y2, 2)))
    return dist


# 픽셀 사이즈에 따른 공의 크기 구하기
def make_ball_size(h):
    ball_size = round(h/4.27, 2)  # 공이 잡힌 픽셀값에 실제 공의 크기를 곱함
    pix_size = round(1 / ball_size, 2)
    return pix_size


# 구질 판단
def define_line(angle, dist):
    if -5 <= angle <= 5:
        line_ins = "straight"
    elif 5 < angle < 30 and dist < 150:
        line_ins = "fade"
    elif 5 < angle < 30 and dist > 150:
        line_ins = "hook"
    elif -30 < angle < -5 and dist < 150:
        line_ins = "draw"
    elif -30 < angle < -5 and dist > 150:
        line_ins = "slice"
    elif -30 >= angle:
        line_ins = "pull"
    elif 30 <= angle:
        line_ins = "push"
    else:
        line_ins = "no detection line"
    return line_ins


# main
def infer(json_data, out_path, model):
    file_data = json_data["data"]
    file_docid = file_data["creatorId"]
    file_str = "D:/test_in/"
    file_path = file_data["fileName"]
    file = (file_path.split("/")[-1]).split("-")[-1]

    print(f"{file} in froc")

    pix_size = 0
    velocity = 0
    angle_degrees = 0
    init_ball_pos = 0
    parabola_list = []
    line_orb = ""
    last_center = 0

    vidcap = cv2.VideoCapture(file_str + file_path)

    _, frame_w, frame_h, fps = make_video_writer(vidcap)
    total_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_length = round(total_frame / fps, 2)

    res_json = {}
    center_list = []

    res_json["docid"] = file_docid
    res_json["file_name"] = file_path
    res_json["shot"] = "Pitch Shot"
    res_json["status"] = 1
    res_json["total_frame"] = total_frame
    res_json["fps"] = fps
    res_json["size"] = frame_w, frame_h
    res_json["pix_size"] = pix_size
    res_json["velocity"] = velocity
    res_json["impact_angle"] = angle_degrees
    res_json["init_ball_position"] = init_ball_pos
    res_json["line_orb"] = line_orb
    res_json["frame"] = {}
    res_json["last_center"] = last_center
    res_json["results_video_file"] = ""
    res_json["results_json_file"] = ""
    res_json["thumbnail"] = ""

    while vidcap.isOpened():
        ret, frame = vidcap.read()

        cur_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))-1  # 1 이 아닌 0 부터 세기 시작
        res_json["frame"][f"{cur_frame}"] = {}


