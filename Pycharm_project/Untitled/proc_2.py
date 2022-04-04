import cv2
import math
import numpy as np
import torch
import json

vid_list = ["mp4", "m4v", "mov", "mwa", "wmv", "asf", "3gp", "avi", "mpg", "mpeg", "ts", "mkv", "3g2", "webm"]

ball_model = torch.hub.load(
    repo_or_dir="D:/pycharmProject/OT-OD/yolov5/",
    model="custom", source="local",
    path="D:/pycharmProject/OT-OD/yolov5/03_23_1280/03_23_1280/weights/best.pt"
)

person_model = torch.hub.load(
    repo_or_dir="D:/pycharmProject/OT-OD/yolov5",
    model="custom", source="local",
    path="D:/pycharmProject/OT-OD/yolov5/03_23_1280/03_23_1280/weights/best.pt"
)

ball_model.conf = 0.7
ball_model.iou = 0.3
ball_model.classes = [1]

person_model.conf = 0.7
person_model.iou = 0.3
person_model.classes = [0]


def make_coords(idx, data):
    x1, y1, x2, y2, conf = \
        int(data[idx][0]), int(data[idx][1]), int(data[idx][2]), int(data[idx][3]), round(data[idx][4], 2)
    return x1, y1, x2, y2, conf


def make_center(x1, y1, x2, y2):
    w = np.abs(x1 - x2)
    h = np.abs(y1 - y2)

    center_x = int(x1 + w / 2)
    center_y = int(y1 + h / 2)
    return center_x, center_y


def make_video_writer(vidcap):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = round(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    total_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fourcc, w, h, fps, total_frame


def make_angle(dx, dy):
    if dx == 0:
        dx = 1

    angle = round(math.atan(dy / dx) * 180 / math.pi)

    if dx < 0:
        angle += 180

    return angle


def make_distance(x1, y1, x2, y2):
    dist = int(math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)))
    return dist


def make_ball_size(h):
    ball_size = round(h / 4.27, 2)
    pix_size = round(1 / ball_size, 2)
    return pix_size


def define_line(angle, dist):
    if -5 <= angle <= 5:
        line_ins = "st"

    elif 5 < angle < 30 and dist < 150:
        line_ins = "fd"

    elif 5 < angle < 30 and dist > 150:
        line_ins = "hk"

    elif -30 < angle < -5 and dist < 150:
        line_ins = "dr"

    elif -30 < angle < -5 and dist > 150:
        line_ins = "sl"

    elif -30 >= angle:
        line_ins = "pl"

    elif 30 <= angle:
        line_ins = "ps"

    else:
        line_ins = "no detection line"

    return line_ins


def infer(json_data, out_path, img_size, model):
    res_json = {}

    file_data = json_data["data"]
    file_docid = file_data["creatorId"]
    file = file_data["fileName"]

    file_str = "D:/test_in/"
    file_path = file_str + file

    res_json["status"] = 1

    vid_name = file.split(".")[-1]
    vid_name = vid_name.lower()

    if vid_name not in vid_list:
        print(f"{file} is not video format.")
        pass

    pix_size = 0
    velocity = 0
    angle_degrees = None
    init_ball_pos = None
    parabola_list = []
    line_ins = ""

    px1, py1, px2, py2 = None, None, None, None
    x01, y01, x02, y02 = None, None, None, None
    x11, y11, x22, y22 = None, None, None, None

    vidcap = cv2.VideoCapture(file_path)

    _, frame_w, frame_h, fps, total_frame = make_video_writer(vidcap)

    center_list = []

    res_json["docId"] = file_docid
    res_json["file_name"] = file_path
    res_json["shot"] = json_data["data"]["shot"]
    res_json["total_frame"] = total_frame
    res_json["fps"] = fps
    res_json["size"] = frame_w, frame_h
    res_json["pix_size"] = pix_size
    res_json["velocity"] = velocity
    res_json["impact_angle"] = angle_degrees
    res_json["init_ball_position"] = init_ball_pos
    res_json["line_orb"] = line_ins
    res_json["frame"] = {}
    res_json["last_center"] = 0
    res_json["results_video_file"] = ""
    res_json["results_json_file"] = ""
    res_json["thumbnail"] = ""
    res_json["parabola"] = parabola_list

    while vidcap.isOpened():
        ret, frame = vidcap.read()

        cur_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))

        res_json["frame"][f"{cur_frame}"] = {}
        res_json["frame"][f"{cur_frame}"]["bbox"] = []
        res_json["frame"][f"{cur_frame}"]["center"] = []

        max_conf_dict = {}

        if ret is True:
            person_results = person_model(frame, size=img_size).xyxy[0]
            ball_results = ball_model(frame, size=img_size).xyxy[0]

            if len(person_results) > 0:
                ball_pos_list = [make_coords(idx, ball_results) for idx in range(len(ball_results))]
