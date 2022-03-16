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
    return fourcc, w, h, fps


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
    ball_size = round(h / 4.27, 2)  # 공이 잡힌 픽셀값에 실제 공의 크기를 계산
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
    velocity = None
    angle_degrees = 0
    init_ball_pos = 0
    parabola_list = []
    line_orb = ""
    last_center = None

    vidcap = cv2.VideoCapture(file_str + file_path)

    fourcc, frame_w, frame_h, fps = make_video_writer(vidcap)

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
    res_json["init_ball_position"] = 0
    res_json["line_orb"] = line_orb
    res_json["frame"] = {}
    res_json["last_center"] = last_center
    res_json["results_video_file"] = ""
    res_json["results_json_file"] = ""
    res_json["thumbnail"] = ""

    x01, y01, x02, y02 = None, None, None, None
    x11, y11, x22, y22 = None, None, None, None

    while vidcap.isOpened():
        ret, frame = vidcap.read()

        cur_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))-1  # 1 이 아닌 0 부터 세기 시작
        res_json["frame"][f"{cur_frame}"] = {}  # frame 정보
        res_json["frame"][f"{cur_frame}"]["bbox"] = []  # 해당 프레임의 bbox
        res_json["frame"][f"{cur_frame}"]["center"] = []  # 해당 프레임의 중심좌표

        max_conf_dict = {}

        if ret is True:  # 영상이 제대로 열린 경우
            results = model(frame, size=1920).xyxy[0]  # 분석

            if len(results) > 0:  # 객체가 검출 된 경우
                for idx in range(len(results)):
                    try:
                        if int(results[idx][-1]) == 0:  # 사람이 검출 된 경우
                            px, _, _, py, _ = make_coords(idx, results)
                        elif int(results[idx][-1]) == 1:  # 공이 검출 된 경우
                            x1, y1, x2, y2, conf = make_coords(idx, results)
                            max_conf_dict[f"{x1}/{y1}/{x2}/{y2}"] = conf  # 공이 검출 된 좌표의 conf 값들을 저장함

                        max_conf_coords = max(max_conf_dict, key=max_conf_dict.get)  # max_conf_dict 중 conf 값이 가장 높은 값을 가져옴

                        x11, y11, x22, y22 = \
                            int(max_conf_coords.split("/")[0]), int(max_conf_coords.split("/")[1]), \
                            int(max_conf_coords.split("/")[2]), int(max_conf_coords.split("/")[3])
                        # 해당 프레임의 최종 좌표값

                        center_x, center_y = make_center(x11, y11, x22, y22)
                        center_list.append([center_x, center_y])

                        res_json["frame"][f"{cur_frame}"]["bbox"] = x11, y11, x22, y22
                        res_json["frame"][f"{cur_frame}"]["center"] = center_x, center_y
                        res_json["last_center"] = cur_frame

                        if x01 is not None:
                            pre_center_x, pre_center_y = make_center(x01, y01, x02, y02)
                            center_dist = make_distance(pre_center_x, pre_center_y, center_x, center_y)

                            if center_dist > 100:
                                if init_ball_pos == 0:
                                    init_ball_pos = cur_frame - 1  # 많은 움직임이 시작 된 위치 저장
                                    res_json["init_ball_position"] = init_ball_pos

                                if pix_size == 0:
                                    pix_size = make_ball_size(np.abs(y01 - y02))
                                    """
                                    dist 가 100 이상이면 이전 프레임과 현재 프레임의 차이가 크다는 것이고,
                                    이는 현재 프레임이 직전 프렐임보다 많이 움직였으며 이전 프레임은 공의 초기 위치라고 판단 가능
                                    """
                                    res_json["pix_size"] = pix_size

                    except ValueError:
                        pass

                    except KeyError:
                        pass

                    except IndexError:
                        pass

            x01, y01, x02, y02 = x11, y11, x22, y22  # 현재 프레임의 좌표 정보를 다음 프레임으로 넘겨줌

            if cur_frame % 100 == 0:
                print(f"{cur_frame}/{total_frame} ({round(cur_frame / total_frame * 100)})%")

        else:
            break

    original_center = [res_json["frame"][f"{init_ball_pos+idx}"]["center"] for idx in range(total_frame-init_ball_pos)
                       if len(res_json["frame"][f"{init_ball_pos+idx}"]["center"]) > 0]
    original_x = [original_center[idx][0] for idx in range(len(original_center)) if len(original_center) > 0]
    original_y = [original_center[idx][1] for idx in range(len(original_center)) if len(original_center) > 0]
    # 공의 초기 위치에서부터 연산 시작

    pix_dist_x = [int(original_x[idx] - original_x[idx+1]) for idx in range(len(original_x))
                  if idx+1 < len(original_x)]
    pix_dist_y = [int(original_y[idx]) - original_y[idx+1] for idx in range(len(original_y))
                  if idx+1 < len(original_y)]

    velocity = [abs(round(pix_dist_y[idx] * pix_size * fps / 100, 2))
                for idx in range(len(pix_dist_y))]

    if len(velocity) > 0:
        velocity = max(velocity)  # 연산 된 값들 중 가장 높은 값만 가져온다
        res_json["velocity"] = velocity

    cm_x = [round(pix_dist_x[idx] * velocity / fps, 2) for idx in range(len(pix_dist_x))]
    cm_y = [round(pix_dist_y[idx] * velocity / fps, 2) for idx in range(len(pix_dist_y))]

    xy = []
    if len(center_list) > 0:
        xy = [center_list[0]]

    temp = 0
    temp2 = 0

    for idx in range(len(cm_x)):
        temp += cm_x[idx]
        temp2 += cm_y[idx]

        xy.append([round(xy[idx][0] + temp, 2), round(xy[idx][1] + temp2, 2)])

    for idx in range(len(cm_y)):
        try:
            angle_radians = math.atan2(cm_y[idx]-cm_y[idx+1], cm_x[idx]-cm_x[idx+1])
            angle_degrees = 180 - round(math.degrees(angle_radians))

            if angle_degrees > 180:
                angle_degrees -= 180

            res_json["impact_angle"] = angle_degrees

        except IndexError:
            pass

    tm = 0
    while True:
        X = (velocity * math.cos(angle_degrees * math.pi / 180)) * tm
        Y = (velocity * math.sin(angle_degrees * math.pi / 180)) * tm - (9.8 * tm * tm * (1/2))

        nx = original_x[0] + int(X)
        ny = original_y[0] - int(Y)

        parabola_list.append([int(nx), int(ny)])

        if ny <= original_y[0]:
            tm += 0.03
        else:
            break

        res_json["parabola"] = parabola_list

        with open(f"D:/test_out/{file.split('.')[-2]}.json", "w") as f:
            json.dump(res_json, f)

    return res_json

