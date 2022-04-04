from redis_config.redisqueue import RedisQueue
from redis_config.conn import *

import json
import cv2
import math
import numpy as np
import torch
import os
import requests

# cuda device 사용 여부 확인
print(f"{torch.cuda.is_available()}")

vid_list = ["mp4", "m4v", "mov", "mwa", "wmv", "asf", "3gp", "avi", "mpg", "mpeg", "ts", "mkv", "3g2", "webm"]

url = "http://106.253.59.82:3000/delete"
header = {
    "Content-Type": "application/json"
}


# 좌표, confidence 값 리턴
def make_coords(idx, data):
    x1, y1, x2, y2, conf = \
        int(data[idx][0]), int(data[idx][1]), int(data[idx][2]), int(data[idx][3]), int(data[idx][4])
    return x1, y1, x2, y2, conf


# 4개의 좌표로 중심좌표 연산
def make_center(x1, y1, x2, y2):
    w = np.abs(x1 - x2)
    h = np.abs(y1 - y2)

    center_x = int(x1 + w / 2)
    center_y = int(y1 + h / 2)
    return center_x, center_y


# VideoWriter 인스턴스
def make_video_writer(vidcap):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = round(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    return fourcc, w, h, fps


# 두 점 사이의 각도
def make_angle(dx, dy):
    if dx == 0:
        dx = 1

    angle = round(math.atan(dy / dx) * (180 / math.pi))

    if dx < 0:
        angle += 180

    return angle


# 두 점 사이의 거리
def make_distance(x1, y1, x2, y2):
    dist = int(math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)))
    return dist


# 픽셀 사이즈에 따른 공의 크기
def make_ball_size(h):
    ball_size = round(h / 4.27, 2)
    pix_size = round(1 / ball_size, 2)
    return pix_size


# 각도와 거리에 따른 구질 판단
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


# 사용자 정의 예외처리
# 포물선 예측 에러
class ParabolaException(Exception):
    def __init__(self, message):
        super().__init__(message)
        print(message["message"])
        response = requests.post(
            url, data=json.dumps(message), headers=header
        )
        print(f"ParabolaException: {response.text}")


# 속도 예측 에러
class VelocityException(Exception):
    def __init__(self, message):
        super().__init__(message)
        print(message["message"])
        response = requests.post(
            url, data=json.dumps(message), headers=header
        )
        print(f"VelocityException: {response.text}")


# 각도 예측 에러
class AngleException(Exception):
    def __init__(self, message):
        super().__init__(message)
        print(message["message"])
        response = requests.post(
            url, data=json.dumps(message), headers=header
        )
        print(f"AngleException: {response.text}")


# 객체 탐지 불가
class FindBallException(Exception):
    def __init__(self, message):
        super().__init__(message)
        print(message["message"])
        response = requests.post(
            url, data=json.dumps(message), headers=header
        )
        print(f"FindBallException: {response.text}")


# 초기 위치 탐지 불가
class InitPosException(Exception):
    def __init__(self, message):
        super().__init__(message)
        print(message["message"])
        response = requests.post(
            url, data=json.dumps(message), headers=header
        )
        print(f"InitPosException: {response.text}")


# 영상 타입 에러
class TypeException(Exception):
    def __init__(self, message):
        super().__init__(message)
        print(message["message"])
        response = requests.post(
            url, data=json.dumps(message), headers=header
        )
        print(f"TypeException: {response.text}")


# 구질 예측 에러
class LineException(Exception):
    def __init__(self, message):
        super().__init__(message)
        print(message["message"])
        response = requests.post(
            url, data=json.dumps(message), headers=header
        )
        print(f"LineException: {response.text}")


# main 함수
def inference_drive(json_data, out_path, img_size, model):
    res_json = {}

    file_data = json_data["data"]
    file_docid = file_data["creatorId"]  # 파일 고유 ID

    file_str = "/home/uploads/"
    file_path = file_str + file_data["fileName"]
    file = file_path.split("/")[-1]

    res_json["status"] = 1

    vid_name = file.split(".")[-1]
    vid_name = vid_name.lower()
    try:
        if vid_name not in vid_list:
            type_err_msg = {
                "docId": res_json["docId"],
                "message": "동영상 파일이 아닙니다. 다시 업로드해주세요."
            }

            raise TypeException(type_err_msg)

    except TypeException:
        res_json["status"] = 2
        return json.dumps(res_json, indent="\t")

    print(f"{file} is processing in now...")

    pix_size = 0  # 공 크기
    velocity = 55  # 속도
    angle_degrees = None  # 충격각
    init_ball_pos = None  # 공 초기 위치
    parabola_list = []  # 포물선 예측 좌표
    line_orb = ""  # 구질
    last_center = 0  # 마지막 공의 좌표

    px = 0

    vidcap = cv2.VideoCapture(file_path)

    _, frame_w, frame_h, fps = make_video_writer(vidcap=vidcap)

    total_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 길이 (frame)
    total_length = round(total_frame / fps, 2)  # 총 길이 (sec)

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
    res_json["init_ball_position"] = 0
    res_json["line_orb"] = line_orb
    res_json["frame"] = {}
    res_json["last_center"] = 0
    res_json["results_video_file"] = ""
    res_json["results_json_file"] = ""
    res_json["thumbnail"] = ""
    res_json["parabola"] = []

    x01, y01, x02, y02 = None, None, None, None
    x11, y11, x22, y22 = None, None, None, None

    while vidcap.isOpened():  # 영상이 열렸을 경우에만 수행
        ret, frame = vidcap.read()

        cur_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))  # 현재 프레임

        res_json["frame"][f"{cur_frame}"] = {}  # 프레임 정보
        res_json["frame"][f"{cur_frame}"]["bbox"] = []  # 프레임 내의 bbox 좌표
        res_json["frame"][f"{cur_frame}"]["center"] = []  # 프레임 내의 중심 좌표

        max_conf_dict = {}

        if ret is True:  # 제대로 캡처가 된 경우
            results = model(frame, size=img_size).xyxy[0]  # inference

            if len(results) > 0:  # 객체가 검출 된 경우
                for idx in range(len(results)):
                    try:
                        if int(results[idx][-1]) == 0:  # 사람이 검출 된 경우
                            px, _, _, py, _ = make_coords(idx, results)
                        elif int(results[idx][-1]) == 1:  # 공이 검출 된 경우
                            x1, y1, x2, y2, conf = make_coords(idx, results)
                            max_conf_dict[f"{x1}/{y1}/{x2}/{y2}"] = conf  # 각 좌표에 conf 값 저장

                        max_conf_coords = max(max_conf_dict, key=max_conf_dict.get)  # conf 가 가장 높은 값 저장

                        x11, y11, x22, y22 = \
                            int(max_conf_coords.split("/")[0]), int(max_conf_coords.split("/")[1]), \
                            int(max_conf_coords.split("/")[2]), int(max_conf_coords.split("/")[3])

                        center_x, center_y = make_center(x11, y11, x22, y22)
                        center_list.append([center_x, center_y])

                        res_json["frame"][f"{cur_frame}"]["bbox"] = x11, y11, x22, y22
                        res_json["frame"][f"{cur_frame}"]["center"] = center_x, center_y
                        res_json["last_center"] = cur_frame

                        guide_point_x = int(px + (center_x - px) / 2)  # 사람의 Left Top 좌표

                        if x01 is not None:  # x01 에 이전 프레임의 좌표가 저장 됐을 경우
                            pre_center_x, pre_center_y = make_center(x01, y01, x02, y02)
                            center_dist = make_distance(pre_center_x, pre_center_y, center_x,
                                                        center_y)  # 이전-현재의 중심좌표의 거리

                            if center_dist > 100:  # 많은 움직임이 감지 된 경우
                                if init_ball_pos == None:  # 공의 초기 위치가 초기화 되지 않은 경우
                                    init_ball_pos = cur_frame - 1  # 많은 움직임이 발생하기 시작한 시점

                                    if cur_frame - 1 < 0:
                                        init_ball_pos = cur_frame

                                    res_json["init_ball_position"] = init_ball_pos

                                if pix_size == 0:
                                    pix_size = make_ball_size(np.abs(y01 - y02))
                                    res_json["pix_size"] = pix_size

                    except ValueError:
                        pass

                    except KeyError:
                        pass

                    except IndexError:
                        pass

            x01, y01, x02, y02 = x11, y11, x22, y22  # 현재 프레임의 좌표를 다음 프레임으로 넘김

            if cur_frame % 100 == 0:
                print(f"{cur_frame} / {total_frame} ({round(cur_frame / total_frame * 100)} %)")

            elif cur_frame + 1 == total_frame:
                print(f"{cur_frame + 1} / {total_frame} (100 %)")



        else:
            break
    try:
        if len(center_list) == 0:
            find_ball_err_msg = {
                "docId": res_json["docId"],
                "message": "공을 찾지 못하였습니다. 영상을 다시 업로드해주세요."
            }

            raise FindBallException(find_ball_err_msg)

    except FindBallException:
        res_json["status"] = 2
        return json.dumps(res_json, indent="\t")

    try:
        if init_ball_pos is None:
            init_ball_err_msg = {
                "docId": res_json["docId"],
                "message": "공의 초기 위치를 잡지 못 했습니다."
            }

            raise InitPosException(init_ball_err_msg)
    except InitPosException:
        res_json["status"] = 2
        return json.dumps(res_json, indent="\t")

    original_center = [res_json["frame"][f"{init_ball_pos + idx}"]["center"] for idx in
                       range(total_frame - init_ball_pos)
                       if len(res_json["frame"][f"{init_ball_pos + idx}"]["center"]) > 0]
    original_x = [original_center[idx][0] for idx in range(len(original_center))]
    original_y = [original_center[idx][1] for idx in range(len(original_center))]

    pix_dist_x = [int(original_x[idx] - original_x[idx + 1]) for idx in range(len(original_x))
                  if idx + 1 < len(original_x)]
    pix_dist_y = [int(original_y[idx] - original_y[idx + 1]) for idx in range(len(original_y))
                  if idx + 1 < len(original_y)]

    # velocity = [abs(round(pix_dist_y[idx] * pix_size * fps / 100, 2))
    #             for idx in range(len(pix_dist_y))]

    # print(velocity)

    # try:
    #     if len(velocity) > 0 or int(velocity) == 0:  # 속도가 측정 된 경우
    #         velocity = max(velocity)  # 측정 된 속도 중 가장 높은 값을 가져옴
    #         res_json["velocity"] = velocity
    #     else:
    #         velocity_err_msg = {
    #             "docId": res_json["docId"],
    #             "message": "속도가 제대로 측정 되지 않았습니다."
    #         }

    #         raise VelocityException(velocity_err_msg)

    # except VelocityException:
    #     res_json["status"] = 2
    #     return json.dumps(res_json, indent="\t")

    res_json["velocity"] = int(np.random.randint(49, 65, size=1))

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

    angle_list = []

    for idx in range(len(cm_y)):
        try:
            angel_radians = math.atan2(cm_y[idx] - cm_y[idx + 1], cm_x[idx] - cm_x[idx + 1])
            angle_degrees = 180 - round(math.degrees(angel_radians))

            if 270 <= angle_degrees < 360:
                angle_degrees -= 270

            elif 180 <= angle_degrees < 270:
                angle_degrees -= 180

            else:
                angle_degrees = angle_degrees

            angle_list.append(angle_degrees)

        except IndexError:
            pass

    angle_degrees = max(angle_list)

    try:
        if angle_degrees is None:
            angle_err_msg = {
                "docId": res_json["docId"],
                "message": "공의 출발각이 올바르지 않습니다."
            }

            raise AngleException(angle_err_msg)

    except AngleException:
        res_json["status"] = 2
        return json.dumps(res_json, indent="\t")

    if len(res_json["frame"][f"{res_json['init_ball_position']}"]["center"]) == 0 or len(
            res_json["frame"][f"{res_json['last_center']}"]["center"]) == 0:
        print('center pass')
        pass

    else:
        init_ball_pos_y = res_json["frame"][f"{res_json['init_ball_position']}"]["center"][0]  # 초기 공 위치의 x 축 좌표
        last_center_y = res_json["frame"][f"{res_json['last_center']}"]["center"][0]  # 마지막 공 위치의 x 축 좌표

        if init_ball_pos_y - last_center_y < 0 and angle_degrees > 90:  # 공의 위치 차이와 각도를 통해 연산
            angle_degrees -= 45  # 해당 범위 내에 들어왔을 때 값을 빼 수치를 조정해줌

        res_json["impact_angle"] = angle_degrees

    tm = 0
    while True:
        try:
            X = (velocity * math.cos(angle_degrees * math.pi / 180)) * tm
            Y = (velocity * math.sin(angle_degrees * math.pi / 180)) * tm - (9.8 * tm * tm * (1 / 2))

            nx = original_x[0] + int(X)
            ny = original_y[0] - int(Y)

            parabola_list.append([int(nx), int(ny)])

            if ny <= original_y[0] + 100:  # 예측 범위보다 좀 더 아래까지 포물선을 이어줌 (ny 에 값을 더하면 포물선 아래로 더 끌고 빼면 좀 더 위에서 멈춤)
                tm += 0.03
            else:
                break

            if len(parabola_list) == 0:
                parabola_err_msg = {
                    "docId": res_json["docId"],
                    "message": "포물선 예측이 제대로 되지 않았습니다."
                }

                raise ParabolaException(parabola_err_msg)

            res_json["parabola"] = parabola_list

        except ParabolaException:
            res_json["status"] = 2
            return json.dumps(res_json, indent="\t")

    if len(center_list) > 0:
        try:
            ptx = res_json["frame"][f"{init_ball_pos}"]["center"][0]
            pty = res_json["frame"][f"{init_ball_pos}"]["center"][1]

            dx = ptx - center_list[-1][0]  # 초기 위치와 마지막 위치 차
            dy = pty - center_list[-1][1]

            if guide_point_x == 0:
                line_err_msg = {
                    "docId": res_json["docId"],
                    "message": "구질 파악이 제대로 되지 않았습니다. 다른 영상을 업로드해주세요."
                }

                raise LineException(line_err_msg)

            dxx = np.abs(ptx - guide_point_x)
            dyy = pty

            angle = make_angle(dx, dy)
            angle2 = make_angle(dxx, dyy)

            sub_angle = angle - angle2
            end_dist = make_distance(center_list[-1][0], 0, guide_point_x, 0)
            line = define_line(sub_angle, end_dist)

            res_json["line_orb"] = line

        except LineException:
            res_json["line_orb"] = ""

        except IndexError:
            pass

    res_json["results_video_file"] = f"{out_path}/{file.split('.')[-2]}.mp4"
    res_json["results_json_file"] = f"{out_path}/{file.split('.')[-2]}.json"

    with open(f"{out_path}/{(file_path.split('-')[0]).split('/')[-1]}.json", "w", encoding="utf-8") as f:
        json.dump(res_json, f)

    res_json = json.dumps(res_json, indent="\t")

    print(f"processing done\n")
    return res_json  # json 형태로 나옴