import os

import cv2

from proc import *


def draw_polylines(frame, coords_list, RGB):
    coords_list = np.array(coords_list)

    frame = cv2.polylines(
        frame, [coords_list],
        isClosed=False, color=(0, 0, 0),
        thickness=10
    )

    coords_list = np.ndarray.tolist(coords_list)
    return coords_list, frame


def draw_dot(frame, coords_list, RGB):
    frame = cv2.circle(
        frame, (coords_list[0], coords_list[1]), radius=5, color=(int(RGB[2]), int(RGB[1]), int(RGB[0])),
        thickness=-1
    )
    return frame


def drawing(json_data, out_path):
    results_json = {}

    file_str = "D:/test_in/"
    file_path = json_data["file_name"]
    file = file_path.split("/")[-1]

    parabola = json_data["parabola"]

    print(f"{file} drawing..")

    vidcap = cv2.VideoCapture(file_str + file)

    total_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc, w, h, fps = make_video_writer(vidcap)

    out = cv2.VideoWriter(
        out_path + file.split(".")[-2] + "_test.mp4", fourcc, fps, (w, h)
    )

    center_list = []

    RGB = np.random.randint(0, 255, size=3)

    parabola_param = 0

    idx = 0

    choose_draw_func = np.random.randint(0, 2, size=1)  # 그림을 그릴 때 공으로 그릴지 선으로 그릴지

    while vidcap.isOpened():
        ret, frame = vidcap.read()

        cur_frame = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))

        if ret is True:
            for k in json_data["frame"]:
                if k == str(cur_frame) and int(k) < int(json_data["init_ball_position"]):  # 최초 공이 잡히지 않은 경우엔 워터마크와 글만 적음
                    frame = cv2.putText(
                        frame, f"{cur_frame}", (50, 50), color=(int(RGB[2]), int(RGB[1]), int(RGB[0])),
                        thickness=3, lineType=cv2.LINE_AA, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3
                    )
                    out.write(frame)

                elif k == str(cur_frame) and int(k) >= int(json_data["init_ball_position"]):  # 최초로 공이 잡힌 순간
                    try:
                        last_center = json_data["frame"][str(json_data["last_center"])]["center"]  # last center 의 중심좌표
                        center_x = json_data["frame"][k]["center"][0]  # 중심좌표
                        center_y = json_data["frame"][k]["center"][1]
                        center_list.append([center_x, center_y])
                    except IndexError:
                        pass
                    if k == str(cur_frame) and int(k) < int(json_data["last_center"]):  # 공이 마지막으로 잡힌 순간보다 이전
                        if choose_draw_func == 0:
                            for i in range(len(center_list)):
                                frame = draw_dot(frame, coords_list=center_list[i], RGB=RGB)

                        if choose_draw_func == 1:
                            center_list, frame = draw_polylines(frame, coords_list=center_list, RGB=RGB)

                        frame = cv2.putText(
                            frame, f"{cur_frame}", (50, 50), color=(int(RGB[2]), int(RGB[1]), int(RGB[0])),
                            thickness=3, lineType=cv2.LINE_AA, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3
                        )

                        out.write(frame)

                    elif k == str(cur_frame) and int(k) > int(json_data["last_center"]):
                        parabola_idx = int(len(json_data["parabola"]) / fps / 2)

                        if len(last_center) != 0:  # last center 가 제대로 들어온 경우
                            last_center = np.array([last_center])
                            if parabola_param == 0:
                                parabola = np.array(parabola)
                                parabola = last_center + (parabola - parabola[0])  # 포물선 좌표에 공의 마지막 좌표값을 더해 해당 좌표부터 포물선이 시작하도록 함
                                parabola_param += 1

                            else:
                                parabola = np.array(parabola)

                        if type(parabola) != np.array:
                            parabola = np.array(parabola)

                        _, frame = draw_polylines(frame, center_list, RGB)

                        frame = cv2.putText(
                            frame, f"{cur_frame}", (50, 50), color=(int(RGB[2]), int(RGB[1]), int(RGB[0])),
                            thickness=3, lineType=cv2.LINE_AA, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3
                        )

                        if choose_draw_func == 0:
                            for i in range(len(parabola[:idx+parabola_idx])):
                                frame = draw_dot(frame, parabola[:idx+parabola_idx][i], RGB)

                        elif choose_draw_func == 1:
                            frame = cv2.polylines(
                                frame, [parabola[:idx + parabola_idx]],  # 일정 길이로 잘라서 그려준다
                                isClosed=False, color=(int(RGB[2]), int(RGB[1]), int(RGB[0])),
                                thickness=10
                            )

                        parabola = np.ndarray.tolist(parabola)

                        idx += parabola_idx

                        out.write(frame)

            if cur_frame % 100 == 0:
                print(f"{cur_frame}/{total_frame} ({round(cur_frame / total_frame * 100)})%")

            if cur_frame+1 == total_frame:
                print(f"{cur_frame+1}/{total_frame} (100)%")

        else:
            break

    print(f"{file} drawing done\n")


file_list = os.listdir("D:/test_in/")

for file in file_list:
    try:
        jj = {"data": {"creatorId": "ffasfdfs", "fileName": f"{file}", "flag": "Insert", "shot": "Tee Shot"}}

        enc_json = infer(json_data=jj, out_path="D:/test_out/", model=model)
        drawing(enc_json, out_path="D:/test_out/")

    except:
        print(file, "has error")
        pass
