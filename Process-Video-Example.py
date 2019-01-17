import os

import cv2


global CV_CAP_PROP_POS_FRAMES = 1


def video_path(file=""):
    file = file.strip()
    if file is "":
        raise ValueError('No filename provided!')
    if os.path.exists(os.path.abspath(file)):
        return os.path.abspath(file)
    else:
        raise OSError('No Such File Found!')


def sift_through_vid(position=None, cap=None):
    if cap is None:
        raise ValueError('No VideoCapture Object was provided')
    if position is None:
        raise ValueError('No position frame was provided')
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            cv2.imshow('video', frame)
            pos_frame = cap.get(CV_CAP_PROP_POS_FRAMES)
            print(str(pos_frame)+" frames")
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            print('frame is not ready')
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.CV_CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break


def manage_video(vid=""):
    vid = video_path(vid)
    cap = cv2.VideoCapture(vid)
    while not cap.isOpened():
        cap = cv2.VideoCapture(vid)
        cv2.waitKey(1000)
        print('Waiting for video Header')
    return cap


def  process_video(vid=""):
    try:
        cap = manage_video(vid)
        pos_frame = cap.get(CV_CAP_PROP_POS_FRAMES)
        sift_through_vid(position=pos_frame, cap=cap) #goes through the video from the position frame
    except Exception  as e:
        print(e)
    finally:
        print('Session Terminated...')

    return
