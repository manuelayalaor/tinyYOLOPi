import os

import cv2


FRAMES_TO_SAVE = 30


def video_path(file=""):
    file = file.strip()
    if file is "":
        raise ValueError('No filename provided!')
    if os.path.exists(file):
        return os.path.abspath(file)
    else:
        raise OSError('No Such File Found!')


def scrub_vid(position=None, cap=None):
    if cap is None:
        raise ValueError('No VideoCapture Object was provided')
    if position is None:
        raise ValueError('No position frame was provided')
    current_frame = 0 #use this to keep track of frames to save
    while True:
        flag, frame = cap.read()
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if flag:
            # The frame is ready and already captured
            cv2.imshow('video', frame)
            if current_frame % FRAMES_TO_SAVE == 0:
                cv2.imwrite("Frame-%d.jpg" % current_frame, frame)

            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_frame += 1
            print("Frame:[" + str(pos_frame) + "]")
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print('frame is not ready')
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
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


def process_video(vid=""):
    try:
        cap = manage_video(vid)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        scrub_vid(position=pos_frame, cap=cap)#goes through the video from the position frame
    except Exception as e:
        print(e)
    finally:
        print('Session Terminated...')
    return


DEFAULT_VIDEO_FILE = "pedestrian-dataset/crosswalk.avi"


def main():
    arg = ""
    print("If you wish to use the default file write: default")
    while arg.strip() is "":
        try:
            arg = str(input("provide video file to process:")).strip()
            if not arg:
                continue

            elif arg.lower() == "default":
                process_video(vid=DEFAULT_VIDEO_FILE)

            elif arg:
                process_video(vid=arg)
        except Exception as e:
            arg = ""
            print(e)
            print("Sorry invalid filename, try again!")

    return


if __name__ == "__main__":
    main()
