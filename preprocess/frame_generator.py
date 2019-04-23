from .process_video import manage_video
import cv2
import csv

VIDEOS = ["preprocess/pedestrian-dataset/crosswalk.avi", "preprocess/pedestrian-dataset/fourway.avi", "preprocess/pedestrian-dataset/night.avi" ]

def frame_generator():
    counter = 0

    for i in VIDEOS:
        csv_path = i.replace(".avi", ".csv")
        with open(i.replace(".avi", ".csv")) as _csv:
            csv_reader = csv.reader(_csv)
            if counter == 0:
                next(csv_reader)
            try:
                cap = manage_video(vid=i)
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if cap is None:
                    raise ValueError('No VideoCapture Object was provided')
                if pos_frame is None:
                    raise ValueError('No position frame was provided')
                while True:
                    flag, frame = cap.read()
                    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if flag:
                        # The frame is ready and already captured
                        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        yield frame, next(csv_reader)
                        counter += 1
                    else:
                        # The next frame is not ready, so we try to read it again
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                        # It is better to wait for a while for the next frame to be ready
                        cv2.waitKey(1000)

                    if cv2.waitKey(10) == 27:
                        break
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        # If the number of captured frames is equal to the total number of frames,
                        # we stop
                        break
            except Exception as e:
                print(e)
            finally:
                print('Session Terminated...')