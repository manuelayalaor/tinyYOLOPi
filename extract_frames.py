try:
    from tinyYOLOPi.preprocess.frame_generator import frame_generator
except:
    from preprocess.frame_generator import frame_generator

import cv2
tester = frame_generator()
current_frame = 0
for i in tester:
    cv2.imwrite("preprocess/pedestrian-dataset/extracted/Frame-%d.jpg" % current_frame, i)
    current_frame+=1
