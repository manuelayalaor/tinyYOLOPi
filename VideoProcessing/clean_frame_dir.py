import os

path = "../captured-frames"
for file in os.scandir(path):
    if file.name.endswith(".jpg"):
        os.unlink(file.path)