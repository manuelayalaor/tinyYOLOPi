# tinyYOLOPi
Tiny YOLO on a Raspberry Pi
<p>The primary goal for this repository is to create, train and utilize a CNN with YOLO capabilities for a small UAV prototype.</p>

## Tasks
- Decide on image size(width-height) and (RGB)color channels to use 
- Create Training Data set(decide on epoch, batch-size)
- Create Experimental Data set (The model shall never be exposed to these beforehand, this is to avoid overfitting)
- Modify jupyter notebook file to desired loss-function(currently using softmax-crossentropy)
- Plot and save results in an ordered database(This is for comparing the loss functions to decide the best approach for our use case)
