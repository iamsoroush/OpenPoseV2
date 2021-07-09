# OpenPoseV2 Tensorflow2/Keras

This is a *Tensorflow2/Keras* Implementation of **OpenPose-Body25(V2)**.
I've converted the weights into a `.h5` file which will be loaded into model's graph that is written using **Tensorflow2/Keras**.
Also I've included Sort tracker for person-tracking and integrated it into OpenPose, which can be found as **DetecTracker** class. 


Here's the [original implementation of **OpenPose**](https://github.com/CMU-Perceptual-Computing-Lab/openpose). This repo is based on [this paper](https://arxiv.org/pdf/1812.08008.pdf).

# How to use?
- Use this command to clone the conda environment:

`conda env create -n tf2_openpose -f environment.lock.yml`

- Or install using `requirements.txt`

- Download the model from [here](https://drive.google.com/file/d/1bccsdNB4CsrjRlRVkFjEps_V_G4DMu_J/view?usp=sharing) and move it to `src/model/` or run `python3 download_model.py`

- Detect and draw poses from a RGB image:
  
```
import cv2
from src import OpenPoseV2, OpenPoseV2Config, HyperConfig

## Instantiate OpenPoseV2
openpose_config = OpenPoseV2Config()
openpose_config.input_res = 512
hyper_config = HyperConfig()
hyper_config.drawing_stick = 2
openpose = OpenPoseV2(openpose_config, hyper_config)

## Load image
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

## Detect
detections = openpose.get_detections(img)

## Visualize
drawed = img.copy()
for detection in detections:
    openpose.draw_detection(drawed, detection)

cv2.imshow(drawed)
```
  
- 

- Take a look at [this colab notebook](https://github.com/iamsoroush/OpenPoseV2/blob/master/tf_openpose.ipynb) as a quick start for using `DetecTracker`.

# Demo
Here's a result from **DetecTracker**:

https://user-images.githubusercontent.com/33500173/124981056-5e589f00-e04a-11eb-868a-713d1a7f5374.mp4
