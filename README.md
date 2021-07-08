# OpenPoseV2 Tensorflow2/Keras
This is a *Tensorflow2/Keras* Implementation of **OpenPose-Body25(V2)**.
I've converted the weights into a `.h5` file which will be loaded into model's graph that is written using **Tensorflow2/Keras**.
Also I've included two trackers for person-tracking and integrated it into OpenPose, which can be found as **DetecTracker** class. 

Here's the original implementation of **OpenPose**. This repo is based on this paper.

# How to use?
- Use this command to clone the conda environment:

`conda env create -n tf2_openpose -f environment.lock.yml`

- Download the model from [here](https://drive.google.com/file/d/1bccsdNB4CsrjRlRVkFjEps_V_G4DMu_J/view?usp=sharing) and move it to `src/model/` or run `python3 download_model.py`
- Take a look at [this colab notebook](https://colab.research.google.com/drive/1SJ5lgcgBjdcgyDHtXuLKtJrpJXXXjfNe?usp=sharing) as a quick start.

# Demo
Here's a result from **DetecTracker**:



https://user-images.githubusercontent.com/33500173/124981056-5e589f00-e04a-11eb-868a-713d1a7f5374.mp4



<!-- <video width="320" height="240" controls>
  <source src="https://github.com/iamsoroush/OpenPoseV2/blob/master/out.mp4" type="video/mp4">
</video>

<!-- ![Alt text](https://github.com/iamsoroush/OpenPoseV2/blob/master/out.mp4) -->

<!-- [result]: https://github.com/iamsoroush/OpenPoseV2/blob/master/out.mp4 "DetecTracker Result" --> -->
