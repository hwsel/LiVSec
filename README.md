# LiVSec

In the `LiVSec` project, we developed 1) a 3D face authentication system, 2) a generative model which generates 
perturbations that can prevent the face models from being exploited to bypass DL-based face authentications while
maintaining the required quality and functionality of the 3D video surveillance, and 3) an end-to-end security-preserving
live 3D video surveillance system.

The details of this project can be found in our MMSys'23 paper:

Zhongze Tang, Huy Phan, Xianglong Feng, Bo Yuan, Yao Liu, and Sheng Wei. 2023. 
Security-Preserving Live 3D Video Surveillance. 
In Proceedings of the 14th ACM Multimedia Systems Conference (MMSys ’23), June 7–10, 2023, Vancouver, BC, Canada. 
https://doi.org/10.1145/3587819.3590975

The paper can be found under the `paper` folder.

## Repository Hierarchy

```text
├── FaceAuthentication              // Face authentication system
...
├── ProtectedFaceGeneration         // The core of LiVSec, the generative model that adds protection to 3D video surveillance
│   ├── modules                     // Where you should put two pre-trained models
...
├── System                          // Security-preserving live 3D video surveillance system
│   ├── modules                     // Where you should put two pre-trained models
│   ├── John_3dts.mp4               // The source of Dataset #2 w/ timestamp.
...
├── paper
├── LICENSE
└── README.md

```

## 1. Environmental Setup

### Hardware requirements

A workstation with GPU is required to train/infer the models, and run the system.

In our project, an Nvidia RTX A6000 GPU is used to evaluate the system.

### Software requirements

The project is developed and tested in the following environment:
```text
Ubuntu 20.04 LTS
CUDA 11.8
Python 3.7
PyTorch 1.7.1+cu110
PyTorch-Lightning 1.2.3
```
Please set up the development environment by following the instructions below:

1. Update Ubuntu first.

    ```shell
    sudo apt update
    sudo apt upgrade
    ```

2. Install [CUDA](https://developer.nvidia.com/cuda-downloads) (a version >=11.0 && <12.0 is okay)
following the official instructions.

3. (Optional) Install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to setup 
a virtual environment. This will help protect your dev environment from being a mess. 
Set up the Conda virtual environment w/ Python 3.7. We highly recommend using PyCharm GUI to do so directly.
If you prefer the command line, try the following commands to create a virtual environment called `LiVSec_MMSys23`.
    ```shell
    conda create -n LiVSec_MMSys23 python=3.7
    conda activate LiVSec_MMSys23
    ```

4. Install the required Python libraries (in the virtual environment if you have one).

    ```shell
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    pip install matplotlib pytorch-lightning==1.2.3 wandb 
    pip install lpips opencv-python==4.3.0.38
    ```

5. Install `ffmpeg`.
   ```shell
   sudo apt install ffmpeg
   ```
   Please note that in this project, we assume the path to the binary file of `ffmpeg` is `/usr/bin/ffmpeg`. You can find its 
   path in your system by typing `which ffmpeg` in the terminal. If it is different from this path, please modify Line 30 of
   `main.py` under `System` folder to the `ffmpeg` path in your system.

6. Install `v4l2loopback` following the [official instructions](https://github.com/umlaeute/v4l2loopback).
   
   Make sure to enable it before running the LiVSec system.
   ```shell
   sudo modprobe v4l2loopback
   ```
   
   `v4l2loopback` by default will create a virtual camera called `/dev/video0`, run the following code, and if it is created successfully,
the output will be `video0`. Refer to its official documentation for more details.
   
   ```shell
   ls /dev | grep video0
   ```

7. Download the models from https://drive.google.com/drive/folders/17WVDVuHnQpau84fJzXwZFBzaGKyjUEE5?usp=share_link.

   Put two pre-trained models under `./ProtectedFaceGeneration/modules` and `./System/modules`.

## 2. Reproduce the Results of LiVSec

### Prepare datasets

Run `preprocess_data.py` under `FaceAuthentication/utils` to download and preprocess Dataset #1. 

If you want to change where to put the dataset, change Line 126 of `preprocess_data.py`.
The default is `/data/faceid`.
If you change it, you should also
change `--data_path` argument every time you call the models (either the face authentication model or the protected
face generation model).

Dataset #2 comes from `System/John_3dts.mp4`. No additional actions are needed for Dataset #2 since they are all included.

### Results for Dataset #1

Run `result_collect.py` under `ProtectedFaceGeneration` folder. The output will show all the results.

### Results for Dataset #2

Run `main.py` under `System` folder. The output will show all the results.

Please note that the `MODE` in Line 27 has to be set as `RESULT_COLLECT_MODE`.

## 3. Train Your Own Models

You should follow `Prepare datasets` section above to prepare the datasets first.

### Train the face authentication model

Run `train.py` under `FaceAuthentication` folder. If you want to change the training hyper-parameters, check Line 41-59 of the `train.py`.

### Train the protected face generation model

You should have a pretrained face authentication model, and put it under `ProtectedFaceGeneration/modules`.
Change Line 21 of `ProtectedFaceGeneration/main_module.py` to your own face authentication model path.

Run `train.py` under `ProtectedFaceGeneration` folder. If you want to change the training hyper-parameters, check Line 43-59 of the `train.py`.

## 4. The End-to-end Security-preserving Live 3D Video Surveillance System

## 5. Cite Our Work

Zhongze Tang, Huy Phan, Xianglong Feng, Bo Yuan, Yao Liu, and Sheng Wei. 2023. 
Security-Preserving Live 3D Video Surveillance. 
In Proceedings of the 14th ACM Multimedia Systems Conference (MMSys ’23), June 2023. 
https://doi.org/10.1145/3587819.3590975

## 6. Contact

If you have any questions or any idea to discuss, you can email Zhongze Tang (zhongze.tang@rutgers.edu) directly.

## License

MIT