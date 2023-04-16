# LiVSec - MMSys'23

In the *LiVSec* project, to defend against the face spoofing attacks that face authentication systems can be effectively 
compromised by the 3D face models presented in the 3D surveillance video, 
we propose to proactively and benignly inject adversarial perturbations to the surveillance video in real time,
which prevents the face models from being exploited to bypass deep learning-based face authentications while maintaining
the required quality and functionality of the 3D video surveillance. The details of this project can be found in our MMSys'23 paper:

Zhongze Tang, Huy Phan, Xianglong Feng, Bo Yuan, Yao Liu, and Sheng Wei. 2023. 
Security-Preserving Live 3D Video Surveillance. 
In Proceedings of the 14th ACM Multimedia Systems Conference (MMSys ’23), June 7–10, 2023, Vancouver, BC, Canada. 
https://doi.org/10.1145/3587819.3590975

The paper can be found under the `paper` folder.

This repo contains both code and instructions for the following three components:

1. Reproduce the experimental results reported in the paper.
2. Train your own 3D face authentication model and the real-time perturbation generator to prevent the face models in the surveillance video from being exploited to spoof the face authentication.
3. Set up the end-to-end security-preserving live 3D video surveillance system integrating the perturbation generator.

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
├── LiVSec_MMSys23_ReproducibilityAppendix.pdf
└── README.md

```

## 1. Environmental Setup

### Hardware requirements

A workstation with an Nvidia GPU is required to train/infer the models, and run the system.

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

5. [For System] Install `ffmpeg`.
   ```shell
   sudo apt install ffmpeg
   ```
   Please note that in this project, we assume the path to the binary file of `ffmpeg` is `/usr/bin/ffmpeg`. You can find its 
   path in your system by typing `which ffmpeg` in the terminal. If it is different from this path, please modify Line 30 of
   `main.py` under `System` folder to the `ffmpeg` path in your system.

6. [For System] Install `v4l2loopback` following the [official instructions](https://github.com/umlaeute/v4l2loopback).
   
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

Dataset #1 comes from https://vap.aau.dk/rgb-d-face-database/. Run `preprocess_data.py` under `FaceAuthentication/utils` to download and preprocess Dataset #1. 

If you want to change where to put the dataset, change Line 126 of `preprocess_data.py`.
The default is `/data/faceid`.
If you change it, you should also
change `--data_path` argument every time you call the models (either the face authentication model or the protected
face generation model).

The 3D video in Dataset #2 is exported from Depthkit (See Appendix section of this doc for more details).
We provide the exported video we use in this project, which is `System/John_3dts.mp4`. No additional actions are needed for Dataset #2 since they are all included.

### Results for Dataset #1

Run `result_collect.py` under `ProtectedFaceGeneration` folder. The output will show all the results.

### Results for Dataset #2

Run `main.py` under `System` folder. The output will show all the results in `REUSE-1` mode.

Change Line 28 of `main.py` to the designated reuse frequency, e.g., 5 or 10, which was used in the paper, to produce 
corresponding results.

Please note that to see the results, the `MODE` in Line 27 has to be set as `RESULT_COLLECT_MODE`.

## 3. [Optional] Train Your Own Models

See the 7th step of `Environmental Setup` to download the pretrained models. You can also train your own face authentication and 
protected face generation models.

You should follow `Prepare datasets` section above to prepare the datasets first.

### Train the face authentication model

Run `train.py` under `FaceAuthentication` folder. If you want to change the training hyper-parameters, check Line 41-59 of the `train.py`.

### Train the protected face generation model

You should have a pretrained face authentication model, and put it under `ProtectedFaceGeneration/modules`.
Change Line 21 of `ProtectedFaceGeneration/main_module.py` to your own face authentication model path.

Run `train.py` under `ProtectedFaceGeneration` folder. If you want to change the training hyper-parameters, check Line 43-59 of the `train.py`.

## 4. [Optional] The End-to-end Security-preserving Live 3D Video Surveillance System

The protected frames will be sent to the virtual camera `/dev/video0`. Any end-to-end live streaming protocols can be 
adopted to build the live streaming system. In this project, we use `DashCast` to generate a live DASH stream 
from it at the server end, and use `MP4Client` to watch the playback at the client end.

### Server-end configuration

The code under `System` folder runs on the server-end, so you should do the following at the server end.

1. Install `DashCast` (included in the `GPAC 0.8.1`). Find more versions about `GPAC 0.8.1`, see [this page](https://gpac.wp.imt.fr/downloads/gpac-legacy-builds/).
   ```shell
   wget https://download.tsi.telecom-paristech.fr/gpac/legacy_builds/linux64/gpac/gpac_0.8.1-latest-legacy_amd64.deb
   sudo apt install ./gpac_0.8.1-latest-legacy_amd64.deb
   ```

2. Install `Node.js`. We recommend using tools like `nvs` to easily install and manage `Node.js`.
   See [official documentation](https://github.com/jasongin/nvs) of `nvs` for more information.
   ```shell
   # install nvs
   export NVS_HOME="$HOME/.nvs"
   git clone https://github.com/jasongin/nvs "$NVS_HOME"
   . "$NVS_HOME/nvs.sh" install
   
   # add LTS version of the node
   nvs add lts
   ```

3. Download the DASH Low Latency Web Server.
   ```shell
   git clone https://github.com/gpac/node-gpac-dash
   cd node-gpac-dash && mkdir livsec
   ```
   We assume the path of the web server is `<YOUR_PATH>/node-gpac-dash`. 
   Change Line 34 of `Server/main.py` to `<YOUR_PATH>/node-gpac-dash/livsec`.

4. Start DASH streaming.
   We provide two types of DASH, the default one and the low-latency one. Start the corresponding DASH server.

   ```shell
   cd <YOUR_PATH>/node-gpac-dash
   nvs use lts  # enable node
   
   # for default DASH
   node gpac-dash.js
   
   # for low latency DASH
   node gpac-dash.js -segment-marker eods -chunk-media-segments
   ```
   
   Change the `DASH_MODE` parameter in Line 33 of `System/main.py` to switch the DASH streaming mode. 
   You also need to change Line 27 of it to `STREAMING_MODE` to disable the result collection.
   Finally, run `main.py`, and the DASH profiles/segments generation will start automatically. You can check the generated
   DASH files in `<YOUR_PATH>/node-gpac-dash/livsec/output`.

### Client-end configuration

1. Install `MP4Client` by installing `GPAC 0.8.1` following the aforementioned steps.
2. Watch the playback.
   ```shell
   # default DASH
   MP4Client http://localhost:8000/livsec/output/manifest.mpd
   
   # low latency DASH
   MP4Client http://127.0.0.1:8000/livsec/output/dashcast.mpd -opt Network:BufferLength=200 -opt DASH:LowLatency=chunk -opt DASH:UseServerUTC=no
   ```

### Known issues

For both DASH streaming modes, the streaming is not smooth. And in the low-latency mode, there is a distortion at the 
client end. We will keep working on this and provide updates in this GitHub repo.

## 5. Appendix 

### How to get Dataset #2
1. Get a Depthkit account at https://www.depthkit.tv/signup.
2. Log in and download Depthkit (as of the paper submission, the version is 0.6.1), and the Sample Project for Depthkit Core (which contains a pre-shot volumetric video) at https://www.depthkit.tv/downloads.
3. Install Depthkit and unzip the Sample Project for Depthkit Core. Use Depthkit to open the sample project.
4. Click the Editor button (it looks like a film) in the upper-left corner of the window to begin editing the sample project.
5. Select the ISOLATE tab, and change Depth Range to the range of 1.27m to 2.37m. This step filters foreground and background.
6. Change all numbers for Horizontal Crop and Vertical Crop to 0px. 
7. Use the two selectors (look like label icons) in the lower-right of the window to select the video clip with the length you desire.
8. Go to the EXPORT tab, and set the Output Width and Output Height to 2048. 
9. Click the EXPORT button, and you will get an RGBD video similar to Figure 5 in the paper.

Please note that, while conducting the experiments for this paper, we used an older version of Depthkit (around late 2019 or early 2020) to edit and export the RGBD video. Now the newest version of Depthkit cannot generate the same RGBD video clip. For example, in Step 8, when using the current version of Depthkit, the Output Width and Output Height can only be set to 1920 and 2160, respectively.


## 6. Cite Our Work

Zhongze Tang, Huy Phan, Xianglong Feng, Bo Yuan, Yao Liu, and Sheng Wei. 2023. 
Security-Preserving Live 3D Video Surveillance. 
In Proceedings of the 14th ACM Multimedia Systems Conference (MMSys ’23), June 2023. 
https://doi.org/10.1145/3587819.3590975

## 7. Contact

If you have any questions or any idea to discuss, you can email Zhongze Tang (zhongze.tang@rutgers.edu) directly.

## 8. License

MIT