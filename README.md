# Minimal TimeLens Implementation
## Gustavo Silvera
### [CMU Computational Photography, Fall 2021](http://graphics.cs.cmu.edu/courses/15-463/)

This repository contains the code and reports for the final project for my [15-463 computational photography course at CMU](http://graphics.cs.cmu.edu/courses/15-463/final_project.html). For this project I decided to make a minimal implementation of [TimeLens: Event-based Video Frame Interpolation](https://github.com/uzh-rpg/rpg_timelens) and capture some of my own results.

For imaging hardware, I was provided with a [Nikon D3500 DSLR](https://www.nikonusa.com/en/nikon-products/product/dslr-cameras/d3500.html) for video (1920x1080@60hz) and a [DVXplorer Lite (Academic)](https://shop.inivation.com/products/dvxplorer-lite-academic-rate) (320x240@5000hz) for events. Guidance and hardware for project was provided by the course professor [Dr. Ioannis Gkioulekas](https://www.cs.cmu.edu/~igkioule/).

## Project Proposal
The proposal (pdf) can be found [here](writeup/proposal.pdf)

## Project Final Report
The final report (pdf) can be found [here](writeup/final-report.pdf)

## Examples
| 15hz                                   | 30hz                                   |
| -------------------------------------- | -------------------------------------- |
| ![totoro_15](assets/totoro_15.gif)     | ![totoro_30](assets/totoro_30.gif)     |
| ![totoro_1_15](assets/totoro_1_15.gif) | ![totoro_1_30](assets/totoro_1_30.gif) |
| ![tennis_15](assets/tennis_15.gif)     | ![tennis_30](assets/tennis_30.gif)     |

## Installation
Note that the primary dependencies required for this implementation are [`numpy`](https://numpy.org/) and [`pytorch`](https://pytorch.org/get-started/locally/).

```bash
pip install pytorch-cuda torchvision cudatoolkit
pip install dv # for the DVXplorer camera software
conda install numpy matplotlib scikit-image
# or optionally, use the exact same conda config I use
conda env create -f environment.yml 
```

```bash
sudo apt install ffmpeg # for video -> frame conversion
sudo apt install dv-gui # for the DVXplorer camera ui
```

## Hardware Setup
First off, to initialize your own data capture setup you'll need a video and event camera in such a way to synchronize the two cameras in both space and time. 
- To synchronize the output streams in space, the following approaches have been tested:
  - Place the two cameras close to each other with a small baseline, then transform/crop the outputs in post to match. 
  - Utilize a beam-splitter to have both cameras see the same scene, then perform necessary transformations in post. 
- To synchronize the output streams in time, you should create some kind of very-noticeable events that both cameras would pick up
  - A good example that I use is to toggle the lights in the room at the start of capturing data so both cameras pick up this event. 
    - Find these points in post and crop the data so events are synchronized
To further understand how I set up my camera setup, you can read my [final report here](writeup/final-report.pdf)

## Usage
Assuming you have valid output streams from both cameras I'm using (not necessarily a particular video camera) I have provided a script to convert the raw data streams into usable data for the interpolation pipeline:

1. Go to the `data/` directory and `mkdir` a new directory for the video just recorded, for ex: `mkdir hand-wave`
2. Move the `.aedat4` file provided from the `dv-gui` capture of the DVXplorer camera to `data/hand-wave` and rename it to `events.aedat4`
3. Move the `.MOV` video captured from the video camera to a new directory `data/hand-wave/images`
   1. Convert the `.MOV` to still RGB frames via `ffmpeg -i {FILE}.MOV %005d.jpg` 
4. (For the time synchrony step described in **Hardware Setup**)
   1. Find the first and last video frame where the camera goes dark, change these `"lights_out"` parameters in `critical_events` of [`data_process.py`](src/data_process.py)
   2. For all scenes you want to capture in the recording, find the start and end of these events and add them to `critical_events` as a tuple of frame indices (start, end). 
5. Variable setup:
   1. Note that if you used a beam-splitter, you'll need to flip the x coordinates of the event camera. This can be done by editing the `used_beam_splitter` variable in `data_process.py`. 
   2. You should update the `dvXres` for the resolution of your event camera, and `rgbres` to the resolution of the video camera
   3. You should update the `fps` variable to the framerate capture of the video capture. 
   4. You should update the `temporal_res` variable for the remporal resolution of the event camera, the DVXplorer has microsecond-level granularity
   5. You should update the `images_set` variable for the name of the dataset you want to use. 
6. Run the `src/data_process.py` file and visualize the events and video overlaid to see what kinds of transformations would make the overlay fit better. 

Finally, with all the data preprocessing done, you should see a valid directoy in the `data` directory that looks like
```bash
.
├── events
│   └── events.npz
├── images
│   ├── 000350.png
│   ├── 000351.png
│   ├── ...
│   ├── 000353.png
│   ├── 000399.png
│   └── timestamp.txt
├── overlay
│   ├── 00000.jpg
│   ├── 00001.jpg
│   ├── ...
│   ├── 00023.jpg
│   └── 00024.jpg
└── side_by_side
    ├── 00000.jpg
    ├── 00001.jpg
    ├── ...
    ├── 00030.jpg
    ├── 00031.jpg
    ├── movie.gif
    └── movie.mp4
```

Then you can finally run 
```bash
python src/main.py
```
after editing the `image_set` variable in [`src/main.py`](src/main.py). Also feel free to edit the number of inserted frames (`num_inserts`) to change how deep the interpolation should work. 

Once your frames have been generated in `out_dir` (see [`src/main.py`](src/main.py)) the recommended way to convert the frames to a movie without loss in quality is 
```bash
# in out dir (lots of .jpg's)
# to a 15fps video, change -framerate for something else
ffmpeg -framerate 15 -i %04d.jpg -codec copy movie.mp4 
```

## Acknowledgements
This project is based on the work done in [TimeLens: Event-based Video Frame Interpolation](http://rpg.ifi.uzh.ch/TimeLens.html) ([github](https://github.com/uzh-rpg/rpg_timelens)), from the Huawei Technologies Zurich Research Center and Department of Informatics and Neuroinformatics at University of Zurich and ETH Zurich.