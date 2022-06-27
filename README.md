# 3D Vision 2022 Project: NICE-SLAM with Adaptive Feature Grids

Team 18: **[Anqi Li](https://github.com/AngieALAL), [Deheng Zhang](https://github.com/dehezhang2), [Feichi Lu](https://github.com/Caroline171840094), [Ganlin Zhang](https://github.com/zhangganlin)**

## Milestones

- [x] **March 07:** Group formation and project selection
- [x] **March 13**: Proposal
- [x] **March 14:** Proposal presentations
- [x] **March 14 - March 21:** Literature Reviewing
- [x] **March 14 - April 25:** Correctly allocate sparse feature voxels based on ground truth camera pose and depth image. Avoid using any dense arrays for storage (No need to use NICE-SLAM) => don’t need the source code
- [x] **April 25:** Midterm presentations
- [x] **April 25 - May 16:** Integrate into NICE-SLAM, features can be updated correctly
- [x] **May 16 - May 25:** Solve interpolation and sample points problem
- [x] **May 25 - May 30:** Do experiments on different datasets.
- [x] **May 30:** Final project presentations
- [x] **May 30 - June 13:** Final project reports

## Overview

Our project is mainly about reducing the memory usage of nice-slam. Our main contributions includes:

1. Reducing the memory usage by smartly sample the points for rendering.

   <img src="assets/Screen Recording 2022-06-13 at 1.49.43 PM-5121733.gif" alt="Screen Recording 2022-06-13 at 1.49.43 PM" style="zoom:20%;" />

2. Design and implement sparse version of map interpolation.

   <img src="assets/Screen Recording 2022-06-13 at 1.49.43 PM-5121895.gif" alt="Screen Recording 2022-06-13 at 1.49.43 PM" style="zoom:20%;" />

3. Design and implement the sparse feature representation (voxel hashing) and integrate it into the nice-slam pipeline.

   <img src="assets/Screen Recording 2022-06-13 at 1.49.43 PM.gif" alt="Screen Recording 2022-06-13 at 1.49.43 PM" style="zoom:20%;" />

   

## Installation

First you have to make sure that you have all dependencies in place. The simplest way to do so, is to use [Anaconda](https://www.anaconda.com/).

You can create an anaconda environment called `nice-slam`. For linux, you need to install **libopenexr-dev** before creating the environment.

```
sudo apt-get install libopenexr-dev
conda env create -f spare_nice_slam.yaml
conda activate nice-slam
```

Or you can use 

```
sudo apt-get install libopenexr-dev
conda env create -f ./nice-slam-1.0-alpha/environment.yaml
conda activate nice-slam
pip install einops
pip install torch torchvision
```

## Demo

Firstly, go to the `nice-slam-1.0-alpha` folder and run following command:

```
python -W ignore run.py configs/Demo/demo.yaml
```

Then, run the following command to visualize.

```
python visualizer.py configs/Demo/demo.yaml 
```

**NOTE:** This is for demonstration only, its configuration/performance may be different from our paper.

Alternatively, you can use [MeshLab](https://www.meshlab.net/) to visualize the mesh  `output/Demo/mesh/final_mesh.ply` . 

<img src="assets/Screen Recording 2022-06-13 at 1.35.15 PM.gif" alt="Screen Recording 2022-06-13 at 1.35.15 PM" style="zoom:30%;" />

## Evaluation

### Reconstruction Error

To evaluate the reconstruction error, first download the ground truth Replica meshes where unseen region have been culled.

```
bash scripts/download_cull_replica_mesh.sh
```

Then run the command below. The 2D metric requires rendering of 1000 depth images, which will take some time. Use `-2d` to enable 2D metric. Use `-3d` to enable 3D metric.

```
# assign any output_folder and gt mesh you like, here is just an example
python src/tools/eval_recon.py --rec_mesh output/Replica/room0/mesh/final_mesh_eval_rec.ply --gt_mesh cull_replica_mesh/room0.ply -2d -3d
```

## Acknowledgement

- We express gratitude to [NICE-SLAM](https://pengsongyou.github.io/nice-slam), we benefit a lot from both their papers and codes.
- Thanks to [Songyou Peng](https://pengsongyou.github.io/). He has provided many insightful guidance about how to integrate voxel hashing into the nice-slam framework. We would like to express our sincere appreciation to [Zihan Zhu](https://zzh2000.github.io/), who has provided many intelligent suggestions on this project. 
