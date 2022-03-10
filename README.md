# NICE-SLAM-with-Adaptive-Feature-Grids-
## Deadlines

- [x] **March 07: Group formation and project selection** - Students select from a list of project proposals and we assign them to the topics.
- [ ] **March 11: Project proposal documents** - Students submit their project proposal documents after discussing with their assigned supervisors.
- [ ] **March 14: Proposal presentations** - Students present their project proposals during lecture.
- [ ] **April 25: Midterm presentations** - Students present their progress on their projects during lecture.
- [ ] **May 30: Final project presentations** - Students present their projects in a joint session.
- [ ] **June 13: Final project reports** - Students submit their final reports for the projects.

## Milestones

- [ ] Correctly allocate sparse feature voxels based on ground truth camera pose and depth image. Avoid using any dense arrays for storage (No need to use NICE-SLAM) => don’t need the source code
- [ ] Integrate into NICE-SLAM, features can be updated correctly, able to work on e.g. Replica dataset
- [ ] Test on kitti dataset, with whole pipeline (tracking & mapping) working correctly

## Links

### [NICE-SLAM: Neural Implicit Scalable Encoding for SLAM](https://pengsongyou.github.io/media/nice-slam/NICE-SLAM.pdf)

* **Abstract & Intro**: 
  * SLAM: simultaneous localization and mapping
  * Requirement
    * Real-time computation
    * Predictive: Can make prediction for regions without observation
    * Scalable: Can be scaled up to large scenes
    * Robust to noise
  * Limitation of current methods: 
    * Over smoothed scene reconstruction, difficult to scale up to large scenes
    * Does not incorporate location information in the observations
  * Idea: 
    * Use multi-level location information (hierarchical scene representation)
    * Incorporate inductive biases of neural implicit decoders pretrained at different spatial resolutions. 
    * Minimizing re-rendering losses
* **Related work**:
  * World-centric map representation, voxel grids => more accurate geometry at lower grid resolutions
  * iMAP

### [DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors](https://arxiv.org/pdf/2012.05551.pdf)



### [Real-time 3D Reconstruction at Scale using Voxel Hashing](https://niessnerlab.org/papers/2013/4hashing/niessner2013hashing.pdf)

### [DI-Fusion code](https://github.com/huangjh-pub/di-fusion)



