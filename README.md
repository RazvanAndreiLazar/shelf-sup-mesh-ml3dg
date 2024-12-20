# Semester project for ML3DG (Machine Learning for 3D Geometry) @TUM
Our contribution can be seen [here](https://github.com/RazvanAndreiLazar/shelf-sup-mesh-ml3dg/blob/master/ML3D_Report.pdf).

# Shelf-supervised Mesh Prediction in the wild 
in CVPR 2021, Yufei Ye, Shubham Tulsiani, Abhinav Gupta

[Project Page](https://judyye.github.io/ShSMesh/), [Video](https://youtu.be/OAiFEAuzPZk), [Arxiv](https://arxiv.org/abs/2102.06195) 

![](https://judyye.github.io/ShSMesh/data/teaser.gif) 

We aim to infer 3D shape and pose from a single image and are able to train the system with only image collecitons and 
segmentation -- no template, camera pose, or multi-view association. The method consists of 2 steps:  

1. **Category-level Reconstruction.** We first infer a volumetric representation in a canonical
frame, along with the camera pose for the input image. 
2. **Instance-level Specialization.** The coarse volumetric prediction is converted to a mesh-based representation, which is further optimized in the predicted camera frame given the input image.

This code repo is a re-implementation of the paper. The code is developed based on [Pytorch 1.3](https://pytorch.org/) 
(Pytorch >=1.5 adds backprop version check which will trigger a [runtime error](https://discuss.pytorch.org/t/solved-pytorch1-5-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/90256)), 
[Pytorch3d 0.2.0](https://github.com/facebookresearch/pytorch3d/tree/master/pytorch3d),
and integrated [LPIPS](https://github.com/richzhang/PerceptualSimilarity).
To voxelize meshes for evaluation, we use util code in [Occupancy Net](https://github.com/autonomousvision/occupancy_networks/tree/master/im2mesh) but did not include it in this reimplementation.  


## Demo: Estimate mesh with our pretrained model
Download pretrained models to `weights/`

| dataset | model |
| --- | --- |
|OpenImages-50 | [tar link](https://drive.google.com/file/d/1BOIcPRXdpKPNi6G3pjDSkyaKGCf5kMDm/view?usp=sharing) |
|Chairs in the wild | [link](https://drive.google.com/file/d/1cZVOB7doSC2-DyqkzUSLoSkvKrYPdZOq/view?usp=sharing) |
|Quadrupeds | [link](https://drive.google.com/file/d/1IpQMvZnProHcENIa-GC_IEaxY9e1zvH6/view?usp=sharing) |
|CUB-200-2011 | [link](https://drive.google.com/file/d/1Y-jf-CxhVX83FDDsT4hA6UG44xFQTFqG/view?usp=sharing) |
 
```
python demo.py  --checkpoint=weights/wildchair.pth
```
Similar results should be saved at `outputs/`

|input | output shape | output shape w/ texture | 
|---| ---| --- |
|![](examples/allChair_0.png) | ![](examples/demo_out/allChair_0_mesh_az.gif) | ![](examples/demo_out/allChair_0_meshTexture_az.gif)|

or for other curated categories:

```
python demo.py  --checkpoint=weights/cub.pth --demo_image examples/cub_0.png
python demo.py  --checkpoint=weights/wildchair.pth --demo_image examples/wildchair_0.png
python demo.py  --checkpoint=weights/quad.pth --demo_image examples/llama.png
```

for openimages 50 categories, the following script will reconstruct images under `data/demo_images/`:
```
python demo_all_cls.py 
```


## Training
To train your own model, set up dataset following [`dataset.md`](docs/dataset.md) before running 
```
python train_test.py     --dataset allChair --cfg_file config/pmBigChair.json 
```

For more training details, please refer to [`train.md`](docs/train.md)


## Citation
If you find this work useful, please consider citing:
```
@inProceedings{ye2021shelf,
  title={Shelf-Supervised Mesh Prediction in the Wild},
  author={Ye, Yufei and Tulsiani, Shubham and  Gupta, Abhinav},
  year={2021},
  booktitle={Computer Vision and Pattern Recognition (CVPR)}
}
```
