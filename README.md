<div align="center">
  <img src="./assets/logo.png" width="30%">
</div>
<h1 align="center">💃DreamWaltz: Make a Scene with Complex 3D Animatable Avatars</h1>
<p align="center">

### [Project Page](https://dreamwaltz3d.github.io/) | [Paper](https://arxiv.org/abs/2305.12529) | [Video]()

This repository contains the official implementation of the following paper:
> DreamWaltz: Make a Scene with Complex 3D Animatable Avatars
<br>[Yukun Huang](https://github.com/hyk1996/)<sup>*1,2</sup>, [Jianan Wang](https://github.com/wendyjnwang/)<sup>*1</sup>, [Ailing Zeng](https://ailingzeng.site/)<sup>1</sup>, [He Cao](https://github.com/CiaoHe/)<sup>1</sup>, [Xianbiao Qi](http://scholar.google.com/citations?user=odjSydQAAAAJ&hl=zh-CN/)<sup>1</sup>, Yukai Shi<sup>1</sup>, Zheng-Jun Zha<sup>2</sup>, [Lei Zhang](https://www.leizhang.org/)<sup>1</sup><br>
> <sup>∗</sup> Equal contribution. <sup>1</sup>International Digital Economy Academy &nbsp; <sup>2</sup>University of Science and Technology of China

Code will be released soon.

## Introduction
DreamWaltz is a learning framework for text-driven 3D animatable avatar creation using pretrained 2D diffusion model [ControlNet](https://github.com/lllyasviel/ControlNet) and human parametric model [SMPL](https://smpl.is.tue.mpg.de/). The core idea is to optimize a deformable NeRF representation from skeleton-conditioned diffusion supervisions, which ensures 3D consistency and generalization to arbitrary poses.

<p align="middle">
<img src="assets/teaser.gif" width="80%">
<br>
<em>Figure 1. DreamWaltz can generate animatable avatars (a) and construct complex scenes (b)(c)(d).</em>
</p>

## Results

### Static Avatars

<p align="middle">
<image src="assets/canonical_half.gif" width="80%">
<br>
<em>Figure 2. DreamWaltz can create canonical avatars from textual descriptions.</em>
</p>

### Animatable Avatars

<p align="middle">
<image src="assets/animation_sp.gif" width="80%">
<br>
<em>Figure 3. DreamWaltz can animate canonical avatars given motion sequences.</em>
</p>
  
### Complex Scenes

<p align="middle">
<image src="assets/animation_obj.gif" width="80%">
<br>
<em>Figure 4. DreamWaltz can make complex 3D scenes with avatar-object interactions.</em>
</p>

<p align="middle">
<image src="assets/animation_scene.gif" width="80%">
<br>
<em>Figure 5. DreamWaltz can make complex 3D scenes with avatar-scene interactions.</em>
</p>

<p align="middle">
<image src="assets/animation_mp.gif" width="80%">
<br>
<em>Figure 6. DreamWaltz can make complex 3D scenes with avatar-avatar interactions.</em>
</p>

## Reference
If you find this repository useful for your work, please consider citing it as follows:
```bibtex
@article{huang2023dreamwaltz,
 title={DreamWaltz: Make a Scene with Complex 3D Animatable Avatars},
 author={Yukun Huang and Jianan Wang and Ailing Zeng and He Cao and Xianbiao Qi and Yukai Shi and Zheng-Jun Zha and Lei Zhang},
 year={2023},
 eprint={2305.12529},
 archivePrefix={arXiv},
 primaryClass={cs.CV}
}

@article{huang2023dreamtime,
 title={DreamTime: An Improved Optimization Strategy for Text-to-3D Content Creation},
 author={Yukun Huang and Jianan Wang and Yukai Shi and Xianbiao Qi and Zheng-Jun Zha and Lei Zhang},
 year={2023},
 eprint={2306.12422},
 archivePrefix={arXiv},
 primaryClass={cs.CV}
}
