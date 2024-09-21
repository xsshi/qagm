### QAGM-ICTAI 2024

This project is based on the famous graph matching open-source projects [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) [![GitHub stars](https://camo.githubusercontent.com/e92c9d3155564f041f370bec74d83f7178087db8def34b071ae45c233293b546/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f5468696e6b6c61622d534a54552f5468696e6b4d617463682e7376673f7374796c653d736f6369616c266c6162656c3d53746172266d61784167653d38363430)](https://github.com/Thinklab-SJTU/ThinkMatch/). Our work proposed a novel conclusion of Quadratic Assignment Contrastive Loss for generalized utilization. And the graph matching problem is a proper platform to test the performance of our method. 



## Get Started

### Docker (RECOMMENDED)

Some of the module needs C++ supporting and we highly encouraged to directly use the docker environment. Get the recommended docker image by

```bash
docker pull runzhongwang/thinkmatch:torch1.10.0-cuda11.3-cudnn8-pyg2.0.3-pygmtools0.3.8
docker run --gpus all --name thinkmatch -p 10000:22 -it runzhongwang/thinkmatch:torch1.10.0-cuda11.3-cudnn8-pyg2.0.3-pygmtools0.3.8
pip install ortools==9.4.1874
```

Note we train our model on a single 3090 GPU. The training time is about 9 hours for Pascal VOC and 4 hours for Spair71k.


### Manual configuration (for Ubuntu, NOT RECOMMENDED)

The below python environment is provided by [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) and we do not guarantee the integrity.

1. Install and configure Pytorch 1.6 (with GPU support). 

1. Install ninja-build: ``apt-get install ninja-build``

1. Install python packages: 

   ```bash
   pip install tensorboardX scipy easydict pyyaml xlrd xlwt pynvml pygmtools
   ```

1. Install building tools for LPMP: 

   ```bash
   apt-get install -y findutils libhdf5-serial-dev git wget libssl-dev
   
   wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz && tar zxvf cmake-3.19.1.tar.gz
   cd cmake-3.19.1 && ./bootstrap && make && make install
   ```

1. Install and build LPMP:

   ```bash
   python -m pip install git+https://git@github.com/rogerwwww/lpmp.git
   ```

   You may need ``gcc-9`` to successfully build LPMP. Here we provide an example installing and configuring ``gcc-9``: 

   ```bash
   apt-get update
   apt-get install -y software-properties-common
   add-apt-repository ppa:ubuntu-toolchain-r/test
   
   apt-get install -y gcc-9 g++-9
   update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
   ```

1. Install torch-geometric:

   ```bash
   export CUDA=cu101
   export TORCH=1.6.0
   /opt/conda/bin/pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   /opt/conda/bin/pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   /opt/conda/bin/pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   /opt/conda/bin/pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
   /opt/conda/bin/pip install torch-geometric==1.6.3
   ```

1. If you have configured ``gcc-9`` to build LPMP, be sure to switch back to ``gcc-7`` because this code repository is based on ``gcc-7``. Here is also an example:

   ```bash
   update-alternatives --remove gcc /usr/bin/gcc-9
   update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
   ```

### Available datasets

Note: All following datasets can be automatically downloaded and unzipped by `pygmtools` in this code, but we recommend downloading the dataset yourself as it is much faster.

1. PascalVOC-Keypoint

   1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/TrainVal/VOCdevkit/VOC2011``

   1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like ``data/PascalVOC/annotations``

   1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``. **This file must be added manually.**

   Please cite the following papers if you use PascalVOC-Keypoint dataset:

   ```
   @article{EveringhamIJCV10,
     title={The pascal visual object classes (voc) challenge},
     author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
     journal={International Journal of Computer Vision},
     volume={88},
     pages={303–338},
     year={2010}
   }
   
   @inproceedings{BourdevICCV09,
     title={Poselets: Body part detectors trained using 3d human pose annotations},
     author={Bourdev, L. and Malik, J.},
     booktitle={International Conference on Computer Vision},
     pages={1365--1372},
     year={2009},
     organization={IEEE}
   }
   ```

1. Willow-Object-Class

   1. Download [Willow-ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip)

   1. Unzip the dataset and make sure it looks like ``data/WillowObject/WILLOW-ObjectClass``

   Please cite the following paper if you use Willow-Object-Class dataset:

   ```
   @inproceedings{ChoICCV13,
     author={Cho, Minsu and Alahari, Karteek and Ponce, Jean},
     title = {Learning Graphs to Match},
     booktitle = {International Conference on Computer Vision},
     pages={25--32},
     year={2013}
   }
   ```

1. SPair-71k

   1. Download [SPair-71k dataset](http://cvlab.postech.ac.kr/research/SPair-71k/)

   1. Unzip the dataset and make sure it looks like ``data/SPair-71k``

   Please cite the following papers if you use SPair-71k dataset:

   ```
   @article{min2019spair,
      title={SPair-71k: A Large-scale Benchmark for Semantic Correspondence},
      author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
      journal={arXiv prepreint arXiv:1908.10543},
      year={2019}
   }
   
   @InProceedings{min2019hyperpixel, 
      title={Hyperpixel Flow: Semantic Correspondence with Multi-layer Neural Features},
      author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
      booktitle={ICCV},
      year={2019}
   }
   ```

   For more information, please see [pygmtools](https://pypi.org/project/pygmtools/).

## Run the Experiment


Run training and evaluation

```bash
python train_eval.py --cfg path/to/your/yaml
```

and replace ``path/to/your/yaml`` by path to your configuration file, e.g.

```bash
python train_eval.py --cfg experiments/vgg16_qagm_willow.yaml
```

Default configuration files are stored in``experiments/`` and you are welcomed to try your own configurations.

### File Organization

```
├── experiments
│   the hyperparameter setting of experiments
├── models
│     └── QAGM
│         the module and training pipeline of COMMON
│          ├── model.py
│          │   the implementation of training/evaluation procedures of QAGM
│          ├── model_config.py
│          │   the declaration of model hyperparameters
│          └── sconv_archs.py
│              the implementation of spline convolution (SpilneCNN) operations, the same with BBGM
├── src
│  the source code of the Graph Matching, from ThinkMatch
│      └── loss_func.py
│          the implementation of loss functions 
├── eval.py
|   evlaution script
└── train_eval.py
    training script
```




## Pretrained Models

We provides pretrained models. The model weights are available via [google drive](https://drive.google.com/drive/folders/1OdpCanr_aO5GxfC3gUXFqWXk8cZBM-nU?usp=share_link)

To use the pretrained models, firstly download the weight files, then add the following line to your yaml file:

```yaml
PRETRAINED_PATH: path/to/your/pretrained/weights
```

