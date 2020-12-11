# DuckieNet
Code accompanying the report for *CS2309: Research Methodology*

> [DuckieNet: Integrating Planning with Semantic Segmentation for Goal-Directed Autonomous Navigation in Crowded Environments](https://www.overleaf.com/read/zhmjvbsvnbgf), by Chen Yuan Bo and Larry Law

## Requirements
Duckie-Net requires Keras v2.2.4, TensorFlow 1.15, Python 3.6, Ros Kinetic.

## Installation

Install the prequisite packages via

```
conda env create -f environment.yaml
conda activate duckienet
conda env list // verify if environment was installed correctly
```

> **Important**: Remember to swap out the `path_to_repo` with your own repository path! Do a find and swap of all occurrences in the whole repository.

## Running Experiments
1. Straight Road Experiments on DuckieNet
`./gym-duckietown/duckienet.py --seg --map-name straight_road`
2. Straight Road Dynamic Experiments on DuckieNet
`./gym-duckietown/duckienet.py --seg --map-name straight_road_moving`
3. DuckieTown Experiment on DuckieNet
`./gym-duckietown/duckienet.py --seg --map-name udem1`

> You can vary the difficulty by choosing starting tile in `./gym-duckietown/gym_duckietown/maps/udem1.yaml`.

> To test the experiments on I-Net, simply use the I-Net model instead of the DuckieNet model, and remove the `--seg` flag.


## Data Preparation
The data for the semantic segmentation model comprise of input variables RGB images and output variables segmentation labels.

1. Use `generate_valid_positions.ipynb` inside `./gym-duckietown` folder to generate valid spawn positions within DuckieTown.
2. Generate segmentation labels:
    a. Swap out the `textures/` and `meshes/` folders inside `./gym-duckietown/gym_duckietown/` with the `textures/` and `meshes/` folders inside `./segmentation_textures/`.
    b. Use `generate_segmented_images.ipynb` to generate the corresponding segmentation labels.
3. Generate RGB images:
    a. Swap out the `textures/` and `meshes/` folders to the original folders.
    b. Use `generate_segmented_images.ipynb` to generate the corresponding RGB images.

The data for DuckieNet is prepared by following the steps in section 4.1.1 of the report with the command

`./gym-duckietown/generate_data.py --map-name udem1 --save --counter-start 0`

> Data is saved in `./gym-duckietown/data`


## Training
The training of DuckieNet comprise of two phases.

1. Training of BiSeNet. For more information, refer to the README file in the `bisenet` folder.
```
$ cd bisenet
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train_amp.py --model bisenetv2 # or bisenetv1
```
2. Training of DuckieNet
```
$ cd intention_net/intention_net
$ python main.py --ds DUCKIETOWN --mode DLM --input_frame NORMAL --seg
```

> To train Intention-Net (which does not use segmented images), remove the `--seg` flag.

## Testing
Simply run the commands in 'Running Experiments' section, or run the model on any of DuckieTown maps in`./gym-duckietown/gym_duckietown/maps/`.

> Metrics (as stated in section 4.1 of the report) will be logged in `log.txt`.

## Citation
We used materials from the following code repositories.

Wei Gao, David Hsu, Wee Sun Lee, Shengmei Shen, & Karthikk Subramanian. (2017). Intention-Net: Integrating Planning and Deep Learning for Goal-Directed Autonomous Navigation. [https://github.com/AdaCompNUS/intention_net/tree/master/intention_net]

Atsushi Sakai, Daniel Ingram, Joseph Dinius, Karan Chawla, Antonin Raffin, & Alexis Paques. (2018). PythonRobotics: a Python code collection of robotics algorithms. [https://github.com/AtsushiSakai/PythonRobotics]

Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, & Nong Sang. (2018). BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation. [https://github.com/CoinCheung/BiSeNet]

L. Paull, J. Tani, H. Ahn, J. Alonso-Mora, L. Carlone, M. Cap, Y. F. Chen, C. Choi, J. Dusek, Y. Fang, D. Hoehener, S. Liu, M. Novitzky, I. F. Okuyama, J. Pazis, G. Rosman, V. Varricchio, H. Wang, D. Yershov, H. Zhao, M. Benjamin, C. Carr, M. Zuber, S. Karaman, E. Frazzoli, D. Del Vecchio, D. Rus, J. How, J. Leonard, & A. Censi (2017). Duckietown: An open, inexpensive and flexible platform for autonomy education and research. In _2017 IEEE International Conference on Robotics and Automation (ICRA)_ (pp. 1497-1504). [https://github.com/duckietown/gym-duckietown]


