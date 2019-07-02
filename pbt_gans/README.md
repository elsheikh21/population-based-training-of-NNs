# pbt-gan

PBT is an optimization algorithm that maximizes the performance of a network by optimizating a population of models and their hyperparameters. It determines a schedule of hyperparameter settings using an evolutionary strategy of exploration and exploitation - a much more powerful method than simply using a fixed set of hyperparameters throughout the entire training or using grid-search and hand-tuning, which is time-extensive and difficult.

Implementation of PBT-GAN experiments from [paper](https://arxiv.org/pdf/1711.09846.pdf).  
(Refer [here](https://github.com/angusfung/population-based-training) for Toy Experiments from paper)

![alt-text-1](https://github.com/angusfung/pbt-gan/blob/master/images/18/samples_47099.jpg "title-1") ![alt-text-2](https://github.com/angusfung/pbt-gan/blob/master/images/0/samples_47799.jpg "title-2") ![alt-text-1](https://github.com/angusfung/pbt-gan/blob/master/images/13/samples_47399.jpg "title-3")
 

## Setup
It is recommended to run from a virtual environment to ensure all dependencies are met.  
Compatible with both python 2 and 3.
```
virtualenv -p python pbt_env
source pbt_env/bin/activate.csh
pip install -r requirements.txt
```
## Memory Utilization
Memory limits can be set per-worker (as a percentage) by uncommenting `gpu_options` in `pbt_main.py` which can be desired for synchronous training.
 
## Training
### Asynchronous Training
`python pbt_main.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224,localhost:2225,localhost:2226 --job_name=ps --task_index=0`
`python pbt_main.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224,localhost:2225,localhost:2226 --job_name=worker --task_index=0`  
...
  
### Synchronous Training
`python pbt_sequential.py`

## Results
Results for synchronous training with `20` workers.
  
### Inception Plots

Left is PBT with 20 workers. Right compares PBT with no-PBT (grey color).  
Note that PBT significantly outperforms the baseline (no PBT) by about a 0.5 inception score.

![alt-text-1](https://github.com/angusfung/pbt-gan/blob/master/results/inception1.PNG "title-1") ![alt-text-2](https://github.com/angusfung/pbt-gan/blob/master/results/inception3.PNG "title-2") 

Left (smoothed plot). Right (raw plot).   
Blue (with PBT), Grey (no PBT)

![alt-text-3](https://github.com/angusfung/pbt-gan/blob/master/results/inception2.PNG "title-3") ![alt-text-4](https://github.com/angusfung/pbt-gan/blob/master/results/inception_nosmooth.PNG "title-4")

### Learning Rates

Learning schedules discovered by PBT

![alt-text-3](https://github.com/angusfung/pbt-gan/blob/master/results/d_learning.PNG "title-3") ![alt-text-4](https://github.com/angusfung/pbt-gan/blob/master/results/g_learning.PNG "title-4")

## Saved Sessions
The code will automatically restore from a previous save-point under `./checkpoint` if exists. Tensorboard files are stored under `./logs`. Images are stored under `./images`. Pretrained model / checkpoint for 1 worker is provided under `./checkpoint`. Unfortunately due to space limitations, tensorboard logs could not be uploaded.


## Credits
GAN templates from [here](https://github.com/hwalsuklee/tensorflow-generative-model-collections) and [here](https://github.com/igul222/improved_wgan_training)

