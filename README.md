# Fat-to-Thin Policy Optimization

This repo contains the code for our paper [Fat-to-Thin Policy Optimization: Offline RL with Sparse Policies](https://openreview.net/pdf?id=SRjzerUpB2) accepted by ICLR 2025. 


## Run code

We include all the baselines tested in the paper. See `run_ac_offline.py` for available options. To see how Fat-to-thin performs, simply run
```python
python run_ac_offline.py --agent FTT
```

## D4RL installation
If you are using *Ubuntu* and have not installed *d4rl* yet, this section may help

1. Download mujoco

	I am using mujoco210. It can be downloaded from https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
   ```
   mkdir .mujoco
   mv mujoco210-linux-x86_64.tar.gz .mujoco
   cd .mujoco
   tar -xvzf mujoco210-linux-x86_64.tar.gz
   ```

    Then, add mujoco path:

	Open .bashrc file and add the following line:
	```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<Your_path>/.mujoco/mujoco210/bin
    ```

	Save the change and run the following command:
	```
    source .bashrc
    ```
	
2. Install other packages and D4RL
    ```
    pip install mujoco_py
    pip install dm_control==1.0.7
    pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
    ```
	
3. Test the installation in python
    ```
       import gym
       import d4rl
       env = gym.make('maze2d-umaze-v1')
       env.get_dataset()	   
   ```

## Citing
If you find our paper helpful, please consider citing it by 
```
@inproceedings{Zhu2025-FatToThin,
title={Fat-to-Thin Policy Optimization: Offline RL with Sparse Policies},
author={Lingwei Zhu and Han Wang and Yukie Nagai},
booktitle={International Conference on Learning Representations (ICLR)},
year={2025},
}
```
