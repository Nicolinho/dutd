# Dynamic Update-to-Data Ratio: Minimizing World Model Overfitting

Official source code for the ICLR 2023 paper [Dynamic Update-to-Data Ratio: Minimizing World Model Overfitting][website] (DUTD).

DUTD is a general method that can be applied to many model-based reinforcement learning algorithm. 
We used DreamerV2 as underlying base algorithm and hence this code base is built
on top of [DreamerV2](https://github.com/danijar/dreamerv2).
A high-level visual diagram of DUTD can be seen below. 



<p align="left">
<img width="30%" src="https://i.imgur.com/7GuHNXm.png">

[comment]: <> (<a href="url"><img src="https://i.imgur.com/7GuHNXm.png" align="left" height="600" width="600" ></a>)
</p>

If you find our work useful, please reference in your paper:

```
@inproceedings{
dorka2023dynamic,
title={Dynamic Update-to-Data Ratio: Minimizing World Model Overfitting},
author={Nicolai Dorka and Tim Welschehold and Wolfram Burgard},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=ZIkHSXzd9O7}
}
```

[website]: https://openreview.net/forum?id=ZIkHSXzd9O7


## Instructions

Get dependencies:

```sh
pip install tensorflow==2.3.1
pip install tensorflow_probability==0.11.1
pip install pandas
pip install matplotlib
pip install ruamel.yaml
pip install 'gym[atari]'
pip install dm_control
```

Train the agent:

Atari100k

```sh
python3 dreamer.py --logdir ~/logdir/atari100k/atari_pong/1 \
    --configs defaults atari atari100k --task atari_pong
```

DM Control Suite

```sh
python3 dreamer.py --logdir ~/logdir/dmc/dmc_cheetah_run/1 \
    --configs defaults dmc --task dmc_cheetah_run
```


