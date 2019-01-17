# PILCO - Probabilistic Inference for Learning COntrol

This is a re-implementation of the [PILCO algorithm](http://mlg.eng.cam.ac.uk/pilco/) (originally written in MATLAB) in Python using Tensorflow and GPflow. 
This work was mainly carried out for personal development and some of the implementation is based on this [Python implementation](https://github.com/nrontsis/PILCO).
This repository will mainly serve as a baseline for my future research.

I implemented the cart pole benchmark using [MuJoCo](http://www.mujoco.org/) and [OpenAI](https://gym.openai.com/). 
I did this because OpenAI's CartPole environment does not have a continuous action space and because the [InvertedPendulum-v2 environment](https://gym.openai.com/envs/InvertedPendulum-v2/) uses an "inverted" cart pole.
The new environment represents the traditional cart pole benchmark with a continuous action space.

The [env/cart_pole_env.py](env/cart_pole_env.py) file contains the new CartPole class, based on [InvertedPendulum-v2](https://gym.openai.com/envs/InvertedPendulum-v2/).
I also created the [env/cart_pole.xml](env/cart_pole.xml) file defining the MuJoCo environment for the traditional cart pole.

### Prerequisites
The example requires the [MuJoCo](http://www.mujoco.org/) (Multi-Joint dynamics with Contact) physics engine in order to use [OpenAI's](https://gym.openai.com/) Inverted Pendulum [simulation environment](https://gym.openai.com/envs/InvertedPendulum-v2/).
I believe free student licence's are available.

### Installing
Install the requirements using ```pip install -r requirements```.
- Make sure you use Python 3.
- You may want to use a virtual environment for this.

### Example
An example of implementing the code is given for the cart pole environment and can be found in [examples/cart_pole.py](./examples/cart_pole.py).



## Built With
- [Tensorflow](https://www.tensorflow.org/)
- [GPflow](https://github.com/GPflow/GPflow)

## Authors
- Aidan Scannell

## Licence
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements
1. The original implementation of [PILCO](http://mlg.eng.cam.ac.uk/pilco/):

    1. 
        - M. P. Deisenroth, D. Fox, and C. E. Rasmussen 
        - Gaussian Processes for Data-Efficient Learning in Robotics and Control 
        - IEEE Transactions on Pattern Analysis and Machine Intelligence, 2014 
    2. 
        - M. P. Deisenroth and C. E. Rasmussen 
        - PILCO: A Model-based and Data-Efficient Approach to Policy Search 
        - International Conference on Machine Learning (ICML), 2011 

2. I took inspiration and some code from this [Python implementation](https://github.com/nrontsis/PILCO).