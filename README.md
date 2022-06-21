# LearnToPlayGames
Learning various games based on DRL(Deep Reinforcement Learning).
Supported games : Tetris(gym-tetris)
DRL framework : rllib
Supported DRL algorithms : PPO

## Installation
'''
$ pip install -r requirements.txt
'''

## Train
'''
$ python main.py --save --num-gpus NUM_OF_GPUS(ex.1) --ml-framework ML_FRAMEWORK(ex.torch)
'''

## Test
'''
$ python main.py --test --load-from TRAINED_MODEL_PATH
'''

# Experiments