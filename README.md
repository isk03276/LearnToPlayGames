# LearnToPlayGames
**Learning various games based on DRL(Deep Reinforcement Learning)**

- Supported games : Tetris(gym-tetris)  
- DRL framework : rllib  
- Supported DRL algorithms : PPO  


## Installation
Install packages required to execute the code.  
~~~
$ pip install -r requirements.txt
~~~

## Train
You can train the model with the command below.
If you want to accelerate learning, increase the number of gpus.
Set ml-framework according to the machine learning framework you are using. (ex. "torch" or "tf")
~~~
$ python main.py --save --num-gpus NUM_OF_GPUS --ml-framework ML_FRAMEWORK
~~~

## Test
You can test the model with the command below.
The trained models are saved in the "checkpoints" directory.  
~~~
$ python main.py --test --load-from TRAINED_MODEL_PATH
~~~

## Experiments
