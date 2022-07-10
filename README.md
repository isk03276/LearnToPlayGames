# LearnToPlayGames
**Learning various games based on DRL(Deep Reinforcement Learning)**

- Supported games : Tetris(gym-tetris)  
- DRL framework : rllib  
- Supported DRL algorithms : PPO  

<img src="https://user-images.githubusercontent.com/23740495/178166394-40d4190d-54fe-4d86-9215-82fe99f71b62.gif" width="200" height="200"/>



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

### Experiments
[Reach]  
![image](https://user-images.githubusercontent.com/23740495/178166097-d6d2326f-2b63-455b-8489-94084f4a9fdf.png)

