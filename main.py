import argparse

from utils.rllib import save_model, load_model, make_folder_name
from utils.config import load_config

import ray
from ray import tune
from ray.rllib.agents import ppo


def get_env_generator(env_id:str):
    if "tetris" in env_id.lower():
        from envs.tetris import make_tetris_env as env_generator
    elif env_id =="unity":
        from envs.base_unity_env import BaseUnityEnv as env_generator
    else:
        raise NotImplementedError
    return env_generator

def train(trainer, learning_iteration_num, to_save, save_interval):
    path_to_save = "checkpoints/" + make_folder_name()

    status = "[Train] {:2d} reward {:6.2f} len {:4.2f}"
    
    for iter in range(1, learning_iteration_num + 1):
        result = trainer.train()
        print(status.format(
            iter,
            result["episode_reward_mean"],
            result["episode_len_mean"],
        ))
        if to_save and iter % save_interval == 0:
            save_model(trainer, path_to_save)
    
def test(env, trainer, test_num):
    for ep in range(test_num):
        done = False
        obs = env.reset()
        rews = []
        
        status = "[Test] {:2d} reward {:6.2f} len {:4.2f}"
        
        while not done:
            action = trainer.compute_action(obs)
            obs, rew, done, _ = env.step(action)
            rews.append(rew)
        print(status.format(ep + 1, sum(rews)/len(rews), len(rews)))

def run(args):
    ray.init()
    env_id = args.env_id
    env_generator = get_env_generator(env_id) 
    tune.register_env(env_id, lambda _: env_generator(env_id, args.render))
    rllib_configs = load_config(args.config_file_path)
    trainer = ppo.PPOTrainer(env=env_id, config=rllib_configs)
    
    if args.load_from is not None:
        load_model(trainer, args.load_from)
        
    if not args.test:
        train(trainer, args.learning_iteration_num, args.save, args.save_interval)
    test_env = env_generator(env_id, render=True)
    test(test_env, trainer, args.test_num)
    
    ray.shutdown()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game environments to learn")
    parser.add_argument("--env-id", default="TetrisA-v0", type=str, help="game environment id: 'TetrisA-v0', ...")
    parser.add_argument("--built-unity-file", type=str, help="Built unity game file")
    parser.add_argument("--config-file-path", default="configs/default_config.yaml", type=str, help="Rllib config file path")
    parser.add_argument("--render", action="store_true", help="Turn on rendering")
    #model
    parser.add_argument("--save", action="store_true", help="Whether to save the model")
    parser.add_argument("--save-interval", type=int, default=20, help="Model save interval")
    parser.add_argument("--load-from", type=str, help="Path to load the model")
    #train/test
    parser.add_argument("--learning-iteration-num", type=int, default=100000, help="Number of iteration to train the model")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument("--test-num", type=int, default=10, help="Number of episodes to test the model")
    
    args = parser.parse_args()
    run(args)