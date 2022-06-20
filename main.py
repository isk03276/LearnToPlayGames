import argparse

from utils.rllib import save_model, load_model, make_folder_name

import ray
from ray import tune
from ray.rllib.agents import ppo


def get_env_generator(env_id:str):
    if env_id == "TetrisA-v0":
        from envs.tetris import make_tetris_env as env_generator
    else:
        raise NotImplementedError
    return env_generator
    
def run(config):
    ray.init()
    env_id = config.env_id
    env_generator = get_env_generator(env_id) 
    tune.register_env(env_id, lambda _: env_generator(env_id, config.render))
    
    rllib_config = ppo.DEFAULT_CONFIG.copy()
    rllib_config['framework'] = config.ml_framework
    
    trainer = ppo.PPOTrainer(env=env_id, config=rllib_config)
    if config.load_from is not None:
        load_model(trainer, config.load_from)
    path_to_save = "checkpoints/" + make_folder_name()

    status = "{:2d} reward {:6.2f} len {:4.2f}"
    
    for n in range(10000):
        result = trainer.train()
        print(status.format(
            n + 1,
            result["episode_reward_mean"],
            result["episode_len_mean"],
        ))
        if args.save:
            save_model(trainer, path_to_save)
    
    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game environments to learn")
    parser.add_argument("--env-id", default="TetrisA-v0", type=str, help="game environment id: 'TetrisA-v0', ...")
    parser.add_argument("--ml-framework", default="torch", type=str, help="Machine learning framework(ex. 'torch', 'tensorflow', ...)")
    parser.add_argument("--render", action="store_true", help="Turn on rendering")
    #model
    parser.add_argument("--save", action="store_true", help="Whether to save the model")
    parser.add_argument("--load-from", type=str, help="Path to load the model")
    
    args = parser.parse_args()
    run(args)