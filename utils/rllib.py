import datetime

from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents import ppo


def get_ppo_config(framework:str = "torch",
                   num_workers:int = 4,
                   train_batch_size:int = 2000,
                   rollout_fragment_length:int = 250,
                   sgd_minibatch_size:int = 64,
                   num_sgd_iter:int = 10,
                   vf_share_layers:bool = True,
                   use_lstm:bool = False,
                   lstm_cell_size:int = 256,
                   num_gpus:int = 1,
                   log_level:str = "ERROR")-> dict:
    
    """
    Generate rllib ppo config.
    Args:
        framework (str): rllib trainer
        num_workers (int): number of workers for collecting training data 
        train_batch_size (int) : batch size
        rollout_fragment_length (int) : size of connected rollout
        sgd_minibatch_size (int) : size of sgd minibatch
        num_sgd_iter (int) : epoch per data
        vf_share_layers (bool): whether policy and value share layers
        use_lstm (bool)): whether to use lstm layers
        lstm_cell_size (int): LSTM cell size
        num_gpus (int): num of gpus for training
        log_level (str): logging level (ex. 'WARN', 'INFO', 'DEBUG', ...)
        
    Returns:
        dict: rllib ppo config
    """
    rllib_config = ppo.DEFAULT_CONFIG.copy()
    rllib_config['framework'] = framework
    rllib_config['num_workers'] = num_workers
    rllib_config['train_batch_size'] = train_batch_size
    rllib_config['rollout_fragment_length'] = rollout_fragment_length
    rllib_config['sgd_minibatch_size'] = sgd_minibatch_size
    rllib_config['num_sgd_iter'] = num_sgd_iter
    rllib_config['model']['vf_share_layers'] = vf_share_layers
    rllib_config['model']['use_lstm'] = use_lstm
    rllib_config['model']['lstm_cell_size'] = lstm_cell_size
    rllib_config['num_gpus'] = num_gpus
    rllib_config['log_level'] = log_level
    rllib_config['batch_mode'] = "complete_episodes"
    return rllib_config


def make_folder_name()-> str:
    """
    Generate current time as string.
    Returns:
        str: current time
    """
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    return curr_time

def save_model(trainer:Trainer, path_to_save:str):
    """
    Save trained model.
    Args:
        trainer (Trainer): rllib trainer
    """
    trainer.save(path_to_save)

def load_model(trainer:Trainer, path_to_load:str):
    """
    Load trained model.
    Args:
        trainer (Trainer): rllib trainer
        path_to_load (str): path to load
    """
    trainer.restore(path_to_load)
