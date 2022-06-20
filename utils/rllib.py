import datetime

from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents import ppo


def get_ppo_config(framework:str = "torch",
                   num_workers:int = 8,
                   vf_share_layers:bool = True,
                   use_lstm:bool = True,
                   lstm_cell_size:int = 256)-> dict:
    
    """
    Generate rllib ppo config.
    Args:
        framework (str): rllib trainer
        num_workers (int): number of workers for collecting training data 
        vf_share_layers (bool): Whether policy and value share layers
        use_lstm (bool)): Whether to use lstm layers
        lstm_cell_size (int): LSTM cell size
        
    Returns:
        dict: rllib ppo config
    """
    rllib_config = ppo.DEFAULT_CONFIG.copy()
    rllib_config['framework'] = framework
    rllib_config['num_workers'] = num_workers
    rllib_config['model']['vf_share_layers'] = vf_share_layers
    rllib_config['model']['use_lstm'] = use_lstm
    rllib_config['model']['lstm_cell_size'] = lstm_cell_size
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
