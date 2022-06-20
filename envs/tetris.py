from envs.wrappers import FrameProcessingWrapper, FrameRenderingWrapper

import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace


def make_tetris_env(env_id:str = "TetrisA-v0", render=False)-> gym_tetris.TetrisEnv:
    """
    Make tetris env class instance.
    Args:
        env_id (str, optional): Tetris env id. Defaults to "TetrisA-v0".
        render (bool, optional): Whether to render. Defaults to False.

    Returns:
        gym_tetris.TetrisEnv: Tetris env instance.
    """
    env = gym_tetris.make(env_id)
    env = JoypadSpace(env, MOVEMENT)
    env = FrameProcessingWrapper(env)
    if render:
        env = FrameRenderingWrapper(env)
    return env
