import gym
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class BaseUnityEnv(gym.Env):
    def __init__(self, built_unity_file:str, port:int = 5004):
        self.init_env(built_unity_file, port)
        print(self.env)
        print(self.env.observation_space)
        print(self.env.action_space)
    
    def init_env(self, built_unity_file:str, port:int):
        channel = EngineConfigurationChannel()
        param_channel = EnvironmentParametersChannel()
        worker_id = 0 # modifiy###
        seed = 0
        while True:
            worker_id += 1
            seed += 1
            try:
                unity_env = UnityEnvironment(
                    file_name=built_unity_file,
                    seed=seed,
                    side_channels=[channel, param_channel],
                    worker_id=worker_id,
                    base_port=port,
                )
            except UnityWorkerInUseException as e:
                print(e)
                input("is it right?")
            else:
                break
        self.env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)
    