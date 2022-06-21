from utils.image import rgb_to_gray, resize_image, normalize_image
import gym

    
    
class FrameProcessingWrapper(gym.ObservationWrapper):
    """
    A wrapper to resize and grayscale the observations.
    """
    def __init__(self, env, width:int = 84, height:int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.width, self.height, 1))
        
    def observation(self, obs):
        gray_scaled_obs = rgb_to_gray(obs)
        resized_obs = resize_image(gray_scaled_obs, self.width, self.height)
        normalized_obs = normalize_image(resized_obs)
        return normalized_obs
    

class FrameRenderingWrapper(gym.Wrapper):
    """
    Wrapper to enable rendering.
    """
    def step(self, action):
        transition_tuple = self.env.step(action)
        self.env.render()        
        return transition_tuple
