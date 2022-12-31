from typing import Tuple
import gym
import torch
from vicregdonkey.augmentations import TrainTransform
import pkg_resources

import numpy as np

class VICREGWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0','resnet18',pretrained=False)
        DATA_PATH = pkg_resources.resource_filename('vicregdonkey', '/')
        model_name="resnet18.pth"
        self.resnet.load_state_dict(torch.load(DATA_PATH+model_name, map_location=torch.device('cpu')))
        self.resnet.eval()
        #self.resnet.cuda()
        self.data_transforms = TrainTransform()
        self.observation_space = gym.spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (1000,),
            dtype = np.float32
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, ]:
        obs, reward, done, infos = self.env.step(action)
        # Encode with the pre-trained VICREG
        obs = self.data_transforms(obs).unsqueeze(0)
        with torch.no_grad():       
            encoded_image = self.resnet(obs)
        return encoded_image, reward, done, infos

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        obs = self.data_transforms(obs).unsqueeze(0)
        with torch.no_grad():
            encoded_image = self.resnet(obs)
        return encoded_image
