from PIL import Image
from pysc2.env.environment import StepType
import sys, os
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

class ObserverAgent:
    def step(self, obs):
        print(self.observation["rgb_minimap"])

        game_step = self.observation["game_loop"][0]
        minimap = self.observation["rgb_minimap"]
        PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
        # img = Image.fromarray(array)
        # img.save('testrgb.png')
        Image.fromarray((minimap).astype('uint8') * 255).save(PROJ_DIR+'\Frames\minimap'+str(game_step)+'.png')

        """
        super(ReplayObserver, self).step(obs)

        if obs.step_type != StepType.FIRST:
            minimap = obs.observation['minimap']
            Image.fromarray((minimap[3] > 0).astype('uint8') * 255).save('~/{}_minimap.png')"""
