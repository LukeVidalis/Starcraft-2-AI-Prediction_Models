from PIL import Image
from pysc2.env.environment import StepType


class ObserverAgent():
    def step(self, time_step):
        print("{}")#.format(time_step.observation["game_loop"]))








        """
        super(ReplayObserver, self).step(obs)

        if obs.step_type != StepType.FIRST:
            minimap = obs.observation['minimap']
            Image.fromarray((minimap[3] > 0).astype('uint8') * 255).save('~/{}_minimap.png')"""