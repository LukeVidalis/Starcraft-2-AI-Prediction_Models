#from PIL import Image
from pysc2.env.environment import StepType
from pysc2.lib import features


class ObserverAgent():
    def step(self, obs):
        #print("{}")#.format(time_step.observation["game_loop"]))
        #print(obs.observation_spec().obs_spec)
        #minimap = obs.observation["minimap"]
        print(obs)
        #_features = features.Feature
        #unpack = _features.unpack(self, obs)
        #image = _features.unpack_rgb_image()
        #print(image)








        """
        super(ReplayObserver, self).step(obs)

        if obs.step_type != StepType.FIRST:
            minimap = obs.observation['minimap']
            Image.fromarray((minimap[3] > 0).astype('uint8') * 255).save('~/{}_minimap.png')"""