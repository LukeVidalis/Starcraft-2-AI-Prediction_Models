from PIL import Image
# import sys
import os
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)

class ObserverAgent:

    def step(self, obs, filename):
        # game_step = self.observation["game_loop"][0]
        minimap = self.observation["rgb_minimap"]
        # proj_dir = os.path.dirname(os.path.abspath(__file__))
        # Image.fromarray(minimap.astype('uint8')).save(proj_dir+'\\Frames\\'+file+'_frame_'+str(game_step)+'.png')

        return minimap

