
class ObserverAgent:

    def step(self, obs):
        # game_step = self.observation["game_loop"][0]
        minimap = self.observation["rgb_minimap"]
        return minimap

