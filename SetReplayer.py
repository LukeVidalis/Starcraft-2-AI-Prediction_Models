#from pysc2.lib import features, point
#from absl import app, flags
#from pysc2.env.environment import TimeStep, StepType
#from pysc2 import run_configs
#from s2clientprotocol import sc2api_pb2 as sc_pb

import getpass
import json
import platform
import sys
import time

from absl import app
from absl import flags
import mpyq
import six
from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import renderer_human
from pysc2.lib import stopwatch
from pysc2.run_configs import lib as run_configs_lib

from PIL import Image

from s2clientprotocol import sc2api_pb2 as sc_pb
import importlib

FLAGS = flags.FLAGS
#flags.DEFINE_string("replay", None, "Path to a replay file.")
flags.DEFINE_string("agent", None, "Path to an agent.")
#flags.mark_flag_as_required("replay")
flags.mark_flag_as_required("agent")
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_bool("realtime", False, "Whether to run in realtime mode.")
flags.DEFINE_bool("full_screen", False, "Whether to run full screen.")

flags.DEFINE_float("fps", 22.4, "Frames per second to run the game.")
flags.DEFINE_integer("step_mul", 1, "Game steps per observation.")
flags.DEFINE_bool("render_sync", False, "Turn on sync rendering.")
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", "256,192",
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", "128",
                        "Resolution for rendered minimap.")
flags.DEFINE_string("video", None, "Path to render a video of observations.")

flags.DEFINE_integer("max_game_steps", 0, "Total game steps to run.")
flags.DEFINE_integer("max_episode_steps", 0, "Total game steps per episode.")

flags.DEFINE_string("user_name", getpass.getuser(),
                    "Name of the human player for replays.")
flags.DEFINE_enum("user_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "User's race.")
flags.DEFINE_enum("bot_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "AI race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "Bot's strength.")
flags.DEFINE_bool("disable_fog", False, "Disable fog of war.")
flags.DEFINE_integer("observed_player", 1, "Which player to observe.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use to play.")

flags.DEFINE_string("map_path", None, "Override the map for this replay.")
flags.DEFINE_string("replay", None, "Name of a replay to show.")

def main(unused):
    #stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    #stopwatch.sw.trace = FLAGS.trace

    if (FLAGS.map and FLAGS.replay) or (not FLAGS.map and not FLAGS.replay):
        sys.exit("Must supply either a map or replay.")

    if FLAGS.replay and not FLAGS.replay.lower().endswith("sc2replay"):
        sys.exit("Replay must end in .SC2Replay.")

    if FLAGS.realtime and FLAGS.replay:
        # TODO(tewalds): Support realtime in replays once the game supports it.
        sys.exit("realtime isn't possible for replays yet.")

    if FLAGS.render and (FLAGS.realtime or FLAGS.full_screen):
        sys.exit("disable pygame rendering if you want realtime or full_screen.")

    if platform.system() == "Linux" and (FLAGS.realtime or FLAGS.full_screen):
        sys.exit("realtime and full_screen only make sense on Windows/MacOS.")

    if not FLAGS.render and FLAGS.render_sync:
        sys.exit("render_sync only makes sense with pygame rendering on.")

    run_config = run_configs.get()

    interface = sc_pb.InterfaceOptions()
    interface.raw = FLAGS.render
    interface.score = True
    interface.feature_layer.width = 24
    if FLAGS.feature_screen_size and FLAGS.feature_minimap_size:
        FLAGS.feature_screen_size.assign_to(interface.feature_layer.resolution)
        FLAGS.feature_minimap_size.assign_to(
            interface.feature_layer.minimap_resolution)
    if FLAGS.rgb_screen_size and FLAGS.rgb_minimap_size:
        FLAGS.rgb_screen_size.assign_to(interface.render.resolution)
        FLAGS.rgb_minimap_size.assign_to(interface.render.minimap_resolution)

    max_episode_steps = FLAGS.max_episode_steps

    replay_data = run_config.replay_data(FLAGS.replay)
    start_replay = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=FLAGS.disable_fog,
        observed_player_id=FLAGS.observed_player)
    version = get_replay_version(replay_data)


    with run_config.start(version=version,
                      full_screen=FLAGS.full_screen) as controller:
        info = controller.replay_info(replay_data)
        print(" Replay info ".center(60, "-"))
        print(info)
        print("-" * 60)
        map_path = FLAGS.map_path or info.local_map_path
        if map_path:
            start_replay.map_data = run_config.map_data(map_path)
        controller.start_replay(start_replay)

        
        _features = features.features_from_game_info(controller.game_info())
        while True:
            controller.step(self.step_mul)
            obs = controller.observe()
            try:
                agent_obs = _features.transform_obs(obs)
            except:
                pass

            if obs.player_result:  # Episide over.
                _state = StepType.LAST
                discount = 0
            else:
                discount = self.discount

            _episode_steps += self.step_mul

            step = TimeStep(step_type=_state, reward=0,
                            discount=discount, observation=agent_obs)

            agent.step(step, obs.actions)

            if obs.player_result:
                break

            self._state = StepType.MID



def step():
    print("helouhiuhilhuiuphih")

def get_replay_version(replay_data):
  replay_io = six.BytesIO()
  replay_io.write(replay_data)
  replay_io.seek(0)
  archive = mpyq.MPQArchive(replay_io).extract()
  metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
  return run_configs_lib.Version(
      game_version=".".join(metadata["GameVersion"].split(".")[:-1]),
      build_version=int(metadata["BaseBuild"][4:]),
      data_version=metadata.get("DataVersion"),  # Only in replays version 4.1+.
      binary=None)


def entry_point():  # Needed so setup.py scripts work.
  app.run(main)


if __name__ == "__main__":
  app.run(main)