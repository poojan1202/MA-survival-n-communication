from typing import Callable, Optional, List, Union
import time
import os
import json
import pprint

import pygame

import numpy as np

from masurvival.envs.masurvival_env import MaSurvival
from masurvival.envs.macomms_env import MaComms


class MultiAgentPolicy:
    def act(self, observations):
        raise NotImplementedError


class RandomPolicy(MultiAgentPolicy):
    def __init__(self, env):
        self.action_space = env.action_space
        # print(self.action_space)

    def act(self, observations):
        tup = self.action_space.sample()
        num_agents = len(tup) / 2
        arr_list = []
        # arr = np.zeros((num_agents,len(tup[0])+1),dtype=float)
        for i in range(int(num_agents)):
            arr_list.append(tup[2 * i])
            arr_list[i] = np.array(list(tup[2 * i]) + list(tup[2 * i + 1]))
        actions = tuple(arr_list)
        # return actions
        return self.action_space.sample()


def zero_action():
    return [1, 1, 1, 0, 0, 0]


def get_action_from_keyboard(last_pressed):
    keyboard = pygame.key.get_pressed()
    # TODO update this with correct null actions
    action = zero_action()
    if keyboard[pygame.K_w] and not keyboard[pygame.K_s]:
        action[0] = 2
    if keyboard[pygame.K_s] and not keyboard[pygame.K_w]:
        action[0] = 0
    if keyboard[pygame.K_a] and not keyboard[pygame.K_d]:
        action[1] = 2
    if keyboard[pygame.K_d] and not keyboard[pygame.K_a]:
        action[1] = 0
    if keyboard[pygame.K_LEFT] and not keyboard[pygame.K_RIGHT]:
        action[2] = 2
    if keyboard[pygame.K_RIGHT] and not keyboard[pygame.K_LEFT]:
        action[2] = 0
    if keyboard[pygame.K_c] and not last_pressed[pygame.K_c]:
        action[3] = 1
    if keyboard[pygame.K_e] and not last_pressed[pygame.K_e]:
        action[4] = 1
    if keyboard[pygame.K_q] and not last_pressed[pygame.K_q]:
        action[5] = 1
    return action, keyboard


controls_doc = """Keyboard controls:
W,A,S,D: parallel and normal movement
LEFT,RIGHT: angular movement
C: attack
E: use item (last picked up)
Q: give item (last picked up)
-------------
"""


class HumanPolicy(MultiAgentPolicy):

    def __init__(self, env):
        self.action_space = env.action_space
        self.n_agents = env.n_agents

    def act(self, observations):
        actions = self.action_space.sample()
        user_action, self.pressed = get_action_from_keyboard(self.pressed)
        if self.n_agents >= 2:
            actions = (user_action,) + actions[1:]
        elif self.n_agents == 1:
            actions = (user_action,)
        return actions


def demo_env(
        env,
        policy: str = 'random',  # either 'random' or 'interactive'
        max_steps: Optional[int] = None,
        render: bool = False,
        record: Optional[Callable[[int, np.ndarray], None]] = None,
        print_benchmark: bool = False,
) -> None:
    """Run a demo of the environment.

    If max_steps is None, the environment is ran for one episode.
    Otherwise, the environment is ran either until the episode is done
    or until max_steps steps have been performed, whichever occurs
    first.

    If policy == "interactive", the environment always gets rendered in
    a window ("human" render mode), regardless of the value of render.
    Otherwise, the value of render controls whether to render to a
    window or not.

    If record is not None, it forces the environment to render each
    state and is called after each render with two arguments:
    - t: the number of environment steps performed
    - frame: the frame of the rendered environment state (as a numpy
      array)
    Note that the environment is rendered after the initial environment
    reset and after each step. The render mode will be rgb_array if the
    human render mode is not forced in any other way.
    """
    interactive = policy == 'interactive'
    if interactive:
        print(controls_doc)
        policy = HumanPolicy(env)
    else:
        assert policy == 'random'
        policy = RandomPolicy(env)

    render_mode = None
    if record is not None:
        render_mode = 'rgb_array'
    if interactive or render:
        render_mode = 'human'

    times = None if not print_benchmark else []
    t, observation, done = 0, env.reset(), False

    def render():
        if render_mode is not None:
            frame = env.render(mode=render_mode)
            if record is not None:
                record(t, frame)

    render()

    if interactive:
        policy.pressed = pygame.key.get_pressed()
    while not done:
        action = policy.act(observation)

        t_start = time.process_time()
        observation, reward, done, info = env.step(action)
        if t == 1:
            print(action)
        if t == 2:
            print(observation['agent'])
            print(action)
        t_end = time.process_time()

        t += 1
        if times is not None:
            times.append(t_end - t_start)

        render()

        if max_steps is not None and t == max_steps:
            print(f'Maximum number of steps {t} reached, terminating episode.')
            break

    print('Episode complete. Stats printed below.')
    stats = env.flush_stats()
    env.close()
    pprint.PrettyPrinter().pprint(stats)

    if print_benchmark:
        times = np.array(times)
        print(f'Performance test results: {times.mean()}, {times.std()}')

    return


# Script stuff

def main(
        policy: str,
        max_steps: int,
        env_config_fpath: Optional[str],
        render: bool,
        screenshot_fpath: Optional[str],
        screenshot_step: int,
        gif_fpath: Optional[str],
        gif_record_interval: int,
        print_benchmark: bool,
):
    if env_config_fpath is not None:
        with open(env_config_fpath) as f:
            config = json.load(f)
        # env = MaSurvival(config=config)
        env = MaComms(config=config)
    else:
        # env = MaSurvival()
        env = MaComms()

    recording = dict(screenshot=None, gif=None if gif_fpath is None else [])

    def record(t, frame):
        if screenshot_fpath is not None and t == screenshot_step:
            recording['screenshot'] = frame
        if gif_fpath is not None and t % gif_record_interval == 0:
            recording['gif'].append(frame)

    demo_env(
        env,
        policy=policy,
        max_steps=max_steps,
        render=render,
        record=record,
        print_benchmark=print_benchmark,
    )

    if screenshot_fpath is not None:
        import imageio
        print(f'Saving screenshot to {screenshot_fpath}.')
        imageio.imsave(screenshot_fpath, recording['screenshot'])

    if gif_fpath is not None:
        import imageio
        from pygifsicle import optimize
        with imageio.get_writer(gif_fpath, mode='I') as writer:
            for frame in recording['gif']:
                writer.append_data(frame)
        optimize(gif_fpath)


argparse_desc = \
    'Test the environment for one episode, either '
'interactively or with a random policy.'

argparse_args = [
    (['policy'], dict(
        metavar='POLICY',
        type=str,
        default='random',
        nargs='?',
        choices=['random', 'interactive'],
        help='The policy to use for testing, either "random" or "interactive".',
    )),
    (['--max-steps'], dict(
        dest='max_steps',
        metavar='STEPS',
        type=int,
        default=None,
        help='Run only for the given amount of steps.'
    )),
    (['-c', '--config'], dict(
        dest='env_config_fpath',
        metavar='PATH',
        type=str,
        default=None,
        help='Use the given JSON file as the env configuration.'
    )),
    (['-r', '--render'], dict(
        action='store_true',
        dest='render',
        default=False,
        help='Whether to render to a window (when the policy is not interactive).',
    )),
    (['-s', '--screenshot'], dict(
        dest='screenshot_fpath',
        metavar='PATH',
        type=str,
        default=None,
        help='Record a frame of the episode to the given file. Which frame is recorded can be controlled with --screenshot-step.'
    )),
    (['--screenshot-step'], dict(
        dest='screenshot_step',
        metavar='STEP',
        type=int,
        default=0,
        help='Control which step gets recorded by --screenshot (0 corresponds to the frame rendered before the first step).'
    )),
    (['-g', '--gif'], dict(
        dest='gif_fpath',
        metavar='PATH',
        type=str,
        default=None,
        help='Record a GIF to the given file. The gif framerate can be controlled with --gif-record-interval.'
    )),
    (['--gif-record-interval'], dict(
        dest='gif_record_interval',
        metavar='N',
        type=int,
        default=10,
        help='The interval (in number of steps) between each frame of the gif recorded with --gif.'
    )),
    (['--benchmark'], dict(
        dest='print_benchmark',
        action='store_true',
        default=False,
        help='Print benchmark information at the end of the episode.'
    )),
]

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(
        description=argparse_desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for args, kwargs in argparse_args:
        argparser.add_argument(*args, **kwargs)
    cli_args = argparser.parse_args()
    # print(vars(cli_args))
    main(**vars(cli_args))

