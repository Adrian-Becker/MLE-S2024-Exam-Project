import math
import os
import pickle
import random
import time
from collections import deque
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, find_distance_to_coin, ACTION_TO_INDEX, breadth_first_search, \
    determine_next_action, prepare_field_coins, ACTION_INDICES, EPS_START, EPS_END, EPS_DECAY, determine_explosion_timer

from .history import TransitionHistory, Transition

import tqdm

from .stats_logger import StatsLogger, Stat

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions

TRANSITION_ENEMY_EPS_START = 0.999
TRANSITION_ENEMY_EPS_END = 0.00
TRANSITION_ENEMY_EPS_DECAY = 40000

BATCH_SIZE = 128
EPOCHS_PER_ROUND = 10

# Events
MOVED_TOWARDS_COIN_EVENT = "Moved Towards Coin"
MOVED_AWAY_FROM_COIN_EVENT = "Moved Away from Coin"
ESCAPE_BOMB_EVENT = "Escape Bomb"
WAITED_WITHOUT_NEED_EVENT = "Waited Without Need"

PLACED_BOMB_DESTROY_ONE_EVENT = "Placed Bomb Safely Destroy One"
PLACED_BOMB_DESTROY_MULTIPLE_EVENT = "Placed Bomb Safely Destroy Multiple"

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99

SYMMETRY_SYNC_RATE = 5

ROUNDS_PER_SAVE = 250

CONVERSIONS = [
    {
        # nothing
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    },
    {
        # mirror left/right
        0: 0,
        1: 1,
        2: 3,
        3: 2,
        4: 4,
        5: 5
    },
    {
        # mirror top/bottom
        0: 1,
        1: 0,
        2: 2,
        3: 3,
        4: 4,
        5: 5
    },
    {
        # rotate 90deg
        0: 2,
        1: 3,
        2: 1,
        3: 0,
        4: 4,
        5: 5
    },
    {
        # rotate 180deg
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        4: 4,
        5: 5
    },
    {
        # rotate 270deg
        0: 3,
        1: 2,
        2: 0,
        3: 1,
        4: 4,
        5: 5
    }
]


def setup_training(self):
    """
    Initialise agent for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.P = np.copy(self.Q)
    self.transitions = TransitionHistory(TRANSITION_HISTORY_SIZE)
    self.stats_logger = StatsLogger([
        Stat('points', 'points', True, 500, '{:2d}', '{:5.2f}', 32, ' '),
        Stat('P(copy enemy)', 'p-enemy', False, 1, '{:5.2f}', '', 36, '%'),
        Stat('P(exploration)', 'p-explo', False, 1, '{:5.2f}', '', 36, '%'),
        Stat('bombs', 'bombs', True, 500, '{:3d}', '{:6.2f}', 31, ' '),
        Stat('rewards', 'rewards', True, 500, '{:7d}', '{:9.2f}', 32, ' '),
        Stat('iterations p. round', 'iteration', True, 500, '{:3d}', '{:6.2f}', 35, ' '),
        Stat('invalid moves', 'invalid', True, 500, '{:5.2f}', '{:5.2f}', 35, '%'),
        Stat('kills', 'kills', True, 500, '{:1d}', '{:4.2f}', 34, ' '),
        Stat('suicides', 'suicides', True, 500, '{:1d}', '{:4.2f}', 34, ' ')
    ])

    self.round = 0
    self.bombs_dropped = 0
    self.save = 0
    self.total_rewards = 0
    self.iteration_per_round = 0
    self.invalid_moves = 0
    self.suicides = 0
    self.kills = 0

    if not os.path.exists("tables"):
        os.makedirs("tables")

    try:
        os.remove("stats.csv")
    except FileNotFoundError:
        pass


def add_custom_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str],
                      features_old, features_new):
    if e.COIN_COLLECTED not in events and len(old_game_state['coins']) > 0 and len(new_game_state['coins']) > 0:
        # no coin collected => agent might have moved closer to coin
        distance_old = breadth_first_search(
            old_game_state['self'][3],
            *prepare_field_coins(old_game_state, determine_explosion_timer(old_game_state)))
        distance_new = breadth_first_search(
            new_game_state['self'][3],
            *prepare_field_coins(new_game_state, determine_explosion_timer(new_game_state)))

        if distance_old > distance_new:
            events.append(MOVED_TOWARDS_COIN_EVENT)
        elif distance_new > distance_old:
            events.append(MOVED_AWAY_FROM_COIN_EVENT)
    if e.BOMB_DROPPED not in events:
        if old_game_state['explosion_map'][old_game_state['self'][3]] > 0 and \
                new_game_state['explosion_map'][new_game_state['self'][3]] == 0:
            events.append(ESCAPE_BOMB_EVENT)
    if e.BOMB_DROPPED in events:
        self.bombs_dropped += 1
    if e.BOMB_DROPPED in events:
        if features_old[0] == 2:
            events.append(PLACED_BOMB_DESTROY_ONE_EVENT)
        elif features_old[0] == 3:
            events.append(PLACED_BOMB_DESTROY_MULTIPLE_EVENT)
    if e.WAITED in events and (max(features_old[1:5]) > 0 or features_old[0] > 0):
        events.append(WAITED_WITHOUT_NEED_EVENT)


def learning_step(self, transition: Transition):
    features_old, action_old, features_new, rewards = transition
    action_new_Q = np.array(list(map(lambda action: self.Q[features_new][action], ACTION_INDICES))).argmax()
    action_new_P = np.array(list(map(lambda action: self.P[features_new][action], ACTION_INDICES))).argmax()
    valQ = self.Q[features_old][action_old]
    valP = self.P[features_old][action_old]
    self.Q[features_old][action_old] = valQ + LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.P[features_new][action_new_Q] - self.Q[features_old][action_old])
    self.P[features_old][action_old] = valP + LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.Q[features_new][action_new_P] - self.P[features_old][action_old])


def handle_event_occurrence(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    features_old = state_to_features(old_game_state)
    action_old = ACTION_TO_INDEX[self_action]
    features_new = state_to_features(new_game_state)

    add_custom_events(self, old_game_state, self_action, new_game_state, events, features_old, features_new)

    rewards = reward_from_events(self, events)
    self.total_rewards += rewards

    # state_to_features is defined in callbacks.py
    transition = Transition(features_old, action_old, features_new, rewards)
    learning_step(self, transition)
    self.transitions.append(transition)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    Updates the Q-table based on how well the agent did in the current round.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    handle_event_occurrence(self, old_game_state, self_action, new_game_state, events)

    self.iteration += 1
    self.iteration_per_round += 1
    if e.KILLED_OPPONENT in events:
        self.kills += 1
    if e.INVALID_ACTION in events:
        self.invalid_moves += 1


def enemy_game_events_occurred(self, name, old_game_state, self_action, new_game_state, events):
    random_prob = TRANSITION_ENEMY_EPS_END + (TRANSITION_ENEMY_EPS_START - TRANSITION_ENEMY_EPS_END) * \
                  math.exp(-1. * self.iteration / TRANSITION_ENEMY_EPS_DECAY)
    if self.train and random.random() > random_prob:
        return

    if old_game_state is None or self_action is None:
        return

    handle_event_occurrence(self, old_game_state, self_action, new_game_state, events)


def optimize(self):
    if len(self.transitions) < BATCH_SIZE + 1:
        return
    batch = self.transitions.sample(BATCH_SIZE)
    for transition in batch:
        learning_step(self, transition)


def transmute_neighbors(conversion, indices):
    new = [0, 0, 0, 0]
    new[conversion[0]] = indices[0]
    new[conversion[1]] = indices[1]
    new[conversion[2]] = indices[2]
    new[conversion[3]] = indices[3]
    return tuple(new)


def sync_symmetries(Q):
    new_Q = np.zeros_like(Q)

    for current_field in range(0, 4):
        for f1 in range(0, 2):
            for f2 in range(0, 2):
                for f3 in range(0, 2):
                    for f4 in range(0, 2):
                        for mode in range(0, 4):
                            for t1 in range(0, 2):
                                for t2 in range(0, 2):
                                    for t3 in range(0, 2):
                                        for t4 in range(0, 2):
                                            for action in ACTION_INDICES:
                                                value = 0
                                                neighbor_fields = (f1, f2, f3, f4)
                                                trap_fields = (t1, t2, t3, t4)
                                                for conversion in CONVERSIONS:
                                                    converted_neighbor_fields = transmute_neighbors(conversion,
                                                                                                    neighbor_fields)
                                                    converted_trap_fields = transmute_neighbors(conversion, trap_fields)
                                                    value += Q[current_field][converted_neighbor_fields][mode][
                                                        converted_trap_fields][conversion[action]]
                                                new_Q[current_field][neighbor_fields][mode][trap_fields][
                                                    action] = value / 6.0
    return new_Q


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of the round, saves the current state of the Q-table so that it can be restored after training.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.transitions.append(
        Transition(state_to_features(last_game_state),
                   ACTION_TO_INDEX[last_action],
                   (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), reward_from_events(self, events))
    )

    for _ in range(EPOCHS_PER_ROUND):
        optimize(self)
    self.iteration += 1
    self.iteration_per_round += 1
    self.round += 1

    if self.round % SYMMETRY_SYNC_RATE == 0:
        self.Q = sync_symmetries(self.Q)
        self.P = sync_symmetries(self.P)
        pass

    if self.round % ROUNDS_PER_SAVE == 0:
        # Store the model
        self.save += 1
        with open("tables/" + "{:04d}".format(self.save) + "-q-table.pt", "wb") as file:
            pickle.dump(self.Q, file)

    prob_enemy_copy = TRANSITION_ENEMY_EPS_END + (TRANSITION_ENEMY_EPS_START - TRANSITION_ENEMY_EPS_END) * \
                      math.exp(-1. * self.iteration / TRANSITION_ENEMY_EPS_DECAY)
    prob_exploration = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.iteration / EPS_DECAY)

    self.stats_logger.add('p-enemy', prob_enemy_copy * 100)
    self.stats_logger.add('p-explo', prob_exploration * 100)

    self.stats_logger.add('points', last_game_state['self'][1])

    self.stats_logger.add('bombs', self.bombs_dropped)
    self.bombs_dropped = 0

    self.stats_logger.add('invalid', self.invalid_moves / self.iteration_per_round * 100)
    self.invalid_moves = 0

    self.stats_logger.add('iteration', self.iteration_per_round)
    self.iteration_per_round = 0

    self.stats_logger.add('rewards', self.total_rewards)
    self.total_rewards = 0

    self.stats_logger.add('suicides', 1 if e.KILLED_SELF in events else 0)

    self.stats_logger.add('kills', self.kills)
    self.kills = 0

    self.stats_logger.output(self.round, self.iteration)


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculated the rewards for a given action based on its event list.
    Rewards are defined in the global variable GAME_REWARDS.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10000,
        e.KILLED_OPPONENT: 50000,
        e.KILLED_SELF: -10000,
        e.GOT_KILLED: -4000,
        e.INVALID_ACTION: -5000,
        e.WAITED: 5000,
        WAITED_WITHOUT_NEED_EVENT: -9000,
        e.BOMB_DROPPED: 1000,
        e.CRATE_DESTROYED: 1000,
        e.COIN_FOUND: 1000,
        MOVED_TOWARDS_COIN_EVENT: 1000,
        MOVED_AWAY_FROM_COIN_EVENT: -1000,
        ESCAPE_BOMB_EVENT: 2000,
        PLACED_BOMB_DESTROY_ONE_EVENT: 2500,
        PLACED_BOMB_DESTROY_MULTIPLE_EVENT: 4000
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
