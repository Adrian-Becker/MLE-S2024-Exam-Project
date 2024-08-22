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

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions

TRANSITION_ENEMY_EPS_START = 0.999
TRANSITION_ENEMY_EPS_END = 0.00
TRANSITION_ENEMY_EPS_DECAY = 4000000

BOMB_EPS_START = 300.00
BOMB_EPS_END = 75.00
BOMB_EPS_DECAY = 50000

BATCH_SIZE = 128
EPOCHS_PER_ROUND = 10

# Events
MOVED_TOWARDS_COIN_EVENT = "Moved Towards Coin"
MOVED_AWAY_FROM_COIN_EVENT = "Moved Away from Coin"
ESCAPE_BOMB_EVENT = "Escape Bomb"

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99

SYMMETRY_SYNC_RATE = 5

ROUNDS_PER_SAVE = 250


def setup_training(self):
    """
    Initialise agent for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.P = np.copy(self.Q)
    self.transitions = TransitionHistory(TRANSITION_HISTORY_SIZE)
    self.point_history = deque([0], maxlen=100)
    self.bomb_history = deque([0], maxlen=100)
    self.time_history = deque([time.time()], maxlen=100)
    self.round = 0
    self.bombs_dropped = 0
    self.save = 0

    if not os.path.exists("tables"):
        os.makedirs("tables")

    try:
        os.remove("stats.csv")
    except FileNotFoundError:
        pass


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

    features_old = self.last_features
    action_old = ACTION_TO_INDEX[self_action]
    features_new = state_to_features(new_game_state)
    action_new_P = np.array(list(map(lambda action: self.Q[features_new][action], ACTION_INDICES))).argmax()
    action_new_Q = np.array(list(map(lambda action: self.P[features_new][action], ACTION_INDICES))).argmax()

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

    rewards = reward_from_events(self, events)

    valQ = self.Q[features_old][action_old]
    valP = self.P[features_old][action_old]
    self.Q[features_old][action_old] = valQ + LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.P[features_new][action_new_Q] - self.Q[features_old][action_old])
    self.P[features_old][action_old] = valP + LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.Q[features_new][action_new_P] - self.P[features_old][action_old])

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(features_old, action_old, features_new, rewards))

    self.iteration += 1


def enemy_game_events_occurred(self, name, old_game_state, self_action, new_game_state, events):
    random_prob = TRANSITION_ENEMY_EPS_END + (TRANSITION_ENEMY_EPS_START - TRANSITION_ENEMY_EPS_END) * \
                  math.exp(-1. * self.iteration / TRANSITION_ENEMY_EPS_DECAY)
    if self.train and random.random() > random_prob:
        return

    if old_game_state is None or self_action is None:
        return

    features_old = state_to_features(old_game_state)
    action_old = ACTION_TO_INDEX[self_action]
    features_new = state_to_features(new_game_state)
    action_new_Q = np.array(list(map(lambda action: self.Q[features_new][action], ACTION_INDICES))).argmax()
    action_new_P = np.array(list(map(lambda action: self.P[features_new][action], ACTION_INDICES))).argmax()

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

    rewards = reward_from_events(self, events)

    valQ = self.Q[features_old][action_old]
    valP = self.P[features_old][action_old]
    self.Q[features_old][action_old] = valQ + LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.P[features_new][action_new_Q] - self.Q[features_old][action_old])
    self.P[features_old][action_old] = valP + LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.Q[features_new][action_new_P] - self.P[features_old][action_old])
    self.transitions.append(Transition(features_old, action_old, features_new, rewards))


def optimize(self):
    if len(self.transitions) < BATCH_SIZE + 1:
        return
    batch = self.transitions.sample(BATCH_SIZE)
    for transition in batch:
        features_old, action_old, features_new, rewards = transition
        action_new_Q = np.array(list(map(lambda action: self.Q[features_new][action], ACTION_INDICES))).argmax()
        action_new_P = np.array(list(map(lambda action: self.P[features_new][action], ACTION_INDICES))).argmax()
        valQ = self.Q[features_old][action_old]
        valP = self.P[features_old][action_old]
        self.Q[features_old][action_old] = valQ + LEARNING_RATE * (
                rewards + DISCOUNT_FACTOR * self.P[features_new][action_new_Q] - self.Q[features_old][action_old])
        self.P[features_old][action_old] = valP + LEARNING_RATE * (
                rewards + DISCOUNT_FACTOR * self.Q[features_new][action_new_P] - self.P[features_old][action_old])


CONVERSIONS = [
    {
        # nothing
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4
    },
    {
        # mirror left/right
        0: 0,
        1: 1,
        2: 3,
        3: 2,
        4: 4
    },
    {
        # mirror top/bottom
        0: 1,
        1: 0,
        2: 2,
        3: 3,
        4: 4
    },
    {
        # rotate 90deg
        0: 2,
        1: 3,
        2: 1,
        3: 0,
        4: 4
    },
    {
        # rotate 180deg
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        4: 4
    },
    {
        # rotate 270deg
        0: 3,
        1: 2,
        2: 0,
        3: 1,
        4: 4
    }
]


def transmute_neighbors(conversion, indices):
    new = [0, 0, 0, 0]
    new[conversion[0]] = indices[0]
    new[conversion[1]] = indices[1]
    new[conversion[2]] = indices[2]
    new[conversion[3]] = indices[3]
    return tuple(new)


def old_sync_symmetries(self):
    Q = self.Q
    new_Q = np.zeros_like(Q)

    for field1 in range(0, 3):
        for field2 in range(0, 3):
            for field3 in range(0, 3):
                for field4 in range(0, 3):
                    indices_neighbor_fields = (field1, field2, field3, field4)
                    for index_current_square in range(0, 4):
                        for index_coins in range(0, 5):
                            for index_crate in range(0, 5):
                                for index_enemy in range(0, 5):
                                    for action in ACTION_INDICES:
                                        value = Q[indices_neighbor_fields + (
                                            index_current_square, index_coins, index_crate, index_enemy, action)] / 6.0
                                        for conversion in CONVERSIONS:
                                            neighbor_fields = transmute_neighbors(conversion, indices_neighbor_fields)
                                            coins = conversion[index_coins]
                                            crate = conversion[index_crate]
                                            enemy = conversion[index_enemy]
                                            new_Q[
                                                neighbor_fields + (
                                                    index_current_square, coins, crate, enemy, action)] += value
    self.Q = new_Q


def sync_symmetries(Q):
    new_Q = np.zeros_like(Q)

    for field in range(0, 5):
        for index_current_square in range(0, 4):
            for index_coins in range(0, 5):
                for index_crate in range(0, 5):
                    for index_enemy in range(0, 5):
                        norm = 0
                        for action in ACTION_INDICES:
                            value = 0
                            for conversion in CONVERSIONS:
                                neighbor_fields = conversion[field]
                                coins = conversion[index_coins]
                                crate = conversion[index_crate]
                                enemy = conversion[index_enemy]
                                value += Q[(neighbor_fields, index_current_square, coins, crate, enemy, action)]
                            new_Q[(
                            field, index_current_square, index_coins, index_crate, index_enemy, action)] = value / 6.0
                            # norm += value * value
                        # factor = 1000 / np.sqrt(norm) if norm > 1 else 10
                        # for action in ACTION_INDICES:
                        #    new_Q[
                        #        (field, index_current_square, index_coins, index_crate, index_enemy, action)] *= factor
    return new_Q


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of the round, saves the current state of the Q-table so that it can be restored after training.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(
    #    Transition(state_to_features(last_game_state),
    #               ACTION_TO_INDEX[last_action],
    #               (2, 2, 2, 2, 1, 4, 4, 4), reward_from_events(self, events))
    # )
    self.transitions.append(
        Transition(state_to_features(last_game_state),
                   ACTION_TO_INDEX[last_action],
                   (4, 3, 4, 4, 4), reward_from_events(self, events))
    )

    for _ in range(EPOCHS_PER_ROUND):
        optimize(self)
    self.iteration += 1
    self.round += 1

    if self.round % SYMMETRY_SYNC_RATE == 0:
        # self.Q = sync_symmetries(self.Q)
        # self.P = sync_symmetries(self.P)
        pass

    if self.round % ROUNDS_PER_SAVE == 0:
        # Store the model
        self.save += 1
        with open("tables/" + "{:04d}".format(self.save) + "-q-table.pt", "wb") as file:
            pickle.dump(self.Q, file)

    prob_enemy_copy = TRANSITION_ENEMY_EPS_END + (TRANSITION_ENEMY_EPS_START - TRANSITION_ENEMY_EPS_END) * \
                      math.exp(-1. * self.iteration / TRANSITION_ENEMY_EPS_DECAY)
    prob_exploration = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.iteration / EPS_DECAY)

    prob_enemy_copy = "{:5.2f}".format(prob_enemy_copy * 100)
    prob_exploration = "{:5.2f}".format(prob_exploration * 100)

    bomb_reward = "{:6.2f}".format(BOMB_EPS_END +
                                   (BOMB_EPS_START - BOMB_EPS_END) * math.exp(-1. * self.iteration / BOMB_EPS_DECAY))

    points = last_game_state['self'][1]
    self.point_history.append(points)
    points = "{:2d}".format(points)
    min_points = "{:2d}".format(min(self.point_history))

    avg_points = "{:5.2f}".format(sum(self.point_history) / len(self.point_history), 2)

    bombs = "{:3d}".format(self.bombs_dropped)
    self.bomb_history.append(self.bombs_dropped)
    avg_bombs = "{:6.2f}".format(sum(self.bomb_history) / len(self.point_history), 2)

    current_round = '{:8d}'.format(self.round)

    current_time = time.time()
    rounds_per_second = len(self.time_history) / (current_time - self.time_history[0])
    minutes_per_five_thousand_rounds = 5000 / rounds_per_second / 60
    self.time_history.append(current_time)

    print("\033c", end="")
    print("Round \033[93m\033[1m" + current_round +
          "\033[0m; \033[93m" + '{:5.2f}'.format(rounds_per_second) + "it/s\033[0m; \033[93m" +
          '{:4.2f}'.format(minutes_per_five_thousand_rounds) + "m\033[0m per 5k it; avg_points=\033[92m\033[1m" +
          avg_points + "\033[0m; min_point=\033[92m" + min_points + "\033[0m; points=\033[92m" + points +
          "\033[0m; P(copy enemy)=\033[96m" + prob_enemy_copy + "%\033[0m; P(exploration)=\033[96m" + prob_exploration +
          "%\033[0m; avg_bombs=\033[91m\033[1m" + avg_bombs + "\033[0m; bombs=\033[91m" + bombs +
          "\033[0m; bomb reward=\033[91m" + bomb_reward + "\033[0m")
    self.bombs_dropped = 0

    if self.round % 10 == 0:
        with open("stats.csv", "a") as stats:
            stats.write(current_round + ", " + avg_points + ", " + points + ", " + avg_bombs + ", " + bombs + ", " +
                        prob_enemy_copy + ", " + prob_exploration + "\n")


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculated the rewards for a given action based on its event list.
    Rewards are defined in the global variable GAME_REWARDS.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1000,
        e.KILLED_OPPONENT: 5000,
        e.KILLED_SELF: -500,
        e.GOT_KILLED: -400,
        e.INVALID_ACTION: -100,
        e.WAITED: -20,
        e.BOMB_DROPPED: BOMB_EPS_END + (BOMB_EPS_START - BOMB_EPS_END) * math.exp(
            -1. * self.iteration / BOMB_EPS_DECAY),
        e.CRATE_DESTROYED: 200,
        e.COIN_FOUND: 50,
        MOVED_TOWARDS_COIN_EVENT: 100,
        MOVED_AWAY_FROM_COIN_EVENT: -100,
        ESCAPE_BOMB_EVENT: 500
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
