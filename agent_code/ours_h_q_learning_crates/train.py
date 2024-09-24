import math
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
TRANSITION_ENEMY_EPS_END = 0.01
TRANSITION_ENEMY_EPS_DECAY = 10000

BATCH_SIZE = 128
EPOCHS_PER_ROUND = 3

# Events
MOVED_TOWARDS_COIN_EVENT = "Moved Towards Coin"
MOVED_AWAY_FROM_COIN_EVENT = "Moved Away from Coin"
ESCAPE_BOMB_EVENT = "Escape Bomb"

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99

SYMMETRY_SYNC_RATE = 5

ROUNDS_PER_SAVE = 100

GAME_REWARDS = {
    e.COIN_COLLECTED: 1000,
    e.KILLED_OPPONENT: 5000,
    e.KILLED_SELF: -2000,
    e.GOT_KILLED: -500,
    e.INVALID_ACTION: -100,
    e.WAITED: -20,
    e.BOMB_DROPPED: 100,
    e.CRATE_DESTROYED: 200,
    e.COIN_FOUND: 50,
    MOVED_TOWARDS_COIN_EVENT: 100,
    MOVED_AWAY_FROM_COIN_EVENT: -100,
    ESCAPE_BOMB_EVENT: 500
}


def setup_training(self):
    print(math.inf > math.inf)
    """
    Initialise agent for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.transitions = TransitionHistory(TRANSITION_HISTORY_SIZE)
    self.point_history = deque([0], maxlen=100)
    self.round = 0
    self.start_time = time.time()


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

    features_old = state_to_features(old_game_state)
    action_old = ACTION_TO_INDEX[self_action]
    features_new = state_to_features(new_game_state)
    action_new = np.array(list(map(lambda action: self.Q[features_new][action], ACTION_INDICES))).argmax()

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

    rewards = reward_from_events(self, events)

    self.Q[features_old][action_old] += LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.Q[features_new][action_new] - self.Q[features_old][action_old])

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(features_old, action_old, features_new, rewards))

    self.iteration += 1


def enemy_game_events_occurred(self, name, old_game_state, self_action, new_game_state, events):
    return
    random_prob = TRANSITION_ENEMY_EPS_END + (TRANSITION_ENEMY_EPS_START - TRANSITION_ENEMY_EPS_END) * \
                  math.exp(-1. * self.iteration / TRANSITION_ENEMY_EPS_DECAY)
    if self.train and random.random() > random_prob:
        return

    if old_game_state is None or self_action is None:
        return

    features_old = state_to_features(old_game_state)
    action_old = ACTION_TO_INDEX[self_action]
    features_new = state_to_features(new_game_state)
    action_new = np.array(list(map(lambda action: self.Q[features_new][action], ACTION_INDICES))).argmax()

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

    self.Q[features_old][action_old] += LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.Q[features_new][action_new] - self.Q[features_old][action_old])

    self.transitions.append(Transition(features_old, action_old, features_new, rewards))


def optimize(self):
    if len(self.transitions) < BATCH_SIZE + 1:
        return
    batch = self.transitions.sample(BATCH_SIZE)
    for transition in batch:
        features_old, action_old, features_new, rewards = transition
        action_new = np.array(list(map(lambda action: self.Q[features_new][action], ACTION_INDICES))).argmax()
        self.Q[features_old][action_old] += LEARNING_RATE * (rewards +
                                                             DISCOUNT_FACTOR * self.Q[features_new][action_new] -
                                                             self.Q[features_old][action_old])


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


def sync_symmetries(self):
    Q = self.Q
    new_Q = np.zeros_like(Q)

    for field in range(0, 5):
        for index_current_square in range(0, 4):
            for index_coins in range(0, 5):
                for index_crate in range(0, 5):
                    for index_enemy in range(0, 5):
                        for action in ACTION_INDICES:
                            value = Q[(
                                field, index_current_square, index_coins, index_crate, index_enemy, action)] / 6.0
                            for conversion in CONVERSIONS:
                                neighbor_fields = conversion[field]
                                coins = conversion[index_coins]
                                crate = conversion[index_crate]
                                enemy = conversion[index_enemy]
                                new_Q[(neighbor_fields, index_current_square, coins, crate, enemy, action)] += value
    self.Q = new_Q


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
        #sync_symmetries(self)
        pass

    if self.round % ROUNDS_PER_SAVE == 0:
        # Store the model
        with open("q-table_no_history.pt", "wb") as file:
            pickle.dump(self.Q, file)

    prob_enemy_copy = TRANSITION_ENEMY_EPS_END + (TRANSITION_ENEMY_EPS_START - TRANSITION_ENEMY_EPS_END) * \
                      math.exp(-1. * self.iteration / TRANSITION_ENEMY_EPS_DECAY)
    prob_exploration = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.iteration / EPS_DECAY)

    prob_enemy_copy = str(round(prob_enemy_copy * 100, 2))
    prob_exploration = str(round(prob_exploration * 100, 2))

    points = last_game_state['self'][1]
    self.point_history.append(points)

    avg_points = str(round(sum(list(self.point_history)) / len(self.point_history), 2))

    if self.round % 10 == 0:
        with open("stats.csv", "a") as file:
            file.write(str(int(time.time() - self.start_time)) + ", " + str(self.round) + ", " + avg_points + "\n")

    print(
        f"Round \033[93m\033[1m{self.round}\033[0m of 70000; avg_points=\033[92m\033[1m{avg_points}\033[0m; points=\033[92m{points}\033[0m; P(copy enemy)=\033[96m{prob_enemy_copy}%\033[0m; P(exploration)=\033[96m{prob_exploration}%\033[0m")


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculated the rewards for a given action based on its event list.
    Rewards are defined in the global variable GAME_REWARDS.
    """
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum