import math
import os
import pickle
import random
import time
from collections import deque
from pyexpat import features
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, ACTION_TO_INDEX, ACTION_INDICES, EPS_START, EPS_END, EPS_DECAY
from features import breadth_first_search, prepare_field_coins, determine_explosion_timer, count_destroyable_crates, \
    determine_coin_value_scored_reward, determine_crate_value_scored_reward, prepare_escape_path_fields, \
    determine_escape_direction, determine_trap_escape_directions_improved, determine_trap_enemy_directions, \
    determine_trap_filter, determine_is_worth_to_move_crates_scored_reward, count_destroyable_enemies, \
    determine_enemy_value_scored_reward, determine_is_worth_to_move_enemies_trap_filter

from rewards import distance_to_nearest_bomb

from .history import TransitionHistory, Transition

import tqdm

from stats_logger import StatsLogger, Stat

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions

TRANSITION_ENEMY_EPS_START = 0.999
TRANSITION_ENEMY_EPS_END = 0.00
TRANSITION_ENEMY_EPS_DECAY = 4000

BATCH_SIZE = 128
EPOCHS_PER_ROUND = 10

# Events
FLEEING_CORRECTLY_EVENT = "Fleeing Correctly"
NOT_FLEEING_CORRECTLY_EVENT = "Not Fleeing Correctly"

FLEEING_TRAP_CORRECTLY_EVENT = "Fleeing Trap Correctly"
NOT_FLEEING_TRAP_CORRECTLY_EVENT = "Not Fleeing Trap Correcly"

MOVED_TOWARDS_TRAP_EVENT = "Moved towards trap event"
NOT_MOVED_TOWARDS_TRAP_EVENT = "Not Moved towards trap event"

PLACED_BOMB_TRAPPED_ENEMY_EVENT = "Placed Bomb Trapped Enemy Event"
NOT_PLACED_BOMB_TRAPPED_ENEMY_EVENT = "Not Placed Bomb Trapped Enemy Event"

PLACED_BOMB_DESTROY_ONE_EVENT = "Placed Bomb Safely Destroy One"
PLACED_BOMB_DESTROY_MULTIPLE_EVENT = "Placed Bomb Safely Destroy Multiple"
NOT_PLACED_BOMB_DESTROY_ONE_EVENT = "Not Placed Bomb Safely Destroy One"
NOT_PLACED_BOMB_DESTROY_MULTIPLE_EVENT = "Not Placed Bomb Safely Destroy Multiple"
PLACED_UNSAFE_BOMB_EVENT = "Paced Unsafe Bomb"

MOVED_TOWARDS_COIN_EVENT = "Moved Towards Coin"
NOT_MOVED_TOWARDS_COIN_EVENT = "Not Moved Towards Coin"

MOVED_TOWARDS_CRATE_EVENT = "Moved Towards Crate"
NOT_MOVED_TOWARDS_CRATE_EVENT = "Not Moved Towards Crate"
CRATE_PLACED_BOMB = "Placed Bomb Crate"
NOT_CRATE_PLACED_BOMB = "Not Placed Bomb Crate"

MOVED_TOWARDS_ENEMY_EVENT = "Moved Towards Enemy"
NOT_MOVED_TOWARDS_ENEMY_EVENT = "Not Moved Towards Enemy"
ENEMY_PLACED_BOMB = "Placed Bomb Enemy"
NOT_ENEMY_PLACED_BOMB = "Not Placed Bomb Enemy"

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9

SYMMETRY_SYNC_RATE = 5

ROUNDS_PER_SAVE = 1000

CONVERSIONS = [
    {
        # nothing
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6
    },
    {
        # mirror left/right
        0: 0,
        1: 1,
        2: 3,
        3: 2,
        4: 4,
        5: 5,
        6: 6
    },
    {
        # mirror top/bottom
        0: 1,
        1: 0,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6
    },
    {
        # rotate 90deg
        0: 2,
        1: 3,
        2: 1,
        3: 0,
        4: 4,
        5: 5,
        6: 6
    },
    {
        # rotate 180deg
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        4: 4,
        5: 5,
        6: 6
    },
    {
        # rotate 270deg
        0: 3,
        1: 2,
        2: 0,
        3: 1,
        4: 4,
        5: 5,
        6: 6
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
        Stat('points', 'points', True, 50, '{:2d}', '{:5.2f}', 32, ' '),
        Stat('P(copy enemy)', 'p-enemy', False, 1, '{:5.2f}', '', 36, '%'),
        Stat('P(exploration)', 'p-explo', False, 1, '{:5.2f}', '', 36, '%'),
        Stat('bombs', 'bombs', True, 50, '{:3d}', '{:6.2f}', 31, ' '),
        Stat('rewards', 'rewards', True, 50, '{:12d}', '{:14.2f}', 32, ' '),
        Stat('iterations p. round', 'iteration', True, 50, '{:3d}', '{:6.2f}', 35, ' '),
        Stat('invalid moves', 'invalid', True, 50, '{:5.2f}', '{:5.2f}', 35, '%'),
        Stat('kills', 'kills', True, 50, '{:1d}', '{:4.2f}', 34, ' '),
        Stat('suicides', 'suicides', True, 50, '{:1d}', '{:4.2f}', 34, ' '),
        Stat('repeated fields', 'repeated', True, 50, '{:5.2f}', '{:5.2f}', 31, '%'),
        Stat('positive reward', 'p-reward', True, 50, '{:3d}', '{:6.2f}', 35, ' '),
        Stat('negative reward', 'n-reward', True, 50, '{:3d}', '{:6.2f}', 35, ' ')
    ])

    self.round = 0
    self.bombs_dropped = 0
    self.save = 0
    self.total_rewards = 0
    self.iteration_per_round = 0
    self.invalid_moves = 0
    self.suicides = 0
    self.kills = 0
    self.repeated = 0
    self.positive = 0
    self.negative = 0
    self.field_history = deque(maxlen=4)

    if not os.path.exists("tables"):
        os.makedirs("tables")

    try:
        os.remove("stats.csv")
    except FileNotFoundError:
        pass


def add_custom_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str],
                      features_old, features_new):
    # used for computation later on
    explosion_timer = determine_explosion_timer(old_game_state)
    x, y = old_game_state["self"][3]
    current_square = features_old[0]
    escape_direction = features_old[1]
    trap_escape_direction = features_old[2]
    coin_direction = features_old[3]
    crate_direction = features_old[4]
    enemy_direction = features_old[5]
    trap_enemy_direction = features_old[6]
    can_move_up = features_old[7]
    can_move_down = features_old[8]
    can_move_left = features_old[9]
    can_move_right = features_old[10]

    # convert the action we are doing into an index
    # -1: invalid
    # 0: up, 1: down, 2: left, 3: right
    # 4: wait
    # 5: bomb
    action_index = -1
    if e.MOVED_UP in events:
        action_index = 0
    elif e.MOVED_DOWN in events:
        action_index = 1
    elif e.MOVED_LEFT in events:
        action_index = 2
    elif e.MOVED_RIGHT in events:
        action_index = 3
    elif e.WAITED in events:
        action_index = 4
    elif e.BOMB_DROPPED in events:
        action_index = 5

    # we want to give priorities in the following way:
    # has to flee from bomb/trap !!!=> only care about this, ignore everything else
    # can collect coin => only care about coin (and maybe exploding things)
    # can't collect coin but move towards crate/explode crate => only care about crate (and maybe exploding other stuff)
    # can't collect coin + move towards crate/explode crate but move towards enemy => only care about enemy

    # check if we placed a bomb that will kill us
    # CORRESPONDING EVENTS: PLACED_UNSAFE_BOMB_EVENT
    if current_square == 0 and action_index == 5:
        events.append(PLACED_UNSAFE_BOMB_EVENT)

    # we need flee from a bomb!!! ignore everything else
    # CORRESPONDING EVENTS: FLEEING_CORRECTLY_EVENT; NOT_FLEEING_CORRECTLY_EVENT
    if current_square == 1:
        possible_escape_directions = determine_escape_direction(x, y, old_game_state,
                                                                prepare_escape_path_fields(old_game_state))
        if possible_escape_directions.max() > 0:
            if action_index >= 4 or action_index == -1:
                events.append(NOT_FLEEING_CORRECTLY_EVENT)
            else:
                events.append(FLEEING_CORRECTLY_EVENT
                              if possible_escape_directions[action_index] == 1 else NOT_FLEEING_CORRECTLY_EVENT)
        return

    # we need to flee from a trap!!! ignore everything else, we first check based on the feature if we need to flee to
    # save computation
    # CORRESPONDING EVENTS: FLEEING_TRAP_CORRECTLY_EVENT; NOT_FLEEING_TRAP_CORRECTLY_EVENT
    if trap_escape_direction != 4:
        _, trap_fleeing_directions = determine_trap_escape_directions_improved(old_game_state, explosion_timer)
        if trap_fleeing_directions.max() > 0:
            if action_index >= 4 or action_index == -1:
                events.append(NOT_FLEEING_TRAP_CORRECTLY_EVENT)
            else:
                events.append(FLEEING_TRAP_CORRECTLY_EVENT
                              if trap_fleeing_directions[action_index] == 1 else NOT_FLEEING_TRAP_CORRECTLY_EVENT)
        return

    # check if we can trap an enemy, to save computation, we first check based on the features if there even exists such
    # a direction
    # CORRESPONDING EVENTS: MOVED_TOWARDS_TRAP_EVENT; NOT_MOVED_TOWARDS_TRAP_EVENT;
    #                       PLACED_BOMB_TRAPPED_ENEMY_EVENT; NOT_PLACED_BOMB_TRAPPED_ENEMY_EVENT
    if trap_enemy_direction != 4:
        # we should place a bomb to (hopefully) kill an enemy
        if trap_enemy_direction == 5:
            events.append(PLACED_BOMB_TRAPPED_ENEMY_EVENT if action_index == 5 else NOT_PLACED_BOMB_TRAPPED_ENEMY_EVENT)
        elif trap_enemy_direction == 6:
            events.append(MOVED_TOWARDS_TRAP_EVENT if action_index == 4 else NOT_MOVED_TOWARDS_TRAP_EVENT)
        elif action_index >= 4 or action_index == -1:
            events.append(NOT_MOVED_TOWARDS_TRAP_EVENT)
        else:
            directions = determine_trap_enemy_directions(old_game_state, explosion_timer)
            events.append(MOVED_TOWARDS_TRAP_EVENT if directions[action_index] == 1 else NOT_MOVED_TOWARDS_TRAP_EVENT)

    # check if placing a bomb might be a good idea, for this we use a feature to save computation instead of calling the
    # corresponding function
    # CORRESPONDING EVENTS: PLACED_BOMB_DESTROY_ONE_EVENT; NOT_PLACED_BOMB_DESTROY_ONE_EVENT;
    #                       PLACED_BOMB_DESTROY_MULTIPLE_EVENT; NOT_PLACED_BOMB_DESTROY_MULTIPLE_EVENT
    if current_square > 1:
        if current_square == 2:
            events.append(PLACED_BOMB_DESTROY_ONE_EVENT if action_index == 5
                          else NOT_PLACED_BOMB_DESTROY_ONE_EVENT)
        else:
            events.append(PLACED_BOMB_DESTROY_MULTIPLE_EVENT if action_index == 5
                          else NOT_PLACED_BOMB_DESTROY_MULTIPLE_EVENT)

    # check if there are coins we can move to, to save computation, we first check based on the features if there even
    # exists such a direction
    # CORRESPONDING EVENTS: MOVED_TOWARDS_COIN_EVENT; NOT_MOVED_TOWARDS_COIN_EVENT
    if coin_direction != 4:
        if action_index >= 4 or action_index == -1:
            events.append(NOT_MOVED_TOWARDS_COIN_EVENT)
        else:
            # used to filter out directions that may get us trapped
            trap_filter = determine_trap_filter(old_game_state, explosion_timer)

            directions = determine_coin_value_scored_reward(*old_game_state['self'][3], old_game_state, explosion_timer,
                                                            trap_filter)
            if directions.max() >= 1:
                if directions[action_index] == 1:
                    events.append(MOVED_TOWARDS_COIN_EVENT)
                else:
                    events.append(NOT_MOVED_TOWARDS_COIN_EVENT)
        return

    # check if there are crates we can move to, to save computation, we first check based on the features if there even
    # exists such a direction
    # CORRESPONDING EVENTS: MOVED_TOWARDS_CRATE_EVENT; NOT_MOVED_TOWARDS_CRATE_EVENT;
    #                       CRATE_PLACED_BOMB; NOT_CRATE_PLACED_BOMB
    if crate_direction != 4:
        if crate_direction == 5:
            # can place bomb to destroy crates
            events.append(CRATE_PLACED_BOMB if action_index == 5 else NOT_CRATE_PLACED_BOMB)
        elif action_index >= 4 or action_index == -1:
            events.append(NOT_MOVED_TOWARDS_CRATE_EVENT)
        else:
            # used to filter out directions that may get us trapped
            trap_filter = determine_trap_filter(old_game_state, explosion_timer)

            count_crates = count_destroyable_crates(x, y, old_game_state, explosion_timer)

            if current_square <= 1 or count_crates == 0:
                directions = determine_crate_value_scored_reward(*old_game_state['self'][3], old_game_state,
                                                                 explosion_timer, trap_filter)
            else:
                directions = determine_is_worth_to_move_crates_scored_reward(x, y, old_game_state, count_crates,
                                                                             explosion_timer, trap_filter)

            if directions.max() >= 1:
                if directions[action_index] == 1:
                    events.append(MOVED_TOWARDS_CRATE_EVENT)
                else:
                    events.append(NOT_MOVED_TOWARDS_CRATE_EVENT)
        return

    # check if there are enemies we can move to, to save computation, we first check based on the features if there even
    # exists such a direction
    # CORRESPONDING EVENTS: MOVED_TOWARDS_ENEMY_EVENT; NOT_MOVED_TOWARDS_ENEMY_EVENT
    #                       ENEMY_PLACED_BOMB; NOT_ENEMY_PLACED_BOMB
    if enemy_direction != 4:
        if enemy_direction == 5:
            # can place bomb to destroy enemies
            events.append(ENEMY_PLACED_BOMB if action_index == 5 else NOT_ENEMY_PLACED_BOMB)
        elif action_index >= 4 or action_index == -1:
            events.append(NOT_MOVED_TOWARDS_ENEMY_EVENT)
        else:
            # used to filter out directions that may get us trapped
            trap_filter = determine_trap_filter(old_game_state, explosion_timer)

            count_enemies = count_destroyable_enemies(x, y, old_game_state, explosion_timer)
            if current_square <= 1 or count_enemies == 0:
                directions = determine_enemy_value_scored_reward(x, y, old_game_state, explosion_timer, trap_filter)
            else:
                directions = determine_is_worth_to_move_enemies_trap_filter(x, y, old_game_state, count_enemies,
                                                                            explosion_timer, trap_filter)

            if directions.max() >= 1:
                if directions[action_index] == 1:
                    events.append(MOVED_TOWARDS_ENEMY_EVENT)
                else:
                    events.append(NOT_MOVED_TOWARDS_ENEMY_EVENT)
        return


def convert_features(conversion, features, action):
    return (features[0], conversion[features[1]], conversion[features[2]], conversion[features[3]],
            conversion[features[4]], conversion[features[5]], conversion[features[6]]) + \
        transmute_neighbors(conversion, features[7: 11]) + (conversion[action],)


def learning_step(self, transition: Transition):
    features_old, action_old, features_new, rewards = transition
    action_new_Q = np.array(list(map(lambda action: self.Q[features_new][action], ACTION_INDICES))).argmax()
    action_new_P = np.array(list(map(lambda action: self.P[features_new][action], ACTION_INDICES))).argmax()
    valQ = self.Q[features_old][action_old]
    valP = self.P[features_old][action_old]

    result_Q = valQ + LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.P[features_new][action_new_Q] - self.Q[features_old][action_old])
    result_P = valP + LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.Q[features_new][action_new_P] - self.P[features_old][action_old])

    self.Q[features_old][action_old] = result_Q
    self.P[features_old][action_old] = result_P

    for conversion in CONVERSIONS:
        converted_features = convert_features(conversion, features_old, action_old)
        self.Q[converted_features] = result_Q
        self.P[converted_features] = result_P


def handle_event_occurrence(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str],
                            is_us=False):
    features_old = self.last_features
    action_old = ACTION_TO_INDEX[self_action]
    features_new = state_to_features(new_game_state, action_old)

    add_custom_events(self, old_game_state, self_action, new_game_state, events, features_old, features_new)

    rewards = reward_from_events(self, events)

    # state_to_features is defined in callbacks.py
    transition = Transition(features_old, action_old, features_new, rewards)
    # learning_step(self, transition)
    self.transitions.append(transition)

    return rewards


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

    rewards = handle_event_occurrence(self, old_game_state, self_action, new_game_state, events, True)
    self.total_rewards += rewards
    if rewards > 0:
        self.positive += 1
    elif rewards < 0:
        self.negative += 1
        # print(self.last_features, self_action, events)

    self.iteration += 1
    self.iteration_per_round += 1
    if e.KILLED_OPPONENT in events:
        self.kills += 1
    if e.INVALID_ACTION in events:
        self.invalid_moves += 1
    if e.BOMB_DROPPED in events:
        self.bombs_dropped += 1


def enemy_game_events_occurred(self, name, old_game_state, self_action, new_game_state, events):
    return
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

    direction_values = [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 1),
        (0, 1, 1, 0),
        (0, 1, 1, 1),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 1, 0),
        (1, 0, 1, 1),
        (1, 1, 0, 0),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
        (1, 1, 1, 1)
    ]

    for current_field in range(4):
        for escape_direction in range(5):
            for trap_direction in range(5):
                for coin_direction in range(5):
                    for crate_direction in range(5):
                        for enemy_direction in range(5):
                            for priority_maker in range(5):
                                for trap_setting_direction in range(6):
                                    for direction in direction_values:
                                        for last_action in range(5):
                                            for action in ACTION_INDICES:
                                                value = 0
                                                for conversion in CONVERSIONS:
                                                    c_escape_direction = conversion[escape_direction]
                                                    c_trap_direction = conversion[trap_direction]
                                                    c_coin_direction = conversion[coin_direction]
                                                    c_crate_direction = conversion[crate_direction]
                                                    c_enemy_direction = conversion[enemy_direction]
                                                    c_trap_setting_direction = conversion[trap_setting_direction]
                                                    c_direction = transmute_neighbors(conversion, direction)
                                                    c_last_action = conversion[last_action]
                                                    c_action = conversion[action]

                                                    value += \
                                                        Q[current_field][c_escape_direction][c_trap_direction][
                                                            c_coin_direction] \
                                                            [c_crate_direction][c_enemy_direction][priority_maker][
                                                            c_trap_setting_direction][c_direction][c_last_action][
                                                            c_action]
                                                new_Q[current_field][escape_direction][trap_direction][coin_direction] \
                                                    [crate_direction][enemy_direction][priority_maker][
                                                    trap_setting_direction][direction][last_action][
                                                    action] = value / 6.0
    return new_Q


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of the round, saves the current state of the Q-table so that it can be restored after training.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.transitions.append(
        Transition(self.last_features,
                   ACTION_TO_INDEX[last_action],
                   (3, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0), reward_from_events(self, events))
    )

    for _ in range(EPOCHS_PER_ROUND):
        optimize(self)
    self.iteration += 1
    self.iteration_per_round += 1
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

    self.stats_logger.add('p-enemy', prob_enemy_copy * 100)
    self.stats_logger.add('p-explo', prob_exploration * 100)

    if self.round % 2 == 0:
        # only track non exploration rounds
        self.stats_logger.add('points', last_game_state['self'][1])
        self.stats_logger.add('bombs', self.bombs_dropped)
        self.stats_logger.add('invalid', self.invalid_moves / self.iteration_per_round * 100)
        self.stats_logger.add('iteration', self.iteration_per_round)
        self.stats_logger.add('rewards', self.total_rewards)
        self.stats_logger.add('suicides', 1 if e.KILLED_SELF in events else 0)
        self.stats_logger.add('kills', self.kills)
        self.stats_logger.add('repeated', self.repeated / self.iteration_per_round * 100)
        self.stats_logger.add('p-reward', self.positive)
        self.stats_logger.add('n-reward', self.negative)

        self.average_iterations = (sum(self.stats_logger.histories['iteration']) /
                                   len(self.stats_logger.histories['iteration']))

    self.bombs_dropped = 0
    self.invalid_moves = 0
    self.iteration_per_round = 0
    self.total_rewards = 0
    self.kills = 0
    self.repeated = 0
    self.positive = 0
    self.negative = 0

    self.stats_logger.output(self.round, self.iteration)


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculated the rewards for a given action based on its event list.
    Rewards are defined in the global variable GAME_REWARDS.
    """
    """game_rewards = {
        # e.KILLED_OPPONENT: 50000,
        # ?
        e.KILLED_SELF: -100000,
        # ?
        e.GOT_KILLED: -100000,

        # MOVEMENT related events
        e.INVALID_ACTION: -10000,
        REPEATED_FIELD_EVENT: -4000,
        # WAITED_WITHOUT_NEED_EVENT: -15000,
        # maybe lower?
        e.WAITED: -5000,

        # COIN related events
        e.COIN_COLLECTED: 10000,
        MOVED_TOWARDS_COIN_EVENT: 6000,
        MOVED_AWAY_FROM_COIN_EVENT: -8000,

        MOVED_TOWARDS_CRATE_EVENT: 3000,
        MOVED_AWAY_FROM_CRATE_EVENT: -5000,

        # e.COIN_FOUND: 1000,
        # INCREASED_DESTROYABLE_CRATES_COUNT: 3000,
        # DECREASED_DESTROYABLE_CRATES_COUNT: -3000,

        # BOMB related events
        PLACED_BOMB_DESTROY_ONE_EVENT: 1000,
        PLACED_BOMB_DESTROY_MULTIPLE_EVENT: 2000,
        #PLACE_BOMB_TARGET_CRATE_EVENT: 2000,
        PLACED_UNSAFE_BOMB_EVENT: -100000,
        e.BOMB_DROPPED: 2000,
        ESCAPE_BOMB_EVENT: 5000,
        NOT_FLEEING_CORRECTLY_EVENT: -10000,
        PLACED_BOMB_TRAPPED_ENEMY_EVENT: 150000,

        MOVED_TOWARDS_TRAP_EVENT: 150000
        #MOVED_AWAY_FROM_BOMB_EVENT: 500,
        #MOVED_TOWARDS_BOMB_EVENT: - 500,
        # e.CRATE_DESTROYED: 1000,
    }"""
    """game_rewards = {
        e.KILLED_SELF: -150,
        e.GOT_KILLED: -50,
        e.INVALID_ACTION: -25,
        REPEATED_FIELD_EVENT: -5,
        MOVED_AWAY_FROM_BOMB_EVENT: 6,
        e.WAITED: -4,
        e.COIN_COLLECTED: +50,
        MOVED_TOWARDS_COIN_EVENT: +10,
        MOVED_AWAY_FROM_COIN_EVENT: -10,
        MOVED_TOWARDS_CRATE_EVENT: +5,
        MOVED_AWAY_FROM_CRATE_EVENT: -5,
        PLACED_BOMB_DESTROY_ONE_EVENT: +20,
        PLACED_BOMB_DESTROY_MULTIPLE_EVENT: +40,
        PLACED_UNSAFE_BOMB_EVENT: -50,
        e.BOMB_DROPPED: +5,
        ESCAPE_BOMB_EVENT: +30,
        NOT_FLEEING_CORRECTLY_EVENT: -30,
        PLACED_BOMB_TRAPPED_ENEMY_EVENT: +50,
        MOVED_TOWARDS_TRAP_EVENT: 50,
    }"""
    game_rewards = {
        FLEEING_CORRECTLY_EVENT: 100,
        NOT_FLEEING_CORRECTLY_EVENT: -1000,

        FLEEING_TRAP_CORRECTLY_EVENT: 100,
        NOT_FLEEING_TRAP_CORRECTLY_EVENT: -1000,

        MOVED_TOWARDS_TRAP_EVENT: 1000000,
        NOT_MOVED_TOWARDS_TRAP_EVENT: -10000000,

        PLACED_BOMB_TRAPPED_ENEMY_EVENT: 5000000,
        NOT_PLACED_BOMB_TRAPPED_ENEMY_EVENT: -50000000,

        # TODO IMPORTANT CHANGE THIS WHEN TRAINING FOR CLASSIC MODE
        PLACED_BOMB_DESTROY_ONE_EVENT: 10,
        PLACED_BOMB_DESTROY_MULTIPLE_EVENT: 2000,
        NOT_PLACED_BOMB_DESTROY_ONE_EVENT: -10,
        NOT_PLACED_BOMB_DESTROY_MULTIPLE_EVENT: -20000,
        PLACED_UNSAFE_BOMB_EVENT: -1000000000,

        MOVED_TOWARDS_COIN_EVENT: 100,
        NOT_MOVED_TOWARDS_COIN_EVENT: -1000,

        MOVED_TOWARDS_CRATE_EVENT: 30000,
        NOT_MOVED_TOWARDS_CRATE_EVENT: -300000,
        CRATE_PLACED_BOMB: 30000,
        NOT_CRATE_PLACED_BOMB: -300000,

        MOVED_TOWARDS_ENEMY_EVENT: 30000,
        NOT_MOVED_TOWARDS_ENEMY_EVENT: -300000,
        ENEMY_PLACED_BOMB: 30000,
        NOT_ENEMY_PLACED_BOMB: -300000
    }
    # game_rewards = {
    #    FOLLOWED_MARKER_EVENT: 1000,
    #    DID_NOT_FOLLOW_MARKER_EVENT: -1000
    # }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    # if reward_sum < 0:
    #    print(events)

    # if FOLLOWED_MARKER_EVENT in events and reward_sum < 0:
    #    print(f"Good move, negative reward {events}")
    # if DID_NOT_FOLLOW_MARKER_EVENT in events and reward_sum > 0:
    #    print(f"Bad move, positive reward {events}")

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
