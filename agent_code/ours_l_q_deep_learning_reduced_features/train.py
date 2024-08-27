import math
from collections import namedtuple, deque

import pickle
import random
from typing import List

import numpy as np
import torch
from torch import nn

import os

import events as e
from .callbacks import state_to_features, ACTION_TO_INDEX, EPS_START, EPS_END, EPS_DECAY
from ..ours_k_q_learning_crates_reduced_enemy_information.callbacks import breadth_first_search, prepare_field_coins, \
    determine_explosion_timer
from ..ours_k_q_learning_crates_reduced_enemy_information.stats_logger import StatsLogger, Stat

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TRANSITION_ENEMY_EPS_START = 0.999
TRANSITION_ENEMY_EPS_END = 0.00
TRANSITION_ENEMY_EPS_DECAY = 40000

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions
MINI_BATCH_SIZE = 128
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

STEPS_BEFORE_SYNC = 10

LEARNING_RATE_OPTIMIZER = 1e-6
DISCOUNT_FACTOR = 0.9
TAU = 0.005

ROUNDS_PER_SAVE = 500

# Events
MOVED_TOWARDS_COIN_EVENT = "Moved Towards Coin"
MOVED_AWAY_FROM_COIN_EVENT = "Moved Away from Coin"
ESCAPE_BOMB_EVENT = "Escape Bomb"
WAITED_WITHOUT_NEED_EVENT = "Waited Without Need"

PLACED_BOMB_DESTROY_ONE_EVENT = "Placed Bomb Safely Destroy One"
PLACED_BOMB_DESTROY_MULTIPLE_EVENT = "Placed Bomb Safely Destroy Multiple"
PLACED_INESCAPABLE_BOMB_EVENT = "Placed Inescapable Bomb"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.iteration = 0
    self.round = 0
    self.bombs_dropped = 0
    self.save = 0
    self.total_rewards = 0
    self.iteration_per_round = 0
    self.invalid_moves = 0
    self.suicides = 0
    self.kills = 0
    self.current_loss = 0

    self.loss = nn.MSELoss()
    self.optimizer = torch.optim.RMSprop(
        self.policy_net.parameters())  # torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE_OPTIMIZER, amsgrad=True)

    self.stats_logger = StatsLogger([
        Stat('points', 'points', True, 500, '{:2d}', '{:5.2f}', 32, ' '),
        Stat('P(copy enemy)', 'p-enemy', False, 1, '{:5.2f}', '', 36, '%'),
        Stat('P(exploration)', 'p-explo', False, 1, '{:5.2f}', '', 36, '%'),
        Stat('bombs', 'bombs', True, 500, '{:3d}', '{:6.2f}', 31, ' '),
        Stat('rewards', 'rewards', True, 500, '{:7d}', '{:9.2f}', 32, ' '),
        Stat('iterations p. round', 'iteration', True, 500, '{:3d}', '{:6.2f}', 35, ' '),
        Stat('invalid moves', 'invalid', True, 500, '{:5.2f}', '{:5.2f}', 35, '%'),
        Stat('kills', 'kills', True, 500, '{:1d}', '{:4.2f}', 34, ' '),
        Stat('suicides', 'suicides', True, 500, '{:1d}', '{:4.2f}', 34, ' '),
        Stat('loss', 'loss', True, 500, '{:8.2f}', '{:8.2f}', 33, ' ')
    ])

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
        if features_old[0] == 2:
            events.append(PLACED_BOMB_DESTROY_ONE_EVENT)
        elif features_old[0] == 3:
            events.append(PLACED_BOMB_DESTROY_MULTIPLE_EVENT)
        elif features_old[0] == 0:
            events.append(PLACED_INESCAPABLE_BOMB_EVENT)

    if e.WAITED in events and (max(features_old[1:5]) > 0 or features_old[0] > 0):
        events.append(WAITED_WITHOUT_NEED_EVENT)


def train_batch(self):
    if len(self.transitions) >= MINI_BATCH_SIZE:
        optimize_network(self, random.sample(self.transitions, MINI_BATCH_SIZE))

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

def handle_event_occurrence(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    features_old = state_to_features(old_game_state)
    features_new = state_to_features(new_game_state)

    # Idea: Add your own events to hand out rewards
    add_custom_events(self, old_game_state, self_action, new_game_state, events, features_old, features_new)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(features_old, torch.Tensor([ACTION_TO_INDEX[self_action]]).to(torch.int64),
                   features_new, torch.Tensor([reward_from_events(self, events)])))

    # training
    train_batch(self)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

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
    if e.BOMB_DROPPED in events:
        self.bombs_dropped += 1

    self.total_rewards += reward_from_events(self, events)

def enemy_game_events_occurred(self, name, old_game_state, self_action, new_game_state, events):
    random_prob = TRANSITION_ENEMY_EPS_END + (TRANSITION_ENEMY_EPS_START - TRANSITION_ENEMY_EPS_END) * \
                  math.exp(-1. * self.iteration / TRANSITION_ENEMY_EPS_DECAY)
    if self.train and random.random() > random_prob:
        return

    if old_game_state is None or self_action is None:
        return

    handle_event_occurrence(self, old_game_state, self_action, new_game_state, events)


def optimize_network(self, batch):
    batch = Transition(*zip(*batch))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                         if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(MINI_BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    self.current_loss = loss

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # state_to_features is defined in callbacks.py
    features_old = state_to_features(last_game_state)
    self.transitions.append(
        Transition(features_old, torch.Tensor([ACTION_TO_INDEX[last_action]]).to(torch.int64),
                   torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), torch.Tensor([reward_from_events(self, events)])))
    # training
    train_batch(self)

    self.iteration += 1
    self.iteration_per_round += 1
    self.round += 1
    if self.round % ROUNDS_PER_SAVE == 0:
        # Store the model
        self.save += 1
        with open("tables/" + "{:04d}".format(self.save) + "-network.pt", "wb") as file:
            pickle.dump(self.policy_net, file)

    prob_enemy_copy = TRANSITION_ENEMY_EPS_END + (TRANSITION_ENEMY_EPS_START - TRANSITION_ENEMY_EPS_END) * \
                      math.exp(-1. * self.iteration / TRANSITION_ENEMY_EPS_DECAY)
    prob_exploration = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.iteration / EPS_DECAY)

    self.stats_logger.add('loss', float(self.current_loss))

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
        e.KILLED_SELF: -20000,
        PLACED_INESCAPABLE_BOMB_EVENT: -20000,
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
