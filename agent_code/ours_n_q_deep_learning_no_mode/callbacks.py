import math
import os
import pickle
import random
from torch import nn
import torch

import numpy as np

import torch.nn.functional as F

from features import mark_bomb, determine_explosion_timer, count_destroyable_crates_and_enemies, \
    determine_current_square, determine_current_square_vector, determine_escape_direction_vector, \
    prepare_escape_path_fields, determine_coin_value_vector, determine_is_worth_to_move_crates, \
    determine_crate_value_vector, determine_is_worth_to_move_enemies, determine_enemy_value_vector

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
ACTION_INDICES = np.array([0, 1, 2, 3, 4, 5]).astype(int)
ACTION_TO_INDEX = {
    'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'WAIT': 4, 'BOMB': 5
}
ACTIONS_PROBABILITIES = [0.2, 0.2, 0.2, 0.2, 0.195, 0.005]

EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 80000


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.forward1 = nn.Linear(20, 64)
        self.forward2 = nn.Linear(64, 64)
        self.forward3 = nn.Linear(64, len(ACTIONS))

    def forward(self, x):
        x = F.relu(self.forward1(x))
        x = F.relu(self.forward2(x))
        return F.relu(self.forward3(x))


def setup(self):
    """
     This is called once when loading each agent.
     :param self: This object is passed to all callbacks and you can set arbitrary values.
     """
    self.iteration = 0
    self.round = 0
    if self.train or not os.listdir("models"):
        self.logger.info("Setting up model from scratch.")
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
    else:
        self.logger.info("Loading model from saved state.")
        files = os.listdir("models")
        highest_number = 0
        file_prefix = "0001"
        for file in files:
            number = int(file[:4])
            if number > highest_number:
                highest_number = number
                file_prefix = file[:4]

        with open(f"models/{file_prefix}-network.pt", "rb") as file:
            self.policy_net = pickle.load(file)
            self.target_net = DQN()
            self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Loaded {file_prefix}")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    random_prob = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.iteration / EPS_DECAY)
    if self.train and random.random() < random_prob and self.round % 2 == 0:
        x, y = game_state['self'][3]
        explosion_timer = determine_explosion_timer(game_state)
        count_crates, count_enemies = count_destroyable_crates_and_enemies(x, y, game_state, explosion_timer)
        current_square = determine_current_square(x, y, game_state, count_crates + count_enemies)

        if current_square > 1 and random.random() > 0.5:
            return 'BOMB'

        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=ACTIONS_PROBABILITIES)

    self.logger.debug("Querying model for action.")
    with torch.no_grad():
        action = ACTIONS[self.policy_net(torch.stack([state_to_features(game_state)])).argmax()]
        return action


def state_to_features(game_state: dict):
    features = []

    x, y = game_state['self'][3]
    explosion_timer = determine_explosion_timer(game_state)
    bomb_input = prepare_escape_path_fields(game_state)

    count_crates, count_enemies = count_destroyable_crates_and_enemies(x, y, game_state, explosion_timer)
    features.extend(determine_current_square_vector(x, y, game_state, count_crates + count_enemies))

    if features[1] > 0:
        features.extend(determine_escape_direction_vector(x, y, game_state, bomb_input))
    else:
        features.extend([0, 0, 0, 0])

    features.extend(determine_coin_value_vector(x, y, game_state, explosion_timer))

    if count_crates > 0:
        features.extend(determine_is_worth_to_move_crates(x, y, game_state, count_crates, explosion_timer))
    else:
        features.extend(determine_crate_value_vector(x, y, game_state, explosion_timer))

    if count_enemies > 0:
        features.extend(determine_is_worth_to_move_enemies(x, y, game_state, count_enemies, explosion_timer))
    else:
        features.extend(determine_enemy_value_vector(x, y, game_state, explosion_timer))

    return torch.asarray(features).float()
