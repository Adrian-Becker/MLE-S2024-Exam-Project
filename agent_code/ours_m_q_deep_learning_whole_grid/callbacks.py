import math
import os
import pickle
import random
from torch import nn
import torch

import numpy as np

import torch.nn.functional as F

from features import mark_bomb, determine_explosion_timer, count_destroyable_crates_and_enemies, \
    determine_current_square

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
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=1)
        self.flatten = nn.Flatten
        self.forward1 = nn.Linear(2704, 128)
        self.forward2 = nn.Linear(128, 64)
        self.forward3 = nn.Linear(64, len(ACTIONS))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = self.flatten(x)
        x = x.view(x.size(0), -1)
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
    stone_field = torch.as_tensor(np.clip(game_state['field'], -1, 0) * -1)
    crate_field = torch.as_tensor(np.clip(game_state['field'], 0, 1))

    coin_field = np.zeros_like(game_state['field'])
    for coin in game_state['coins']:
        coin_field[coin] = 1
    coin_field = torch.as_tensor(coin_field)

    enemy_field = np.zeros_like(game_state['field'])
    for other in game_state['others']:
        enemy_field[other[3]] = 1
    enemy_field = torch.as_tensor(enemy_field)

    self_field = np.zeros_like(game_state['field'])
    self_field[game_state['self'][3]] = 1
    self_field = torch.as_tensor(self_field)

    bomb_field = np.zeros_like(game_state['field'])
    for bomb in game_state['bombs']:
        bomb_field[bomb[0]] = 1
    bomb_field = torch.as_tensor(bomb_field)

    explosion_countdown_field = np.ones_like(game_state['field']) * 1000
    for bomb in game_state['bombs']:
        mark_bomb(game_state['field'], explosion_countdown_field, bomb[1] + 1, bomb[0], True)
    explosion_countdown_field[explosion_countdown_field == 1000] = 0
    explosion_countdown_field = torch.as_tensor(explosion_countdown_field)

    explosion_map = torch.as_tensor(game_state['explosion_map'])

    return torch.stack((
        stone_field, crate_field, coin_field, enemy_field, self_field,
        bomb_field, explosion_countdown_field, explosion_map
    )).float()
