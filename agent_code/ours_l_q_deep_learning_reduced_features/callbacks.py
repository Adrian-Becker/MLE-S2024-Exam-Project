import math
import os
import pickle
import random
from torch import nn
import torch

import numpy as np

import torch.nn.functional as F

from ..ours_k_q_learning_crates_reduced_enemy_information.callbacks import state_to_features as state_to_features_prep

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
        self.layer1 = nn.Linear(10, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


def setup(self):
    """
     This is called once when loading each agent.
     :param self: This object is passed to all callbacks and you can set arbitrary values.
     """
    self.iteration = 0
    self.round = 0
    if self.train or not os.listdir("tables"):
        self.logger.info("Setting up model from scratch.")
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
    else:
        self.logger.info("Loading model from saved state.")
        files = os.listdir("tables")
        highest_number = 0
        file_prefix = "0001"
        for file in files:
            number = int(file[:4])
            if number > highest_number:
                highest_number = number
                file_prefix = file[:4]

        with open(f"tables/{file_prefix}-network.pt", "rb") as file:
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
        features = state_to_features(game_state)
        if features[0] > 1 and random.random() < 0.5:
            return 'BOMB'
        self.logger.debug("Choosing action purely at random.")
        self.last_features = features
        return np.random.choice(ACTIONS, p=ACTIONS_PROBABILITIES)

    self.logger.debug("Querying model for action.")
    with torch.no_grad():
        action = ACTIONS[self.policy_net(state_to_features(game_state)).argmax()]
        return action


def state_to_features(game_state: dict):
    return torch.as_tensor(state_to_features_prep(game_state)).float()
