import math
import os
import pickle
import random
from torch import nn
import torch

import numpy as np

import torch.nn.functional as F

from .helper_functions import distance_to_nearest_coins

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
ACTION_TO_INDEX = {
    'UP': 0,
    'RIGHT': 1,
    'DOWN': 2,
    'LEFT': 3,
    'WAIT': 4
}

EPS_START = 0.999
EPS_END = 0.001
EPS_DECAY = 1000


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(4, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, len(ACTIONS))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.iteration = 0
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_net = pickle.load(file)
            self.target_net = DQN()
            self.target_net.load_state_dict(self.policy_net.state_dict())


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    random_prob = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.iteration / EPS_DECAY)
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2])

    self.logger.debug("Querying model for action.")
    with torch.no_grad():
        action = ACTIONS[self.policy_net(state_to_features(game_state)).argmax()]
        return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []

    # whole grid
    # field = game_state['field']
    # for coin in game_state['coins']:
    #    field[coin] = 2
    # field[game_state['self'][3]] = 3
    # for other in game_state['others']:
    #    field[other[3]] = 4
    # field = field[1:16, 1:16]
    # channels.append(field.flatten())
    channels.append(distance_to_nearest_coins(game_state))

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = torch.stack(channels)

    # and return them as a vector
    return stacked_channels.reshape(-1)
