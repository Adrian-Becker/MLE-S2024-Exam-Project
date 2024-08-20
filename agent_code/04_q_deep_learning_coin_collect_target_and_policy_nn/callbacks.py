import os
import pickle
import random
from torch import nn
import torch

import numpy as np

from .helper_functions import distance_to_nearest_coins

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


def create_network():
    """return nn.Sequential(
        nn.Linear(4, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 16),
        nn.ReLU(),
        nn.Linear(16, len(ACTIONS))
    )"""
    return nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, len(ACTIONS))
    )


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.policy_model = create_network()
        self.target_model = create_network()
        self.target_model.load_state_dict(self.policy_model.state_dict())
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.policy_model = pickle.load(file)
            self.target_model = create_network()
            self.target_model.load_state_dict(self.policy_model.state_dict())


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    random_prob = .2
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2])

    self.logger.debug("Querying model for action.")
    with torch.no_grad():
        return ACTIONS[self.policy_model(torch.from_numpy(state_to_features(game_state)).to(torch.float32)).argmax()]


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
    #field = game_state['field']
    #for coin in game_state['coins']:
    #    field[coin] = 2
    #field[game_state['self'][3]] = 3
    #for other in game_state['others']:
    #    field[other[3]] = 4
    #field = field[1:16, 1:16]
    #channels.append(field.flatten())
    channels.append(distance_to_nearest_coins(game_state))

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)

    # and return them as a vector
    return stacked_channels.reshape(-1)
