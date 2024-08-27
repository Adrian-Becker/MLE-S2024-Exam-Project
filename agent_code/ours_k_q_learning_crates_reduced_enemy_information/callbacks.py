import math
import os
import pickle
import random

import numpy as np
from features import state_to_features_for_q_learning as state_to_features

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
ACTION_INDICES = np.array([0, 1, 2, 3, 4, 5]).astype(int)
ACTION_TO_INDEX = {
    'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'WAIT': 4, 'BOMB': 5
}
ACTIONS_PROBABILITIES = [0.2, 0.2, 0.2, 0.2, 0.195, 0.005]

WIDTH = 17
HEIGHT = 17

TRAP_FLEEING_THRESHOLD = 2

"""
state vector: 
neighbor fields:                                                                                        03^4 (=81)
- 0 safe OR if current square will explode in the future and field is the shortest path away 
- 1 death/wall, 
- 2 unsafe (might explode in >1 turns)

current square: 											                                            04
- 0 no bomb (since dangerous/no benefit/no bomb available), 
- 1 dangerous need to move, 
- 2 bomb destroys one enemey/crate (AND escape path exists), 
- 3 bomb destroys multiple enemies/crates OR is guaranteed to kill an enemy (AND escape path exists)

coins: 0-3;4, 2^4=16 (UP, DOWN, LEFT, RIGHT) 0=not the shortest direction, 1=shortest direction		    05
crates: 0-3;4, 2^4=16 (UP, DOWN, LEFT, RIGHT) 								                            05
        a) no crates to destroy => 0=not the shortest direction, 1=shortest direction
        b) crates to destroy => 0=no improvement, 1=more crates to destroy		
enemies: 0-3;4, 2^4=16 (UP, DOWN, LEFT, RIGHT) 								                            05
        a) no enemies to destroy => 0=not the shortest direction, 1=shortest direction		
        b) enemies to destroy =>0=no improvement, 1=more enemies to destroy	

= 40500
"""
# FEATURE_SHAPE = (3, 3, 3, 3, 4, 5, 5, 5, len(ACTIONS))
FEATURE_SHAPE = (4, 2, 2, 2, 2, 4, 2, 2, 2, 2, len(ACTIONS))

EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 40000


def setup(self):
    """
    This is called once when loading each agent.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.iteration = 0
    self.round = 0
    if self.train or not os.path.isfile("tables/0001-q-table.pt"):
        self.logger.info("Setting up model from scratch.")
        self.Q = np.zeros(FEATURE_SHAPE).astype(np.float64)
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

        with open(f"tables/{file_prefix}-q-table.pt", "rb") as file:
            self.Q = pickle.load(file)
        with np.printoptions(threshold=np.inf):
            print(self.Q)
        print(f"Loaded {file_prefix}")


def determine_next_action(game_state: dict, Q) -> str:
    features = state_to_features(game_state)

    best_action_index = np.array(list(map(lambda action: Q[features][action], ACTION_INDICES))).argmax()

    return ACTIONS[best_action_index]


def act(self, game_state: dict) -> str:
    """
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

    self.last_features = state_to_features(game_state)
    best_action_index = np.array(list(map(lambda action: self.Q[self.last_features][action], ACTION_INDICES))).argmax()
    action = ACTIONS[best_action_index]
    if not self.train:
        print(f"{action} {self.last_features}")
        #print(self.Q[self.last_features])
        pass
    return action
