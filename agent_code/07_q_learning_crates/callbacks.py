import math
import os
import pickle
import random

import numpy as np

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT']
ACTION_INDICES = np.array([0, 1, 2, 3, 4]).astype(int)
ACTION_TO_INDEX = {
    'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3, 'WAIT': 4, 'BOMB': 5
}
ACTIONS_PROBABILITIES = [0.2, 0.2, 0.2, 0.2, 0.2]

WIDTH = 17
HEIGHT = 17

"""
state vector: 
coins: 0-3;4, 2^4=16 (UP, DOWN, LEFT, RIGHT) 0=not the shortest direction, 1=shortest direction		    05
crates: 0-3;4, 2^4=16 (UP, DOWN, LEFT, RIGHT) 								                            05
        a) no crates to destroy => 0=not the shortest direction, 1=shortest direction
        b) crates to destroy => 0=no improvement, 1=more crates to destroy		
enemies: 0-3;4, 2^4=16 (UP, DOWN, LEFT, RIGHT) 								                            05
        a) no enemies to destroy => 0=not the shortest direction, 1=shortest direction		
        b) enemies to destroy =>0=no improvement, 1=more enemies to destroy	

neighbor fields: 0 safe, 1 death/wall, 2 unsafe (might explode in >1 turns) (3^4=81)			        81

current square: 											                                            04
- 0 no bomb (since dangerous/no benefit/no bomb available), 
- 1 dangerous need to move, 
- 2 bomb destroys one enemey/crate, 
- 3 bomb destroys multiple enemies/crates OR is guaranteed to kill an enemy

= 40500
"""
FEATURE_SHAPE = (4, 4, 4, 81, 4)


def setup(self):
    """
    This is called once when loading each agent.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("q-table.pt"):
        self.logger.info("Setting up model from scratch.")
        self.Q = np.zeros((5, len(ACTIONS)))
    else:
        self.logger.info("Loading model from saved state.")
        with open("q-table.pt", "rb") as file:
            self.Q = pickle.load(file)
        with np.printoptions(threshold=np.inf):
            print(self.Q)


def determine_next_action(game_state: dict, Q) -> str:
    features = state_to_features(game_state)

    best_action_index = np.array(list(map(lambda action: Q[features, action], ACTION_INDICES))).argmax()

    return ACTIONS[best_action_index]


def act(self, game_state: dict) -> str:
    """
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    random_prob = .2
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=ACTIONS_PROBABILITIES)

    self.logger.debug("Querying model for action.")

    self.action = determine_next_action(game_state, self.Q)
    return self.action


def find_distance_to_coin(position, field: np.array):
    """
    : param field: 1 crates, -1 stone walls, 0 free tiles, 2 coins
    """
    if field[position[0], position[1]] == 1 or field[position[0], position[1]] == -1:
        return math.inf

    distance = {(position[0], position[1]): 0}
    todo = [(position[0], position[1])]
    while len(todo) > 0:
        current = todo.pop(0)
        if field[current] == 2:
            return distance[current]

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if
                     field[x, y] == 0 or field[x, y] == 2]
        for neighbor in neighbors:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                todo.append(neighbor)

    return math.inf


def breadth_first_search(position, field, targets):
    """
    Calculates the distance to the nearest target using BFS.
    :param field: 0 free tiles, 1 blocked, 2 target
    :return distance to the nearest target|inf if no reachable target is present
    """
    position = (position[0], position[1])

    if field[position] == 1:
        return math.inf

    distance = {position: 1}
    todo = [position]

    while len(todo) > 0:
        current = todo.pop(0)
        if targets[current] == 1:
            return distance[current]

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                todo.append(neighbor)

    return math.inf


def prepare_field_coins(game_state: dict):
    field = np.abs(game_state['field'])
    explosion_map = game_state['explosion_map']
    field += explosion_map.astype(int)

    targets = np.zeros_like(field)
    for coin in game_state['coins']:
        targets[coin] = 1

    return np.clip(field, 0, 1), targets


def determine_coin_value(x, y, game_state):
    field, targets = prepare_field_coins(game_state)

    distance_up = breadth_first_search((x, y - 1), field, targets)
    distance_down = breadth_first_search((x, y + 1), field, targets)
    distance_left = breadth_first_search((x - 1, y), field, targets)
    distance_right = breadth_first_search((x + 1, y), field, targets)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    choice = np.random.choice(np.flatnonzero(distances == distances.min()))

    if distances[choice] > 64:
        return 4
    return choice


# TODO
def determine_crate_value(x, y, whole_field):
    field = np.clip(whole_field, 0, 1)
    targets = np.abs(np.clip(whole_field, -1, 0))


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
    position = game_state['self'][3]
    x, y = position
    field = game_state['field']

    channels = []
    channels.append(determine_coin_value(x, y, game_state))

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
