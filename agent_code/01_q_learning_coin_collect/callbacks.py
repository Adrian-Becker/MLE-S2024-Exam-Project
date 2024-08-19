import math
import os
import pickle
import random

import numpy as np

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT']
ACTION_TO_INDEX = {
    'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3, 'WAIT': 4
}
ACTIONS_PROBABILITIES = [0.2, 0.2, 0.2, 0.2, 0.2]

"""
state vector: 
    - for each direction: 0/1/2/3, ranking of shortest distance to nearest coin
"""
STATE_SIZE = 4 ** 4


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
        self.Q = np.zeros((STATE_SIZE, len(ACTIONS)))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)
        with np.printoptions(threshold=np.inf):
            print(self.Q)


def determine_next_action(game_state: dict, Q) -> str:
    field = game_state['field']
    _, __, ___, (x, y) = game_state['self']
    others = [xy for (n, s, b, xy) in game_state['others']]

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((field[d] == 0) and
                (d not in others)):
            valid_tiles.append(d)

    if (x - 1, y) in valid_tiles: valid_actions.append(('LEFT', 0))
    if (x + 1, y) in valid_tiles: valid_actions.append(('RIGHT', 1))
    if (x, y - 1) in valid_tiles: valid_actions.append(('UP', 2))
    if (x, y + 1) in valid_tiles: valid_actions.append(('DOWN', 3))
    if (x, y) in valid_tiles: valid_actions.append(('WAIT', 4))

    if len(valid_actions) > 0:
        index = state_to_index(game_state)
        best_action = 'WAIT'
        best_action_q = -math.inf
        for action in valid_actions:
            q = Q[index, action[1]]
            if q > best_action_q:
                best_action = action[0]
                best_action_q = q
        return best_action
    return 'WAIT'


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
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


def state_to_index(game_state: dict) -> np.array:
    field = game_state['field']
    for coin in game_state['coins']:
        field[coin] = 2

    position = game_state['self'][3]
    x, y = position

    dist_up = find_distance_to_coin([x, y - 1], field)
    dist_down = find_distance_to_coin([x, y + 1], field)
    dist_left = find_distance_to_coin([x - 1, y], field)
    dist_right = find_distance_to_coin([x + 1, y], field)

    distances = [(0, dist_left), (1, dist_right), (2, dist_up), (3, dist_down)]
    distances.sort(key=lambda entry: entry[1])
    index = 0 * (4 ** distances[0][0]) + 1 * (4 ** distances[1][0]) + \
            2 * (4 ** distances[2][0]) + 3 * (4 ** distances[3][0])

    return index


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
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
