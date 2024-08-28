import math
import os
import pickle
import random

import numpy as np

from features import determine_explosion_timer, count_destroyable_crates_and_enemies, create_danger_map, \
    determine_current_square, prepare_escape_path_fields, partially_fill, save_directions, determine_escape_direction, \
    determine_coin_value, determine_is_worth_to_move_crates, determine_crate_value, determine_is_worth_to_move_enemies, \
    determine_enemy_value, determine_escape_direction_scored, save_directions_scored, determine_coin_value_scored, \
    determine_is_worth_to_move_crates_scored, determine_crate_value_scored, determine_is_worth_to_move_enemies_scored, \
    determine_enemy_value_scored

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
FEATURE_SHAPE = (4, 5, 5, 5, 5, 4, len(ACTIONS))

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
        print(self.Q[3, 4, 4, 4, 4, 1])

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
        # print(self.Q[self.last_features])
        pass
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
    position = game_state['self'][3]
    x, y = position

    features = []

    explosion_timer = determine_explosion_timer(game_state)
    count_crates, count_enemies = count_destroyable_crates_and_enemies(x, y, game_state, explosion_timer)

    danger_map = create_danger_map(game_state)

    current_square = determine_current_square(x, y, game_state, count_crates + count_enemies)
    features.append(current_square)

    bomb_input = prepare_escape_path_fields(game_state)

    priority_marker = -1

    if current_square == 1:
        features.append(determine_escape_direction_scored(x, y, game_state, bomb_input)[0])
        priority_marker = 0
    else:
        features.append(save_directions_scored(x, y, game_state, explosion_timer))

    coins, min_distance_coins = determine_coin_value_scored(x, y, game_state, explosion_timer)
    features.append(coins)
    if min_distance_coins < 64 and priority_marker < 0:
        priority_marker = 1

    has_crates = False
    if current_square > 1 and count_crates > 0:
        worth_move = determine_is_worth_to_move_crates_scored(x, y, game_state, count_crates, explosion_timer)
        features.append(worth_move)
        has_crates = True
        if worth_move < 4 and priority_marker < 0:
            priority_marker = 2

    if not has_crates:
        crates, min_distance_crates = determine_crate_value_scored(x, y, game_state, explosion_timer)
        features.append(crates)
        if crates < 4 and priority_marker < 0:
            priority_marker = 2

    has_enemies = False
    if current_square > 1 and count_enemies > 0:
        worth_move = determine_is_worth_to_move_enemies_scored(x, y, game_state, count_enemies, explosion_timer)
        features.append(worth_move)
        has_enemies = True
        if worth_move < 4 and priority_marker < 0:
            priority_marker = 3

    if not has_enemies:
        enemies, min_distance_enemies = determine_enemy_value_scored(x, y, game_state, explosion_timer)
        features.append(enemies)
        if enemies < 4 and priority_marker < 0:
            priority_marker = 3

    priority_marker = max(priority_marker, 0)
    features.append(priority_marker)

    return tuple(features)
