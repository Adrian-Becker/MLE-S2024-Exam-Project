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
    determine_enemy_value_scored, determine_trap_escape_direction_scored, determine_trap_escape_direction_improved, \
    determine_trap_enemy_direction, prepare_field_coins, determine_trap_filter

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
ACTION_INDICES = np.array([0, 1, 2, 3, 4, 5]).astype(int)
ACTION_TO_INDEX = {
    'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'WAIT': 4, 'BOMB': 5
}

ACTIONS_PROBABILITIES = [0.2, 0.2, 0.2, 0.2, 0.18, 0.02]

WIDTH = 17
HEIGHT = 17

TRAP_FLEEING_THRESHOLD = 3

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
FEATURE_SHAPE = (4, 5, 5, 5, 6, 6, 7, 2, 2, 2, 2, len(ACTIONS))

EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 400000

RELATIVE_ROUND_EXPLORATION_THRESHOLD = 0.8

def setup(self):
    """
    This is called once when loading each agent.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.iteration = 0
    self.round = 0
    self.last_action = 4
    if self.train or not os.path.isfile("tables/0001-q-table.pt"):
        if os.path.isfile("starting_point.pt"):
            self.logger.info("Loading starting_point.pt")
            with open(f"starting_point.pt", "rb") as file:
                self.Q = pickle.load(file)
        else:
            self.logger.info("Initial Q Table with 0")
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
            # print(self.Q)
            pass
        print(f"Loaded {file_prefix}")
        # print(self.Q[3, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 4])


def determine_next_action(game_state: dict, Q) -> str:
    features = state_to_features(game_state)

    best_action_index = np.array(list(map(lambda action: Q[features][action], ACTION_INDICES))).argmax()

    return ACTIONS[best_action_index]


def get_feature_string(features):
    return ""
    str = ""
    str_list = [["no bomb", "flee", "bomb crate one", "bomb crate multiple"],
                ["up", "down", "left", "right", "no direction"],
                ["up", "down", "left", "right", "no direction"],
                ["up", "down", "left", "right", "no direction"],
                ["up", "down", "left", "right", "no direction"],
                ["up", "down", "left", "right", "no direction"],
                ["flee", "coin", "crate", "enemy", "trap fleeing"],
                ["up", "down", "left", "right", "no direction", "active trap"],
                ["danger", "no danger"],
                ["danger", "no danger"],
                ["danger", "no danger"],
                ["danger", "no danger"],
                ["up", "down", "left", "right", "no movement"]]
    feature_list = ["Current Square", "Bomb Direction", "Trap Fleeing Direction", "Coin Direction", "Crate Direction",
                    "Enemy Direction", "Priority Marker", "Trap Setting Direction", "UP", "DOWN", "LEFT", "RIGHT",
                    "previous field"]
    for idx in range(len(features)):
        str += f"{feature_list[idx]}: {str_list[idx][features[idx]]}"
        if idx is not len(features) - 1:
            str += "\n"
    return str


def action_from_features(features):
    current_square = features[0]
    escape_direction = features[1]
    trap_escape_direction = features[2]
    coin_direction = features[3]
    crate_direction = features[4]
    enemy_direction = features[5]
    trap_enemy_direction = features[6]
    can_move_up = features[7]
    can_move_down = features[8]
    can_move_left = features[9]
    can_move_right = features[10]

    if current_square == 1:
        return escape_direction

    if trap_escape_direction != 4:
        return trap_escape_direction

    if trap_enemy_direction != 4:
        if trap_enemy_direction == 6:
            return 4
        return trap_enemy_direction

    if current_square == 3 and coin_direction != 4:
        return 5

    if coin_direction != 4:
        return coin_direction

    if crate_direction != 4:
        return crate_direction

    if enemy_direction != 4:
        return enemy_direction

    if can_move_up:
        return 0
    if can_move_down:
        return 1
    if can_move_left:
        return 2
    if can_move_right:
        return 3

    return 4


def act(self, game_state: dict) -> str:
    #self.last_features = state_to_features(game_state, self.last_action)
    #self.last_action = action_from_features(self.last_features)
    #return ACTIONS[self.last_action]
    """
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    random_prob = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.iteration / EPS_DECAY)
    if self.train and random.random() < random_prob and self.round % 2 == 0 and self.round > 10 and self.iteration_per_round >= RELATIVE_ROUND_EXPLORATION_THRESHOLD * self.average_iterations:
        features = state_to_features(game_state, self.last_action)
        if features[0] > 1 and random.random() < 0.5:
            self.last_action = 5
            return 'BOMB'
        self.logger.debug("Choosing action purely at random.")
        self.last_features = features
        action = np.random.choice(ACTIONS, p=ACTIONS_PROBABILITIES)
        self.last_action = ACTION_TO_INDEX[action]
        return action

    self.logger.debug("Querying model for action.")

    self.last_features = state_to_features(game_state, self.last_action)
    best_action_index = np.array(list(map(lambda action: self.Q[self.last_features][action], ACTION_INDICES))).argmax()
    action = ACTIONS[best_action_index]
    if not self.train:
        print(f"{action} {self.last_features}")
        print(f"{get_feature_string(self.last_features)}")
        print(self.Q[self.last_features], "\n")
        pass
    return action


def state_to_features(game_state: dict, last_action) -> np.array:
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

    trap_filter = determine_trap_filter(game_state, explosion_timer)

    current_square = determine_current_square(x, y, game_state, count_crates + count_enemies)
    features.append(current_square)

    bomb_input = prepare_escape_path_fields(game_state)

    if current_square == 1:
        features.append(determine_escape_direction_scored(x, y, game_state, bomb_input)[0])
    else:
        features.append(4)  # save_directions_scored(x, y, game_state, explosion_timer))

    direction = determine_trap_escape_direction_improved(game_state, explosion_timer)
    features.append(direction)

    coins, min_distance_coins = determine_coin_value_scored(x, y, game_state, explosion_timer, trap_filter)
    features.append(coins)

    has_crates = False
    if current_square > 1 and count_crates > 0:
        worth_move = determine_is_worth_to_move_crates_scored(x, y, game_state, count_crates, explosion_timer,
                                                              trap_filter)
        if worth_move == 4:
            worth_move = 5
        features.append(worth_move)
        has_crates = True

    if not has_crates:
        crates, min_distance_crates = determine_crate_value_scored(x, y, game_state, explosion_timer, trap_filter)
        features.append(crates)

    has_enemies = False
    if current_square > 1 and count_enemies > 0:
        worth_move = determine_is_worth_to_move_enemies_scored(x, y, game_state, count_enemies, explosion_timer,
                                                               trap_filter)
        if worth_move == 4:
            worth_move = 5

        features.append(worth_move)
        has_enemies = True

    if not has_enemies:
        enemies, min_distance_enemies = determine_enemy_value_scored(x, y, game_state, explosion_timer, trap_filter)
        features.append(enemies)

    features.append(determine_trap_enemy_direction(game_state, explosion_timer))

    field, _ = prepare_field_coins(game_state, explosion_timer)
    features.append(1 if field[x, y - 1] == 0 else 0)
    features.append(1 if field[x, y + 1] == 0 else 0)
    features.append(1 if field[x - 1, y] == 0 else 0)
    features.append(1 if field[x + 1, y] == 0 else 0)

    #features.append(np.clip(last_action, 0, 4))

    return tuple(features)
