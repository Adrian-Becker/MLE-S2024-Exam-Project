import math
import os
import pickle
import random

import numpy as np

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
ACTION_INDICES = np.array([0, 1, 2, 3, 4, 5]).astype(int)
ACTION_TO_INDEX = {
    'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3, 'WAIT': 4, 'BOMB': 5
}
ACTIONS_PROBABILITIES = [0.2, 0.2, 0.2, 0.2, 0.195, 0.005]

WIDTH = 17
HEIGHT = 17

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
FEATURE_SHAPE = (5, 4, 5, 5, 5, len(ACTIONS))

EPS_START = 0.05
EPS_END = 0.001
EPS_DECAY = 4500000


def setup(self):
    """
    This is called once when loading each agent.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.iteration = 0
    if self.train or not os.path.isfile("q-table_no_history.pt"):
        self.logger.info("Setting up model from scratch.")
        self.Q = np.zeros(FEATURE_SHAPE).astype(np.float32)
    else:
        self.logger.info("Loading model from saved state.")
        with open("q-table_no_history.pt", "rb") as file:
            self.Q = pickle.load(file)
        with np.printoptions(threshold=np.inf):
            print(self.Q)


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
    if self.train and random.random() < random_prob:
        features = state_to_features(game_state)
        if features[1] > 1 and random.random() < 0.5:
            return 'BOMB'
        self.logger.debug("Choosing action purely at random.")
        self.last_features = features
        return np.random.choice(ACTIONS, p=ACTIONS_PROBABILITIES)

    self.logger.debug("Querying model for action.")

    self.last_features = state_to_features(game_state)
    best_action_index = np.array(list(map(lambda action: self.Q[self.last_features][action], ACTION_INDICES))).argmax()
    return ACTIONS[best_action_index]


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


def prepare_field_coins(game_state: dict, explosion_timer):
    field = np.abs(game_state['field'])
    explosion_map = game_state['explosion_map']
    field += explosion_map.astype(int)
    for other in game_state['others']:
        field[other[3]] = 1
    field[explosion_timer != 1000] += 1

    targets = np.zeros_like(field)
    for coin in game_state['coins']:
        targets[coin] = 1

    return np.clip(field, 0, 1), targets


def determine_best_direction(x, y, field, targets):
    distance_up = breadth_first_search((x, y - 1), field, targets)
    distance_down = breadth_first_search((x, y + 1), field, targets)
    distance_left = breadth_first_search((x - 1, y), field, targets)
    distance_right = breadth_first_search((x + 1, y), field, targets)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    choice = np.random.choice(np.flatnonzero(distances == distances.min()))

    if distances[choice] > 64:
        return 4
    return choice


def determine_coin_value(x, y, game_state: dict, explosion_timer):
    field, targets = prepare_field_coins(game_state, explosion_timer)

    return determine_best_direction(x, y, field, targets)


def count_destroyable_crates(x, y, game_state: dict, explosion_timer):
    count_crates = 0
    field = game_state['field']

    for dx in range(1, min(4, 17 - x)):
        if field[x + dx, y] == -1:
            break
        if explosion_timer[x + dx, y] != 1000:
            continue
        if field[x + dx, y] == 1:
            count_crates += 1
    for dx in range(1, min(4, x + 1)):
        if field[x - dx, y] == -1:
            break
        if explosion_timer[x - dx, y] != 1000:
            continue
        if field[x - dx, y] == 1:
            count_crates += 1
    for dy in range(1, min(4, 17 - y)):
        if field[x, y + dy] == -1:
            break
        if explosion_timer[x, y + dy] != 1000:
            continue
        if field[x, y + dy] == 1:
            count_crates += 1
    for dy in range(1, min(4, y + 1)):
        if field[x, y - dy] == -1:
            break
        if explosion_timer[x, y - dy] != 1000:
            continue
        if field[x, y - dy] == 1:
            count_crates += 1

    return count_crates


def count_destroyable_enemies(x, y, game_state: dict, explosion_timer):
    count_enemies = 0
    field = game_state['field']

    for other in game_state['others']:
        field[other[3]] = 2

    for dx in range(1, min(4, 17 - x)):
        if field[x + dx, y] == -1:
            break
        if field[x + dx, y] == 2:
            count_enemies += 1
    for dx in range(1, min(4, x + 1)):
        if field[x - dx, y] == -1:
            break
        if field[x - dx, y] == 2:
            count_enemies += 1
    for dy in range(1, min(4, 17 - y)):
        if field[x, y + dy] == -1:
            break
        if field[x, y + dy] == 2:
            count_enemies += 1
    for dy in range(1, min(4, y + 1)):
        if field[x, y - dy] == -1:
            break
        if field[x, y - dy] == 2:
            count_enemies += 1

    for other in game_state['others']:
        field[other[3]] = 0

    return count_enemies


def count_destroyable_crates_and_enemies(x, y, game_state: dict, explosion_timer):
    count_crates = 0
    count_enemies = 0
    field = game_state['field']

    for other in game_state['others']:
        field[other[3]] = 2

    for dx in range(1, min(4, 17 - x)):
        if field[x + dx, y] == -1:
            break
        if field[x + dx, y] == 2:
            count_enemies += 1
        if explosion_timer[x + dx, y] != 1000:
            continue
        if field[x + dx, y] == 1:
            count_crates += 1
    for dx in range(1, min(4, x + 1)):
        if field[x - dx, y] == -1:
            break
        if field[x - dx, y] == 2:
            count_enemies += 1
        if explosion_timer[x - dx, y] != 1000:
            continue
        if field[x - dx, y] == 1:
            count_crates += 1
    for dy in range(1, min(4, 17 - y)):
        if field[x, y + dy] == -1:
            break
        if field[x, y + dy] == 2:
            count_enemies += 1
        if explosion_timer[x, y + dy] != 1000:
            continue
        if field[x, y + dy] == 1:
            count_crates += 1
    for dy in range(1, min(4, y + 1)):
        if field[x, y - dy] == -1:
            break
        if field[x, y - dy] == 2:
            count_enemies += 1
        if explosion_timer[x, y - dy] != 1000:
            continue
        if field[x, y - dy] == 1:
            count_crates += 1

    for other in game_state['others']:
        field[other[3]] = 0

    return count_crates, count_enemies


def determine_field_state(x, y, prepped_field, explosion_map):
    if prepped_field[x, y] > 0:
        return 2
    if 0 < explosion_map[x, y] < 1000:
        return 1
    if explosion_map[x, y] == 0:
        return 2
    return 0


def is_explosion_time_save(time):
    return time < -1 or time > 0


def find_shortest_escape_path(position, field, bomb_field, explosion_time, ignore_starting_square=False):
    if (field[position] != 0 or bomb_field[position] != 1000) and not ignore_starting_square:
        return math.inf

    todo = [position]
    distances = {position: -1 if ignore_starting_square else 0}

    while len(todo) > 0:
        if ignore_starting_square:
            pass
        current = todo.pop(0)

        if explosion_time[current] == 1000 or explosion_time[current] - distances[current] < -1:
            return distances[current]

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                     if ((field[x, y] == 0 and is_explosion_time_save(explosion_time[x, y] - distances[current] - 1))
                         or (field[x, y] == 1 and explosion_time[x, y] - distances[current] - 1 < -1)) and
                     (bomb_field[x, y] == 1000 or bomb_field[x, y] - distances[current] - 1 < -1)]
        for neighbor in neighbors:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                todo.append(neighbor)

    return math.inf


def mark_bomb(field, target_field, value, position, only_decrease=False):
    directions = [
        [(1, 0), (2, 0), (3, 0)],
        [(-1, 0), (-2, 0), (-3, 0)],
        [(0, 1), (0, 2), (0, 3)],
        [(0, -1), (0, -2), (0, -3)]
    ]

    x, y = position

    if only_decrease:
        target_field[x, y] = min(target_field[x, y], value)
    else:
        target_field[x, y] = value

    for direction in directions:
        for delta in direction:
            cx = x + delta[0]
            cy = y + delta[1]
            if cx < 0 or cy < 0 or cx >= 17 or cy >= 17:
                break
            # stone wall, bomb doesn't pass through
            if field[cx, cy] == -1:
                break
            if only_decrease:
                target_field[cx, cy] = min(target_field[cx, cy], value)
            else:
                target_field[cx, cy] = value


def prepare_escape_path_fields(game_state: dict):
    field = np.copy(game_state['field'])
    for other in game_state['others']:
        # mark other players as crates
        field[other[3]] = 1
    # values > 0 or values < -1 are save
    explosion_timer = np.ones_like(field) * 1000
    bomb_field = np.ones_like(field) * 1000
    for bomb in game_state['bombs']:
        bomb_field[bomb[0]] = bomb[1]
        mark_bomb(game_state['field'], explosion_timer, bomb[1], bomb[0], True)
    explosion_timer[game_state['explosion_map'] == 1] = -1
    return field, bomb_field, explosion_timer


def determine_neighbor_fields(x, y, game_state: dict, explosion_timer):
    field = np.abs(game_state['field'])
    for other in game_state['others']:
        field[other[3]] = 1
    explosion_map = game_state['explosion_map'].astype(int)
    for bomb in game_state['bombs']:
        field[bomb[0]] = 1
    field[explosion_map == 1] = 1

    moves = np.array([
        determine_field_state(x, y - 1, field, explosion_timer),
        determine_field_state(x, y + 1, field, explosion_timer),
        determine_field_state(x - 1, y, field, explosion_timer),
        determine_field_state(x + 1, y, field, explosion_timer)
    ])
    if explosion_timer[x, y] == 1000 or 0 in moves:
        minimum = moves.min()
        if minimum == 2:
            return 4
        return np.random.choice(np.flatnonzero(moves == minimum))
        # return moves

    bomb_input = prepare_escape_path_fields(game_state)
    distances = np.array([
        find_shortest_escape_path((x, y - 1), *bomb_input),
        find_shortest_escape_path((x, y + 1), *bomb_input),
        find_shortest_escape_path((x - 1, y), *bomb_input),
        find_shortest_escape_path((x + 1, y), *bomb_input),
    ])
    # min_distance = distances.min()
    # if min_distance < math.inf:
    #    for i in range(4):
    #        if distances[i] == min_distance:
    #            moves[i] = 0
    # return moves
    minimum = distances.min()
    if minimum < 100:
        return np.random.choice(np.flatnonzero(distances == minimum))
    minimum = moves.min()
    if minimum == 2:
        return 4
    return np.random.choice(np.flatnonzero(moves == minimum))


def determine_crate_value(x, y, game_state: dict, explosion_timer):
    field = np.clip(game_state['field'], -1, 1) * -1
    for other in game_state['others']:
        field[other[3]] = 1
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1
    field = np.clip(field, 0, 1)

    targets = np.abs(np.clip(game_state['field'], 0, 1))

    return determine_best_direction(x, y, field, targets)


def determine_enemy_value(x, y, game_state: dict, explosion_timer):
    field = np.abs(game_state['field'])
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1

    targets = np.zeros_like(field)
    for other in game_state['others']:
        targets[other[3]] = 1

    return determine_best_direction(x, y, np.clip(field, 0, 1), targets)


def determine_is_worth_to_move_crates(x, y, game_state: dict, count_crates, explosion_timer):
    count_up = count_destroyable_crates(x, y - 1, game_state, explosion_timer)
    count_down = count_destroyable_crates(x, y + 1, game_state, explosion_timer)
    count_left = count_destroyable_crates(x - 1, y, game_state, explosion_timer)
    count_right = count_destroyable_crates(x + 1, y, game_state, explosion_timer)

    counts = np.array([count_up, count_down, count_left, count_right, count_crates + 1])
    return np.random.choice(np.flatnonzero(counts == counts.max()))


def determine_is_worth_to_move_enemies(x, y, game_state: dict, count_enemies, explosion_timer):
    count_up = count_destroyable_enemies(x, y - 1, game_state, explosion_timer)
    count_down = count_destroyable_enemies(x, y + 1, game_state, explosion_timer)
    count_left = count_destroyable_enemies(x - 1, y, game_state, explosion_timer)
    count_right = count_destroyable_enemies(x + 1, y, game_state, explosion_timer)

    counts = np.array([count_up, count_down, count_left, count_right, count_enemies + 1])
    return np.random.choice(np.flatnonzero(counts == counts.max()))


def determine_current_square(x, y, game_state: dict, count):
    escape_info = prepare_escape_path_fields(game_state)
    if escape_info[2][x, y] != 1000:
        # is dangerous => we need to move
        return 1
    elif game_state['self'][2]:
        # can place bomb
        if count > 0:
            # worth placing bomb, check for escape path
            mark_bomb(game_state['field'], escape_info[2], 3, (x, y), True)

            if find_shortest_escape_path((x, y), *escape_info, True) < math.inf:
                # print('found safe escape')
                # print(explosion_times)

                if count == 1:
                    # can destroy one
                    return 2
                elif count > 1:
                    # can destroy multiple
                    return 3
    return 0


def determine_explosion_timer(game_state: dict):
    explosion_timer = np.ones_like(game_state['field']) * 1000
    for bomb in game_state['bombs']:
        mark_bomb(game_state['field'], explosion_timer, bomb[1], bomb[0], True)
    return explosion_timer


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

    explosion_timer = determine_explosion_timer(game_state)

    features = []
    # features.extend(determine_neighbor_fields(x, y, game_state, explosion_timer))
    features.append(determine_neighbor_fields(x, y, game_state, explosion_timer))

    count_crates, count_enemies = count_destroyable_crates_and_enemies(x, y, game_state, explosion_timer)

    features.append(determine_current_square(x, y, game_state, count_crates + count_enemies))

    features.append(determine_coin_value(x, y, game_state, explosion_timer))

    if count_crates == 0:
        features.append(determine_crate_value(x, y, game_state, explosion_timer))
    else:
        features.append(determine_is_worth_to_move_crates(x, y, game_state, count_crates, explosion_timer))

    if count_enemies == 0:
        features.append(determine_enemy_value(x, y, game_state, explosion_timer))
    else:
        features.append(determine_is_worth_to_move_enemies(x, y, game_state, count_enemies, explosion_timer))

    return tuple(features)
