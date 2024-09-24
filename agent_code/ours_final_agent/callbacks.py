import pickle

import numpy as np

from .features import determine_explosion_timer, count_destroyable_crates_and_enemies, determine_current_square, \
    prepare_escape_path_fields, determine_escape_direction_scored, determine_coin_value_scored, \
    determine_is_worth_to_move_crates_scored, determine_crate_value_scored, determine_is_worth_to_move_enemies_scored, \
    determine_enemy_value_scored, determine_trap_escape_direction_improved, \
    determine_trap_enemy_direction, prepare_field_coins, determine_trap_filter

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
ACTION_INDICES = np.array([0, 1, 2, 3, 4, 5]).astype(int)
ACTION_TO_INDEX = {
    'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'WAIT': 4, 'BOMB': 5
}


def setup(self):
    """
    This is called once when loading each agent.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    with open("q-table.pt", "rb") as file:
        self.Q = pickle.load(file)
    self.last_action = 4


def act(self, game_state: dict) -> str:
    """
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    self.last_features = state_to_features(game_state, self.last_action)
    best_action_index = np.array(list(map(lambda action: self.Q[self.last_features][action], ACTION_INDICES))).argmax()
    return ACTIONS[best_action_index]


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

    found_move_direction = False

    explosion_timer = determine_explosion_timer(game_state)
    count_crates, count_enemies = count_destroyable_crates_and_enemies(x, y, game_state, explosion_timer)

    trap_filter = determine_trap_filter(game_state, explosion_timer)

    current_square = determine_current_square(x, y, game_state, count_crates + count_enemies)
    if current_square > 1:
        found_move_direction = True
    features.append(current_square)

    bomb_input = prepare_escape_path_fields(game_state)

    if current_square == 1:
        # bomb fleeing direction
        features.append(determine_escape_direction_scored(x, y, game_state, bomb_input)[0])
        found_move_direction = True
    else:
        # trap fleeing direction
        direction = determine_trap_escape_direction_improved(game_state, explosion_timer)
        if direction != 4:
            found_move_direction = True
        features.append(direction)

    coins, min_distance_coins = determine_coin_value_scored(x, y, game_state, explosion_timer, trap_filter)
    if coins != 4:
        found_move_direction = True
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
        if crates != 4:
            found_move_direction = True
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
        if enemies != 4:
            found_move_direction = True
        features.append(enemies)

    trap_direction = determine_trap_enemy_direction(game_state, explosion_timer)
    if trap_direction != 4:
        found_move_direction = True
    features.append(trap_direction)

    if found_move_direction:
        features.append(4)
    else:
        field, _ = prepare_field_coins(game_state, explosion_timer)
        directions = np.array([(1 if field[x, y - 1] == 0 else 0),
                               (1 if field[x, y + 1] == 0 else 0),
                               (1 if field[x - 1, y] == 0 else 0),
                               (1 if field[x + 1, y] == 0 else 0)])
        if directions.max() > 0:
            features.append(np.random.choice(np.flatnonzero(directions == directions.max())))
        else:
            features.append(4)

    return tuple(features)
