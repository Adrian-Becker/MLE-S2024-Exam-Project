import math
import numpy as np

from util import print_field

MAX_ADDITIONAL_DISTANCE_COIN_BFS_SEARCH = 3
MAX_ADDITIONAL_DISTANCE_CRATE_BFS_SEARCH = 3
MAX_ADDITIONAL_DISTANCE_ENEMY_BFS_SEARCH = 3

SHOULD_ESCAPE_THRESHOLD = 1
TRAP_FOLLOW_THRESHOLD = -1
TRAP_FILTER = 1


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


def prepare_field_trap(game_state: dict, explosion_timer):
    field = np.abs(game_state['field'])
    explosion_map = game_state['explosion_map']
    field += explosion_map.astype(int)
    field[explosion_timer != 1000] += 1
    return np.clip(field, 0, 1)


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


def determine_best_direction_vector(x, y, field, targets):
    distance_up = breadth_first_search((x, y - 1), field, targets)
    distance_down = breadth_first_search((x, y + 1), field, targets)
    distance_left = breadth_first_search((x - 1, y), field, targets)
    distance_right = breadth_first_search((x + 1, y), field, targets)

    distances = np.array([distance_up, distance_down, distance_left, distance_right]).astype(np.float32)
    min_distance = distances.min()

    distances[distances > 32] = 32
    distances /= -32
    distances += 1

    return distances, min_distance


def determine_best_direction(x, y, field, targets):
    distance_up = breadth_first_search((x, y - 1), field, targets)
    distance_down = breadth_first_search((x, y + 1), field, targets)
    distance_left = breadth_first_search((x - 1, y), field, targets)
    distance_right = breadth_first_search((x + 1, y), field, targets)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    min_distance = distances.min()
    directions = np.array([0, 0, 0, 0])
    if min_distance < 64:
        directions[distances == min_distance] = 1
    return directions, min_distance


def discover_coins(enemy, field, targets):
    todo = [enemy[3]]
    distance = {enemy[3]: 0}

    coins = []

    while len(todo) > 0:
        current = todo.pop(0)
        if targets[current] > 0 and current not in coins:
            targets[current] = min(targets[current], distance[current])
            todo = [current]
            distance = {current: distance[current]}
            for coin in coins:
                distance[coin] = math.inf
            coins.append(current)
            continue

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                todo.append(neighbor)


def find_winnable_coins(field, coins, enemies):
    targets = np.zeros_like(field)
    for coin in coins:
        targets[coin] = 1000
    for enemy in enemies:
        discover_coins(enemy, field, targets)
    return targets


def breadth_first_search_coins_scored(position, field, targets):
    if field[position] != 0:
        return math.inf, 0

    todo = [position]
    distance = {position: 1}

    score = 0

    min_distance = math.inf
    max_distance = math.inf

    while len(todo) > 0:
        current = todo.pop(0)

        if distance[current] > max_distance:
            return min_distance, score
        if targets[current] - distance[current] >= 0:
            score += 1
            min_distance = min(min_distance, distance[current])
            max_distance = min(max_distance, distance[current] + MAX_ADDITIONAL_DISTANCE_COIN_BFS_SEARCH)

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                todo.append(neighbor)

    return min_distance, score


def breadth_first_search_coins(position, field, targets):
    if field[position] != 0:
        return math.inf

    todo = [position]
    distance = {position: 1}

    while len(todo) > 0:
        current = todo.pop(0)

        if targets[current] - distance[current] > 0:
            return distance[current]

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                todo.append(neighbor)
    return math.inf


def determine_coin_value_vector(x, y, game_state: dict, explosion_timer):
    field, _ = prepare_field_coins(game_state, explosion_timer)
    targets = find_winnable_coins(field, game_state['coins'], game_state['others'])

    distances = np.array([
        breadth_first_search_coins((x, y - 1), field, targets),
        breadth_first_search_coins((x, y + 1), field, targets),
        breadth_first_search_coins((x - 1, y), field, targets),
        breadth_first_search_coins((x + 1, y), field, targets)
    ]).astype(np.float32)

    distances[distances > 32] = 32.0
    distances /= -32.0
    distances += 1

    return distances


def determine_coin_value_scored(x, y, game_state: dict, explosion_timer, trap_filter):
    field, _ = prepare_field_coins(game_state, explosion_timer)
    targets = find_winnable_coins(field, game_state['coins'], game_state['others'])

    field[game_state['self'][3]] = 1

    distance_up, score_up = breadth_first_search_coins_scored((x, y - 1), field, targets)
    distance_down, score_down = breadth_first_search_coins_scored((x, y + 1), field, targets)
    distance_left, score_left = breadth_first_search_coins_scored((x - 1, y), field, targets)
    distance_right, score_right = breadth_first_search_coins_scored((x + 1, y), field, targets)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    distances[trap_filter == 0] = 1000
    min_distance = distances.min()

    if min_distance < 64:
        scores = np.array([score_up, score_down, score_left, score_right])
        scores[distances > min_distance] = 0
        return np.random.choice(np.flatnonzero(scores == scores.max())), min_distance
    return 4, min_distance


def determine_coin_value_scored_reward(x, y, game_state: dict, explosion_timer, trap_filter):
    field, _ = prepare_field_coins(game_state, explosion_timer)
    targets = find_winnable_coins(field, game_state['coins'], game_state['others'])

    field[game_state['self'][3]] = 1

    distance_up, score_up = breadth_first_search_coins_scored((x, y - 1), field, targets)
    distance_down, score_down = breadth_first_search_coins_scored((x, y + 1), field, targets)
    distance_left, score_left = breadth_first_search_coins_scored((x - 1, y), field, targets)
    distance_right, score_right = breadth_first_search_coins_scored((x + 1, y), field, targets)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    distances[trap_filter == 0] = 1000
    min_distance = distances.min()

    if min_distance < 64:
        scores = np.array([score_up, score_down, score_left, score_right])
        scores[distances > min_distance] = 0
        results = np.array([0, 0, 0, 0])
        results[scores == scores.max()] = 1
        return results
    return np.array([0, 0, 0, 0])


def determine_coin_value(x, y, game_state: dict, explosion_timer):
    field, _ = prepare_field_coins(game_state, explosion_timer)
    targets = find_winnable_coins(field, game_state['coins'], game_state['others'])

    distance_up = breadth_first_search_coins((x, y - 1), field, targets)
    distance_down = breadth_first_search_coins((x, y + 1), field, targets)
    distance_left = breadth_first_search_coins((x - 1, y), field, targets)
    distance_right = breadth_first_search_coins((x + 1, y), field, targets)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    min_distance = distances.min()
    directions = np.array([0, 0, 0, 0])
    if min_distance < 64:
        directions[distances == min_distance] = 1
    return directions, min_distance


def count_destroyable_crates(x, y, game_state: dict, explosion_timer):
    if x < 0 or y < 0 or x >= 17 or y >= 17:
        return 0
    if game_state['field'][x, y] != 0:
        return 0
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
    if game_state['field'][x, y] != 0:
        return 0

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
    if game_state['field'][x, y] != 0:
        return 0

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


def find_shortest_escape_path(position, field, bomb_field, explosion_time, ignore_starting_square=False,
                              starting_distance=0):
    if (field[position] != 0 or bomb_field[position] != 1000 or not is_explosion_time_save(
            explosion_time[position])) and not ignore_starting_square:
        return math.inf

    todo = [position]
    distances = {position: (-1 if ignore_starting_square else 0) + starting_distance}

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


def determine_escape_direction_vector(x, y, game_state: dict, bomb_input):
    distances = np.array([
        find_shortest_escape_path((x, y - 1), *bomb_input),
        find_shortest_escape_path((x, y + 1), *bomb_input),
        find_shortest_escape_path((x - 1, y), *bomb_input),
        find_shortest_escape_path((x + 1, y), *bomb_input),
    ]).astype(np.float32)
    distances[distances > 12] = 5
    distances /= -5
    distances += 1
    return distances


def determine_escape_direction_scored(x, y, game_state: dict, bomb_input):
    distances = np.array([
        find_shortest_escape_path((x, y - 1), *bomb_input),
        find_shortest_escape_path((x, y + 1), *bomb_input),
        find_shortest_escape_path((x - 1, y), *bomb_input),
        find_shortest_escape_path((x + 1, y), *bomb_input),
    ])
    min_distance = distances.min()
    if min_distance < math.inf:
        return np.random.choice(np.flatnonzero(distances == min_distance)), min_distance
    return 4, min_distance


def determine_escape_direction(x, y, game_state: dict, bomb_input):
    distances = np.array([
        find_shortest_escape_path((x, y - 1), *bomb_input),
        find_shortest_escape_path((x, y + 1), *bomb_input),
        find_shortest_escape_path((x - 1, y), *bomb_input),
        find_shortest_escape_path((x + 1, y), *bomb_input),
    ])
    moves = np.array([0, 0, 0, 0])

    min_distance = distances.min()
    if min_distance < math.inf:
        moves[distances == min_distance] = 1
    return moves


def breadth_first_search_crates_scored(position, field, game_state, explosion_timer):
    """
    Calculates the distance to the nearest target using BFS.
    :param field: 0 free tiles, 1 blocked, 2 target
    :return distance to the nearest target|inf if no reachable target is present
    """
    position = (position[0], position[1])

    if field[position] == 1:
        return math.inf, 0

    distance = {position: 1}
    todo = [position]

    min_distance = math.inf
    cur_distance = math.inf
    max_score = 0

    while len(todo) > 0:
        current = todo.pop(0)
        x, y = current

        count = count_destroyable_crates(x, y, game_state, explosion_timer)
        if count > 0 and count > max_score:
            min_distance = min(min_distance, distance[current])
            cur_distance = distance[current]
            max_score = count

        if distance[current] + 1 > min_distance:
            return cur_distance, max_score

        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                todo.append(neighbor)

    return min_distance, max_score


def breadth_first_search_enemies_scored(position, field, targets):
    """
    Calculates the distance to the nearest target using BFS.
    :param field: 0 free tiles, 1 blocked, 2 target
    :return distance to the nearest target|inf if no reachable target is present
    """
    position = (position[0], position[1])

    if field[position] == 1:
        return math.inf, 0

    distance = {position: 1}
    todo = [position]

    score = 0
    min_distance = math.inf
    max_distance = math.inf

    while len(todo) > 0:
        current = todo.pop(0)

        if distance[current] > max_distance:
            return min_distance, score

        if targets[current] == 1:
            if distance[current] == 1:
                continue
            score += 1
            min_distance = min(min_distance, distance[current])
            max_distance = max(max_distance, distance[current] + MAX_ADDITIONAL_DISTANCE_ENEMY_BFS_SEARCH)
            continue

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                todo.append(neighbor)

    return min_distance, score


def determine_crate_value_vector(x, y, game_state: dict, explosion_timer):
    field = np.clip(game_state['field'], -1, 1) * -1
    for other in game_state['others']:
        field[other[3]] = 1
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1
    field = np.clip(field, 0, 1)

    field[game_state['self'][3]] = 1

    targets = np.clip(game_state['field'], 0, 1)
    targets[explosion_timer != 1000] = 0

    directions, min_distance = determine_best_direction_vector(x, y, field, targets)
    if min_distance == 1:
        positions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        for i in range(len(directions)):
            if directions[i] > 0.95:
                field[positions[i]] = 1
        directions, _ = determine_best_direction_vector(x, y, field, targets)
        return directions
    return directions


def determine_crate_value_scored(x, y, game_state: dict, explosion_timer, trap_filter):
    field = np.abs(game_state['field'])
    for other in game_state['others']:
        field[other[3]] = 1
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1
    field = np.clip(field, 0, 1)

    distance_up, score_up = breadth_first_search_crates_scored((x, y - 1), field, game_state, explosion_timer)
    distance_down, score_down = breadth_first_search_crates_scored((x, y + 1), field, game_state, explosion_timer)
    distance_left, score_left = breadth_first_search_crates_scored((x - 1, y), field, game_state, explosion_timer)
    distance_right, score_right = breadth_first_search_crates_scored((x + 1, y), field, game_state, explosion_timer)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    distances[trap_filter == 0] = 1000
    min_distance = distances.min()

    if min_distance < 64:
        scores = np.array([score_up, score_down, score_left, score_right])
        scores[distances > min_distance + 1] = 0
        return np.random.choice(np.flatnonzero(scores == scores.max())), min_distance
    return 4, min_distance


def determine_crate_value_scored_reward(x, y, game_state: dict, explosion_timer, trap_filter):
    field = np.abs(game_state['field'])
    for other in game_state['others']:
        field[other[3]] = 1
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1
    field = np.clip(field, 0, 1)

    distance_up, score_up = breadth_first_search_crates_scored((x, y - 1), field, game_state, explosion_timer)
    distance_down, score_down = breadth_first_search_crates_scored((x, y + 1), field, game_state, explosion_timer)
    distance_left, score_left = breadth_first_search_crates_scored((x - 1, y), field, game_state, explosion_timer)
    distance_right, score_right = breadth_first_search_crates_scored((x + 1, y), field, game_state, explosion_timer)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    distances[trap_filter == 0] = 1000
    min_distance = distances.min()

    if min_distance < 64:
        scores = np.array([score_up, score_down, score_left, score_right])
        scores[distances > min_distance + 1] = 0
        results = np.array([0, 0, 0, 0])
        results[scores == scores.max()] = 1
        return results
    return np.array([0, 0, 0, 0])


def determine_crate_value(x, y, game_state: dict, explosion_timer):
    field = np.clip(game_state['field'], -1, 1) * -1
    for other in game_state['others']:
        field[other[3]] = 1
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1
    field = np.clip(field, 0, 1)

    field[game_state['self'][3]] = 1

    targets = np.clip(game_state['field'], 0, 1)
    targets[explosion_timer != 1000] = 0

    directions, min_distance = determine_best_direction(x, y, field, targets)
    if min_distance == 1:
        positions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        for i in range(len(directions)):
            if directions[i] == 1:
                field[positions[i]] = 1
        return determine_best_direction(x, y, field, targets)
    return directions, min_distance


def determine_enemy_value_vector(x, y, game_state: dict, explosion_timer):
    field = np.abs(game_state['field'])
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1

    targets = np.zeros_like(field)
    for other in game_state['others']:
        targets[other[3]] = 1

    directions, _ = determine_best_direction_vector(x, y, np.clip(field, 0, 1), targets)
    return directions


def determine_enemy_value(x, y, game_state: dict, explosion_timer):
    field = np.abs(game_state['field'])
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1

    targets = np.zeros_like(field)
    for other in game_state['others']:
        targets[other[3]] = 1

    return determine_best_direction(x, y, np.clip(field, 0, 1), targets)


def determine_enemy_value_scored(x, y, game_state: dict, explosion_timer, trap_filter):
    field = np.abs(game_state['field'])
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1

    targets = np.zeros_like(field)
    for other in game_state['others']:
        targets[other[3]] = 1

    field[game_state['self'][3]] = 1

    distance_up, score_up = breadth_first_search_enemies_scored((x, y - 1), field, targets)
    distance_down, score_down = breadth_first_search_enemies_scored((x, y + 1), field, targets)
    distance_left, score_left = breadth_first_search_enemies_scored((x - 1, y), field, targets)
    distance_right, score_right = breadth_first_search_enemies_scored((x + 1, y), field, targets)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    distances[trap_filter == 0] = 1000
    min_distance = distances.min()

    if min_distance < 64:
        scores = np.array([score_up, score_down, score_left, score_right])
        scores[distances > min_distance] = 0
        return np.random.choice(np.flatnonzero(scores == scores.max())), min_distance
    return 4, min_distance


def determine_enemy_value_scored_reward(x, y, game_state: dict, explosion_timer, trap_filter):
    field = np.abs(game_state['field'])
    field += game_state['explosion_map'].astype(int)
    field[explosion_timer != 1000] += 1

    targets = np.zeros_like(field)
    for other in game_state['others']:
        targets[other[3]] = 1

    field[game_state['self'][3]] = 1

    distance_up, score_up = breadth_first_search_enemies_scored((x, y - 1), field, targets)
    distance_down, score_down = breadth_first_search_enemies_scored((x, y + 1), field, targets)
    distance_left, score_left = breadth_first_search_enemies_scored((x - 1, y), field, targets)
    distance_right, score_right = breadth_first_search_enemies_scored((x + 1, y), field, targets)

    distances = np.array([distance_up, distance_down, distance_left, distance_right])
    distances[trap_filter == 0] = 1000
    min_distance = distances.min()

    if min_distance < 64:
        directions = np.array([0, 0, 0, 0])
        scores = np.array([score_up, score_down, score_left, score_right])
        scores[distances > min_distance] = 0
        directions[scores == scores.max()] = 1
        return directions
    return [0, 0, 0, 0]


def determine_is_worth_to_move_crates_scored(x, y, game_state: dict, count_crates, explosion_timer, trap_filter):
    directions, double_directions = count_destroyable_crates_double_move_directions(x, y, game_state, explosion_timer,
                                                                                    count_crates)
    directions = np.array(directions)
    double_directions = np.array(double_directions)

    directions[trap_filter == 0] = 0
    double_directions[trap_filter == 0] = 0

    max_directions = directions.max()
    max_double_directions = double_directions.max()

    max_destroy = max(max_directions, max_double_directions)

    if max_destroy <= count_crates:
        return 4

    if max_directions == max_destroy:
        return np.random.choice(np.flatnonzero(directions == max_destroy))

    return np.random.choice(np.flatnonzero(double_directions == max_destroy))


def determine_is_worth_to_move_crates_scored_reward(x, y, game_state: dict, count_crates, explosion_timer, trap_filter):
    directions, double_directions = count_destroyable_crates_double_move_directions(x, y, game_state, explosion_timer,
                                                                                    count_crates)
    directions = np.array(directions)
    double_directions = np.array(double_directions)

    directions[trap_filter == 0] = 0
    double_directions[trap_filter == 0] = 0

    max_directions = directions.max()
    max_double_directions = double_directions.max()

    max_destroy = max(max_directions, max_double_directions)

    if max_destroy <= count_crates:
        return np.array([0, 0, 0, 0])

    final_directions = np.array([0, 0, 0, 0])
    if max_directions == max_destroy:
        final_directions[directions == max_destroy] = 1
        return final_directions

    final_directions[double_directions == max_destroy] = 1
    return final_directions


def count_destroyable_crates_double_move_directions(x, y, game_state: dict, explosion_timer, count_crates):
    return count_destroyable_crates_directions(x, y, game_state, explosion_timer, count_crates), \
        [max(count_destroyable_crates_directions(x, y - 1, game_state, explosion_timer, count_crates)),
         max(count_destroyable_crates_directions(x, y + 1, game_state, explosion_timer, count_crates)),
         max(count_destroyable_crates_directions(x - 1, y, game_state, explosion_timer, count_crates)),
         max(count_destroyable_crates_directions(x + 1, y, game_state, explosion_timer, count_crates))]


def count_and_filter_destroyable_crates(x, y, game_state: dict, explosion_timer, count_crates):
    count = count_destroyable_crates(x, y, game_state, explosion_timer)
    if count <= count_crates:
        return count

    escape_info = prepare_escape_path_fields(game_state)
    mark_bomb(game_state['field'], escape_info[2], 3, (x, y), True)
    if find_shortest_escape_path((x, y), *escape_info, True) > 64:
        return 0
    return count


def count_destroyable_crates_directions(x, y, game_state: dict, explosion_timer, count_crates):
    if game_state['field'][x, y] != 0 or game_state['explosion_map'][x, y] != 0 or explosion_timer[x, y] != 1000:
        return [0, 0, 0, 0]
    return [count_and_filter_destroyable_crates(x, y - 1, game_state, explosion_timer, count_crates),
            count_and_filter_destroyable_crates(x, y + 1, game_state, explosion_timer, count_crates),
            count_and_filter_destroyable_crates(x - 1, y, game_state, explosion_timer, count_crates),
            count_and_filter_destroyable_crates(x + 1, y, game_state, explosion_timer, count_crates)]


def determine_is_worth_to_move_enemies_scored(x, y, game_state: dict, count_enemies, explosion_timer, trap_filter):
    directions = determine_is_worth_to_move_enemies_trap_filter(x, y, game_state, count_enemies,
                                                                explosion_timer, trap_filter)
    if directions.max() > 0:
        return np.random.choice(np.flatnonzero(directions == 1))
    return 4


def determine_is_worth_to_move_crates(x, y, game_state: dict, count_crates, explosion_timer):
    count_up = count_destroyable_crates(x, y - 1, game_state, explosion_timer)
    count_down = count_destroyable_crates(x, y + 1, game_state, explosion_timer)
    count_left = count_destroyable_crates(x - 1, y, game_state, explosion_timer)
    count_right = count_destroyable_crates(x + 1, y, game_state, explosion_timer)

    counts = np.array([count_up, count_down, count_left, count_right])
    positions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

    for i in range(4):
        if counts[i] > count_crates:
            escape_info = prepare_escape_path_fields(game_state)
            mark_bomb(game_state['field'], escape_info[2], 3, positions[i], True)
            if find_shortest_escape_path((x, y), *escape_info, True) > 64:
                counts[i] = 0

    max_count = counts.max()
    directions = np.array([0, 0, 0, 0])
    if max_count > count_crates:
        directions[counts == max_count] = 1
    return directions


def determine_is_worth_to_move_enemies_trap_filter(x, y, game_state: dict, count_enemies, explosion_timer, trap_filter):
    count_up = count_destroyable_enemies(x, y - 1, game_state, explosion_timer)
    count_down = count_destroyable_enemies(x, y + 1, game_state, explosion_timer)
    count_left = count_destroyable_enemies(x - 1, y, game_state, explosion_timer)
    count_right = count_destroyable_enemies(x + 1, y, game_state, explosion_timer)

    counts = np.array([count_up, count_down, count_left, count_right])
    counts[trap_filter == 0] = 0
    positions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

    for i in range(4):
        if counts[i] > count_enemies:
            escape_info = prepare_escape_path_fields(game_state)
            mark_bomb(game_state['field'], escape_info[2], 3, positions[i], True)
            if find_shortest_escape_path((x, y), *escape_info, True) > 64:
                counts[i] = 0

    max_count = counts.max()
    directions = np.array([0, 0, 0, 0])
    if max_count > count_enemies:
        directions[counts == max_count] = 1
    return directions


def determine_is_worth_to_move_enemies(x, y, game_state: dict, count_enemies, explosion_timer):
    count_up = count_destroyable_enemies(x, y - 1, game_state, explosion_timer)
    count_down = count_destroyable_enemies(x, y + 1, game_state, explosion_timer)
    count_left = count_destroyable_enemies(x - 1, y, game_state, explosion_timer)
    count_right = count_destroyable_enemies(x + 1, y, game_state, explosion_timer)

    counts = np.array([count_up, count_down, count_left, count_right])
    positions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]

    for i in range(4):
        if counts[i] > count_enemies:
            escape_info = prepare_escape_path_fields(game_state)
            mark_bomb(game_state['field'], escape_info[2], 3, positions[i], True)
            if find_shortest_escape_path((x, y), *escape_info, True) > 64:
                counts[i] = 0

    max_count = counts.max()
    directions = np.array([0, 0, 0, 0])
    if max_count > count_enemies:
        directions[counts == max_count] = 1
    return directions


def determine_current_square_vector(x, y, game_state: dict, count):
    result = np.zeros((4,))
    result[determine_current_square(x, y, game_state, count)] = 1
    return result


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


def find_escape_path_danger_map(position, field, bomb_field, explosion_time, danger_map, starting_time=0):
    todo = [position]
    distances = {position: starting_time}

    while len(todo) > 0:
        current = todo.pop(0)
        if danger_map[current] - distances[current] <= 0:
            continue

        if (explosion_time[current] == 1000 or explosion_time[current] - distances[current] < -1) and \
                danger_map[current] > 64:
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


def determine_trap_escape_direction_scored(x, y, game_state: dict, bomb_input, danger_map):
    if danger_map[(x, y)] == 1000:
        return 4, math.inf

    distances = np.array([
        find_escape_path_danger_map((x, y - 1), *bomb_input, danger_map, starting_time=1),
        find_escape_path_danger_map((x, y + 1), *bomb_input, danger_map, starting_time=1),
        find_escape_path_danger_map((x - 1, y), *bomb_input, danger_map, starting_time=1),
        find_escape_path_danger_map((x + 1, y), *bomb_input, danger_map, starting_time=1),
    ])
    moves = np.array([0, 0, 0, 0])

    min_distance = distances.min()
    if min_distance < math.inf:
        return np.random.choice(np.flatnonzero(distances == min_distance)), min_distance
    return 4, min_distance


def determine_trap_escape_direction(x, y, game_state: dict, bomb_input, danger_map):
    distances = np.array([
        find_escape_path_danger_map((x, y - 1), *bomb_input, danger_map, starting_time=1),
        find_escape_path_danger_map((x, y + 1), *bomb_input, danger_map, starting_time=1),
        find_escape_path_danger_map((x - 1, y), *bomb_input, danger_map, starting_time=1),
        find_escape_path_danger_map((x + 1, y), *bomb_input, danger_map, starting_time=1),
    ])
    moves = np.array([0, 0, 0, 0])

    min_distance = distances.min()
    if min_distance < math.inf:
        moves[distances == min_distance] = 1
    return moves


def save_directions_scored(x, y, game_state, explosion_timer):
    field, _ = prepare_field_coins(game_state, explosion_timer)
    positions = np.array([
        1 if field[x, y - 1] == 0 else 0,
        1 if field[x, y + 1] == 0 else 0,
        1 if field[x - 1, y] == 0 else 0,
        1 if field[x + 1, y] == 0 else 0
    ])

    max_position = positions.max()
    if max_position > 0:
        return np.random.choice(np.flatnonzero(positions == 1))
    return 4


def save_directions(x, y, game_state, explosion_timer):
    field, _ = prepare_field_coins(game_state, explosion_timer)
    positions = [
        1 if field[x, y - 1] == 0 else 0,
        1 if field[x, y + 1] == 0 else 0,
        1 if field[x - 1, y] == 0 else 0,
        1 if field[x + 1, y] == 0 else 0
    ]
    return positions


def partially_fill(features, game_state, x, y, current_square, count_crates, count_enemies, explosion_timer,
                   bomb_input):
    if current_square == 1:
        features.extend(determine_escape_direction(x, y, game_state, bomb_input))
        features.append(0)
        return features

    coins, min_distance_coins = determine_coin_value(x, y, game_state, explosion_timer)
    coins_valid = coins.max() > 0
    if coins_valid:
        features.extend(coins)
        features.append(1)
        return features

    if current_square > 1 and count_crates > 0:
        features.extend(determine_is_worth_to_move_crates(x, y, game_state, count_crates, explosion_timer))
        features.append(2)
        return features

    crates, min_distance_crates = determine_crate_value(x, y, game_state, explosion_timer)
    crates_valid = crates.max() > 0
    if crates_valid:
        features.extend(crates)
        features.append(2)
        return features

    if current_square > 1 and count_enemies > 0:
        features.extend(determine_is_worth_to_move_enemies(x, y, game_state, count_enemies, explosion_timer))
        features.append(3)
        return features

    enemies, min_distance_enemies = determine_enemy_value(x, y, game_state, explosion_timer)
    if enemies.max() > 0:
        features.extend(enemies)
        features.append(3)
        return features

    features.extend(save_directions(x, y, game_state, explosion_timer))
    features.append(0)
    return features


def fill_reachable_map(field, reachable_field, bomb_field, explosion_time, position, max_depth=3):
    todo = [position]
    distances = {position: -1}

    while len(todo) > 0:
        current = todo.pop(0)
        if distances[current] + 1 == max_depth:
            return

        if reachable_field[current] < distances[current]:
            continue
        reachable_field[current] = distances[current]
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                     if ((field[x, y] == 0 and is_explosion_time_save(explosion_time[x, y] - distances[current] - 1))
                         or (field[x, y] == 1 and explosion_time[x, y] - distances[current] - 1 < -1)) and
                     (bomb_field[x, y] == 1000 or bomb_field[x, y] - distances[current] - 1 < -1)]
        for neighbor in neighbors:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                todo.append(neighbor)


def compute_danger(position, field, time_value, danger_map, bomb_field, explosion_time):
    bomb_field[position] = 3 + time_value + 1
    explosion_time_adapted = np.copy(explosion_time)
    mark_bomb(field, explosion_time_adapted, 3 + time_value + 1, position, True)

    directions = [
        [(1, 0), (2, 0), (3, 0)],
        [(-1, 0), (-2, 0), (-3, 0)],
        [(0, 1), (0, 2), (0, 3)],
        [(0, -1), (0, -2), (0, -3)]
    ]

    x, y = position
    for direction in directions:
        for delta in direction:
            cx = x + delta[0]
            cy = y + delta[1]
            if cx < 0 or cy < 0 or cx >= 17 or cy >= 17:
                break
            # stone wall, bomb doesn't pass through
            if field[cx, cy] == -1:
                break
            if find_shortest_escape_path((cx, cy), field, bomb_field, explosion_time_adapted, True,
                                         time_value + 2) > 64:
                danger_map[cx, cy] = min(danger_map[cx, cy], time_value + 1)
    bomb_field[position] = 1000


def create_danger_map(game_state):
    reachable_field = np.ones_like(game_state['field']) * 1000
    field = game_state['field']
    explosion_time = np.ones_like(field) * 1000
    bomb_field = np.ones_like(field) * 1000
    for bomb in game_state['bombs']:
        bomb_field[bomb[0]] = bomb[1]
        mark_bomb(game_state['field'], explosion_time, bomb[1], bomb[0], True)
    explosion_time[game_state['explosion_map'] == 1] = -1
    for other in game_state['others']:
        fill_reachable_map(field, reachable_field, bomb_field, explosion_time, other[3])

    danger_map = np.ones_like(game_state['field']) * 1000
    for x in range(17):
        for y in range(17):
            if reachable_field[(x, y)] < 1000:
                compute_danger((x, y), field, reachable_field[(x, y)], danger_map, bomb_field, explosion_time)

    return danger_map


def state_to_features_for_q_learning(game_state: dict) -> np.array:
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
    partially_fill(features, game_state, x, y, current_square, count_crates, count_enemies, explosion_timer, bomb_input)

    positions = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
    field = np.abs(game_state['field'])
    for enemy in game_state['others']:
        field[enemy[3]] = 1

    for dposition in positions:
        features.append(
            1 if danger_map[dposition] >= danger_map[position] and field[dposition] == 0 and explosion_timer[
                dposition] == 1000 and game_state['explosion_map'][dposition] == 0 else 0)

    return tuple(features)


def mark_dangerous_bomb_spots(x, y, game_state):
    danger_field = np.zeros_like(game_state['field'])
    field = game_state['field']

    for dx in range(1, min(4, 17 - x)):
        if field[x + dx, y] == -1:
            break
        if field[x + dx, y] == 0:
            danger_field[x + dx, y] = 1

    for dx in range(1, min(4, x + 1)):
        if field[x - dx, y] == -1:
            break
        if field[x - dx, y] == 0:
            danger_field[x - dx, y] = 1

    for dy in range(1, min(4, 17 - y)):
        if field[x, y + dy] == -1:
            break
        if field[x, y + dy] == 0:
            danger_field[x, y + dy] = 1

    for dy in range(1, min(4, y + 1)):
        if field[x, y - dy] == -1:
            break
        if field[x, y - dy] == 0:
            danger_field[x, y - dy] = 1

    return danger_field


def mark_reachable(position, field, marked_field, counter_field):
    todo = [position]
    distances = {position: 0}

    while len(todo) > 0:
        current = todo.pop(0)
        if marked_field[current] == 1:
            counter_field[current] = min(counter_field[current], distances[current])

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                todo.append(neighbor)


def determine_could_escape(position, bomb, time, game_state, explosion_time):
    escape_info = prepare_escape_path_fields(game_state)
    mark_bomb(game_state['field'], escape_info[2], 3 + time, bomb, True)

    return find_shortest_escape_path(position, *escape_info, False, starting_distance=time) < math.inf


def find_trapped_fields(position, bomb, field):
    field = np.copy(field)
    field[bomb] = 1

    target = np.zeros_like(field)
    todo = [position]
    done = [position]
    while len(todo) > 0:
        current = todo.pop(0)
        target[current] = 1
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if
                     0 <= x < 17 and 0 <= y < 17 and field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in done:
                done.append(neighbor)
                todo.append(neighbor)
    return target


def determine_time_to_escape(position, bomb, field):
    target = find_trapped_fields(position, bomb, field)

    todo = [position]
    distances = {position: 0}

    while len(todo) > 0:
        current = todo.pop(0)
        if target[current] < 1:
            return distances[current]

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                     if 0 <= x < 17 and 0 <= y < 17 and field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                todo.append(neighbor)
    return math.inf


def determine_trap_field(game_state, field, explosion_timer, victim, attackers, debug=False):
    marked_field = mark_dangerous_bomb_spots(*victim, game_state)
    counter_field = np.ones_like(marked_field) * 1000
    old_value = field[victim]
    field[victim] = 1
    for other in attackers:
        mark_reachable(other[3], field, marked_field, counter_field)
    field[victim] = old_value
    for x in range(17):
        for y in range(17):
            if counter_field[x, y] != 1000:
                if determine_could_escape(victim, (x, y), counter_field[x, y], game_state, explosion_timer):
                    counter_field[x, y] = 1000
                else:
                    time_to_escape = determine_time_to_escape(victim, (x, y), field)
                    if time_to_escape < math.inf:
                        counter_field[x, y] -= time_to_escape
                    else:
                        counter_field[x, y] = -1000
    return counter_field


def determine_trap_filter(game_state, explosion_timer):
    x, y = game_state['self'][3]
    attackers = game_state['others']

    field = prepare_field_trap(game_state, explosion_timer)
    for attacker in attackers:
        field[attacker[3]] = 1

    # print_field(determine_trap_field(game_state, field, explosion_timer, (x, y - 1), attackers, True))
    # print_field(determine_trap_field(game_state, field, explosion_timer, (x, y + 1), attackers, True))
    # print_field(determine_trap_field(game_state, field, explosion_timer, (x - 1, y), attackers, True))
    # print_field(determine_trap_field(game_state, field, explosion_timer, (x + 1, y), attackers, True))

    trap_directions = np.array([
        1 if determine_trap_field(game_state, field, explosion_timer, (x, y - 1), attackers).min() > TRAP_FILTER else 0,
        1 if determine_trap_field(game_state, field, explosion_timer, (x, y + 1), attackers).min() > TRAP_FILTER else 0,
        1 if determine_trap_field(game_state, field, explosion_timer, (x - 1, y), attackers).min() > TRAP_FILTER else 0,
        1 if determine_trap_field(game_state, field, explosion_timer, (x + 1, y), attackers).min() > TRAP_FILTER else 0
    ])
    return trap_directions


def determine_trap_escape_directions_improved(game_state, explosion_timer):
    victim = game_state['self'][3]
    attackers = game_state['others']

    field = prepare_field_trap(game_state, explosion_timer)
    for attacker in attackers:
        field[attacker[3]] = 1

    trap_field = determine_trap_field(game_state, field, explosion_timer, victim, attackers)
    trap_field[trap_field == -1000] = 1000
    if trap_field.min() <= SHOULD_ESCAPE_THRESHOLD:
        bomb = np.unravel_index(np.argmin(trap_field, axis=None), trap_field.shape)
        x, y = victim
        time_to_escape = np.array([
            math.inf if field[x, y - 1] != 0 else determine_time_to_escape((x, y - 1), bomb, field),
            math.inf if field[x, y + 1] != 0 else determine_time_to_escape((x, y + 1), bomb, field),
            math.inf if field[x - 1, y] != 0 else determine_time_to_escape((x - 1, y), bomb, field),
            math.inf if field[x + 1, y] != 0 else determine_time_to_escape((x + 1, y), bomb, field),
        ])
        directions = np.array([0, 0, 0, 0])
        directions[time_to_escape == time_to_escape.min()] = 1
        return True, directions
    else:
        return False, np.array([0, 0, 0, 0])


def determine_trap_escape_direction_improved(game_state, explosion_timer):
    victim = game_state['self'][3]
    attackers = game_state['others']

    field = prepare_field_trap(game_state, explosion_timer)
    for attacker in attackers:
        field[attacker[3]] = 1

    trap_field = determine_trap_field(game_state, field, explosion_timer, victim, attackers)
    trap_field[trap_field == -1000] = 1000
    if trap_field.min() <= SHOULD_ESCAPE_THRESHOLD:
        bomb = np.unravel_index(np.argmin(trap_field, axis=None), trap_field.shape)
        x, y = victim
        time_to_escape = np.array([
            math.inf if field[x, y - 1] != 0 else determine_time_to_escape((x, y - 1), bomb, field),
            math.inf if field[x, y + 1] != 0 else determine_time_to_escape((x, y + 1), bomb, field),
            math.inf if field[x - 1, y] != 0 else determine_time_to_escape((x - 1, y), bomb, field),
            math.inf if field[x + 1, y] != 0 else determine_time_to_escape((x + 1, y), bomb, field),
        ])
        return np.random.choice(np.flatnonzero(time_to_escape == time_to_escape.min()))
    else:
        return 4


def find_path_to_trap(position, field, target):
    if field[position] != 0:
        return math.inf

    todo = [position]
    distances = {position: 1}

    while len(todo) > 0:
        current = todo.pop(0)
        if target[current] <= TRAP_FOLLOW_THRESHOLD:
            return distances[current]

        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if field[x, y] == 0]
        for neighbor in neighbors:
            if neighbor not in distances:
                distances[neighbor] = distances[position] + 1
                todo.append(neighbor)
    return math.inf


def determine_trap_enemy_direction(game_state, explosion_timer):
    field, _ = prepare_field_coins(game_state, explosion_timer)

    result = np.ones_like(field) * 1000

    attackers = [game_state['self']]
    for victim in game_state['others']:
        trap_field = determine_trap_field(game_state, field, explosion_timer, victim[3], attackers)
        for x in range(17):
            for y in range(17):
                result[x, y] = min(result[x, y], trap_field[x, y])

    position = game_state['self'][3]
    x, y = position
    if result[position] <= 0:
        if game_state['self'][2]:
            return 5
        else:
            return 6

    distances = np.array([
        find_path_to_trap((x, y - 1), field, result),
        find_path_to_trap((x, y + 1), field, result),
        find_path_to_trap((x - 1, y), field, result),
        find_path_to_trap((x + 1, y), field, result)
    ])

    if min(distances) < math.inf:
        return np.random.choice(np.flatnonzero(distances == distances.min()))
    return 4


def determine_trap_enemy_directions(game_state, explosion_timer):
    field, _ = prepare_field_coins(game_state, explosion_timer)

    result = np.ones_like(field) * 1000

    attackers = [game_state['self']]
    for victim in game_state['others']:
        trap_field = determine_trap_field(game_state, field, explosion_timer, victim[3], attackers)
        for x in range(17):
            for y in range(17):
                result[x, y] = min(result[x, y], trap_field[x, y])

    position = game_state['self'][3]
    x, y = position
    if result[position] <= 0:
        return np.array([0, 0, 0, 0])

    distances = np.array([
        find_path_to_trap((x, y - 1), field, result),
        find_path_to_trap((x, y + 1), field, result),
        find_path_to_trap((x - 1, y), field, result),
        find_path_to_trap((x + 1, y), field, result)
    ])

    directions = np.array([0, 0, 0, 0])

    if min(distances) < math.inf:
        directions[distances == distances.min()] = 1
        return directions
    return np.array([0, 0, 0, 0])
