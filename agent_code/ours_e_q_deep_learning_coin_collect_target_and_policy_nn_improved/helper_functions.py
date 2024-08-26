import math
import numpy as np
import torch


def breadth_first_search(position, field: np.array, targets):
    """
    Calculates the distance to the nearest target using BFS.
    :param field: 1 crates, -1 stone walls, 0 free tiles
    :return distance to the nearest target|32 if no reachable target is present
    """
    position = (position[0], position[1])

    if field[position] == 1 or field[position] == -1:
        return 64

    field = np.copy(field)
    for target in targets:
        field[target] = 2

    distance = {position: 1}
    todo = [position]

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

    return 64


def distance_to_nearest_coins_from_position(position, game_state: dict):
    field = game_state['field']
    coins = game_state['coins']
    return breadth_first_search(position, field, coins)


def distance_to_nearest_coins(game_state: dict):
    field = game_state['field']
    coins = game_state['coins']

    position = game_state['self'][3]
    x, y = position

    dist_up = breadth_first_search((x, y - 1), field, coins)
    dist_down = breadth_first_search((x, y + 1), field, coins)
    dist_right = breadth_first_search((x + 1, y), field, coins)
    dist_left = breadth_first_search((x - 1, y), field, coins)

    return torch.Tensor([
        1 / dist_up,  # up
        1 / dist_down,  # down
        1 / dist_right,  # right
        1 / dist_left#,  # left
       # 0 if dist_up < 64 else 1,
       # 0 if dist_down < 64 else 1,
       # 0 if dist_right < 64 else 1,
       # 0 if dist_left < 64 else 1
    ])
