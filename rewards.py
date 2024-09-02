def distance_to_nearest_bomb(game_state: dict):
    x, y = game_state['self'][3]
    nearest_distance = 32
    for bomb in game_state['bombs']:
        bombX, bombY = bomb[0]
        distance = abs(x - bombX) + abs(y-bombY)
        if distance < nearest_distance:
            nearest_distance = distance
    return nearest_distance
