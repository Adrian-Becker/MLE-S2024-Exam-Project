import numpy as np

COLOR_FORMAT = "\u001b[{}m"
DEFAULT_COLOR_MAP = {
    -2: 31,
    -1: 30,
    0: 32,
    1: 31,
    2: 33,
    3: 34,
    4: 35,
    5: 36,
    6: 37
}


def print_field(field, color_map=None):
    if color_map is None:
        color_map = DEFAULT_COLOR_MAP

    output = ''
    for i in range(0, 17):
        for j in range(0, 17):
            value = field[j][i]
            color = color_map[np.clip(value, -2, 6)]
            output += COLOR_FORMAT.format(color) + str(value).rjust(3)
        output += '\n'
    output += '\u001b[0m'
    print(output)
