from collections import namedtuple, deque
from datetime import time

Stats = namedtuple('Stat', ('name', 'key', 'track_average', 'history_size', 'format', 'color'))


class SatsLogger:
    def __init__(self, stats):
        self.stats = stats
        self.histories = {}
        for stat in stats:
            self.histories[stat['name']] = deque([0], maxlen=stat['history_size'])
        self.histories['time'] = deque([time.time()], maxlen=200)

    def add(self, key, value):
        self.histories[key].append(value)

    def print(self, round):
        print('\033[1;1H')

