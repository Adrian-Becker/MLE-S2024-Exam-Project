import os
from collections import namedtuple, deque
import time
import datetime

Stat = namedtuple('Stat', ('name', 'key', 'track_average', 'history_size', 'format', 'format_average', 'color', 'unit'))

COLOR_FORMAT = "\u001b[{}m"
BOLD = "\u001b[1m"
RESET = "\u001b[0m"
AVERAGE_OFFSET = len(BOLD + '; ')


class StatsLogger:
    def __init__(self, stats: list[Stat]):
        self.stats = stats
        self.histories = {}
        for stat in stats:
            self.histories[stat[1]] = deque([0], maxlen=stat[3])
        self.histories['time'] = deque([time.time()], maxlen=200)
        self.start_time = time.time()

        self.min_width_a1 = 0
        self.min_width_a2 = 0
        self.min_width_b1 = 0
        self.min_width_b2 = 0

        self._prepare()

    def add(self, key: str, value: int | float):
        """
        Adds a value to the history tracked by the Logger.

        Args:
            key (str): The key belonging to the stat to add a value to, only stats add when initializing the class may
                be used.
            value (int|float): The value to add to the history.
        """
        self.histories[key].append(value)

    def _get_string_average(self, stat):
        average = sum(self.histories[stat[1]]) / len(self.histories[stat[1]])
        return ['avg. ' + stat[0],
                BOLD + COLOR_FORMAT.format(stat[6]) + stat[5].format(average) + stat[7] + RESET + "; "]

    def _get_string_current_value(self, stat):
        return [stat[0],
                COLOR_FORMAT.format(stat[6]) + stat[4].format(self.histories[stat[1]][-1]) + stat[7] + RESET]

    def _prepare(self):
        for stat in self.stats:
            if stat[2]:
                text = self._get_string_average(stat)
                self.min_width_a1 = max(self.min_width_a1, len(text[0]))
                self.min_width_a2 = max(self.min_width_a2, len(text[1]) - AVERAGE_OFFSET)

                text = self._get_string_current_value(stat)
                self.min_width_b1 = max(self.min_width_b1, len(text[0]))
                self.min_width_b2 = max(self.min_width_b2, len(text[1]))

            text = self._get_string_current_value(stat)
            self.min_width_a1 = max(self.min_width_a1, len(text[0]))
            self.min_width_a2 = max(self.min_width_a2, len(text[1]))

    def output(self, current_round: int, iteration: int):
        """
        Outputs the current data of the logger to both the console and every 10 rounds to stats.csv.
        If output looks weird ensure that *tqdm* is disabled and console emulation is enabled in your IDE.
        """
        if current_round % 10 == 0:
            os.system('cls' if os.name == 'nt' else 'clear')

        output = '\033[3;1H'
        current_time = time.time()
        self.histories['time'].append(current_time)
        runtime = int(current_time - self.start_time)
        rounds_per_second = len(self.histories['time']) / (current_time - self.histories['time'][0])
        time_per_thousand_rounds = 1000 / rounds_per_second
        time_per_thousand_rounds /= 60

        output += (BOLD + "Round " + COLOR_FORMAT.format(33) + '{:6d}'.format(current_round) + RESET +
                   ("(" + str(format(iteration)) + ")").rjust(11) + "; " +
                   COLOR_FORMAT.format(33) + '{:5.2f}'.format(rounds_per_second) + " rounds/s" + RESET + "; " +
                   COLOR_FORMAT.format(33) + '{:6.2f}min'.format(
                    time_per_thousand_rounds) + RESET + " per 1000 rounds; " +
                   "runtime = " + COLOR_FORMAT.format(33) + str(datetime.timedelta(seconds=runtime)).rjust(
                    8) + RESET + (' ' * 5) + '\n\n')

        for stat in self.stats:
            output += ' ' * 5 + "- "
            if stat[2]:
                text = self._get_string_average(stat)
                output += text[0].ljust(self.min_width_a1) + ' = ' + text[1].rjust(self.min_width_a2 + AVERAGE_OFFSET)
                text = self._get_string_current_value(stat)
                output += text[0].ljust(self.min_width_b1) + ' = ' + text[1].rjust(self.min_width_b2) + (' ' * 7) + '\n'
            else:
                text = self._get_string_current_value(stat)
                output += text[0].ljust(self.min_width_a1) + ' = ' + text[1].rjust(self.min_width_a2) + (' ' * 7) + '\n'
        print(output)

        if current_round % 10 == 0:
            with open("stats.csv", "a") as stats_file:
                stats_file.write(str(current_round))
                for stat in self.stats:
                    if stat[2]:
                        stats_file.write(
                            ", " + stat[5].format(sum(self.histories[stat[1]]) / len(self.histories[stat[1]])))
                    stats_file.write(", " + stat[4].format(self.histories[stat[1]][-1]))
                stats_file.write('\n')
