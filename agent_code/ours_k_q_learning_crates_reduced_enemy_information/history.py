import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class TransitionHistory:
    def __init__(self, history_size):
        self.transitions = deque(maxlen=history_size)

    def append(self, transition: Transition):
        self.transitions.append(transition)

    def __len__(self):
        return len(self.transitions)

    def sample(self, num):
        return random.sample(self.transitions, num)
