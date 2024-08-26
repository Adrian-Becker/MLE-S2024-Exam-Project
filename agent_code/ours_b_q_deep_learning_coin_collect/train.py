from collections import namedtuple, deque

import pickle
from typing import List

import numpy as np
import torch
from torch import nn

import events as e
from .callbacks import state_to_features
from .helper_functions import distance_to_nearest_coins_from_position

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
MINI_BATCH_SIZE = 1
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

FEATURE_SIZE = 4

LEARNING_RATE = 0.05
LEARNING_RATE_OPTIMIZER = 0.01
DISCOUNT_FACTOR = 0.85

# Events
MOVED_CLOSER_EVENT = "MOVED_CLOSER"
WAITED_EVENT = "WAITED"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.iteration = 0
    self.loss = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE_OPTIMIZER)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if 'COIN_COLLECTED' not in events and \
            distance_to_nearest_coins_from_position(old_game_state['self'][3], old_game_state) > \
            distance_to_nearest_coins_from_position(new_game_state['self'][3], new_game_state):
        events.append(MOVED_CLOSER_EVENT)
    if self_action == 'WAIT':
        events.append(WAITED_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))

    self.iteration += 1

    if len(self.transitions) >= MINI_BATCH_SIZE: # and self.iteration % MINI_BATCH_SIZE / 2 == 0:
        input = []
        next_input = []
        rewards = []

        for elem in self.transitions:
            if elem[2] is None:
                continue
            input.append(torch.from_numpy(elem[0]).to(torch.float32))
            next_input.append(torch.from_numpy(elem[2]).to(torch.float32))
            rewards.append(elem[3])

        input = torch.stack(input)
        next_input = torch.stack(next_input)
        rewards = torch.FloatTensor(rewards).to(torch.float32)

        with torch.no_grad():
            q_next = self.model(next_input)
            q_next_iteration = q_next.max()
        current_q = self.model(input)

        selected_actions_matrix = torch.zeros(current_q.shape).scatter(1, current_q.argmax(1).unsqueeze(1), 1.0)

        target_q = (1 - LEARNING_RATE) * current_q + \
                   (LEARNING_RATE * (rewards - torch.max(current_q, 1).values + DISCOUNT_FACTOR * q_next_iteration)[:, None] * selected_actions_matrix)
        loss = self.loss(current_q, target_q)
        if self.iteration % 300 == 0:
            print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        MOVED_CLOSER_EVENT: 1,
        WAITED_EVENT: -.3
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
