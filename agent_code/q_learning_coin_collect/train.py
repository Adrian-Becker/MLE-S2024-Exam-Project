from collections import namedtuple, deque

import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, find_distance_to_coin, ACTION_TO_INDEX, state_to_index, \
    determine_next_action

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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

    reward = 0
    if 'COIN_COLLECTED' in events:
        reward += 3
    elif old_game_state['coins'] and new_game_state['coins']:
        field_old = old_game_state['field']
        for coin in old_game_state['coins']:
            field_old[coin] = 2
        field_new = new_game_state['field']
        for coin in new_game_state['coins']:
            field_new[coin] = 2

        reward += find_distance_to_coin(old_game_state['self'][3], field_old) - \
                  find_distance_to_coin(new_game_state['self'][3], field_new)
        reward -= 0.25
        #reward *= 0.25
        if self.action == 'WAIT':
            reward -= 0.5

    old_feature_index = state_to_index(old_game_state)
    old_action = ACTION_TO_INDEX[self.action]
    new_feature_index = state_to_index(new_game_state)
    new_action = ACTION_TO_INDEX[determine_next_action(new_game_state, self.Q)]

    old_q = self.Q[old_feature_index, old_action]
    self.Q[old_feature_index, old_action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * self.Q[new_feature_index, new_action] - self.Q[old_feature_index][old_action])
    if np.isnan(self.Q[old_feature_index, old_action]):
        print("NAN")

    #print(f"STATE (MEAN: {np.mean(self.Q)}, MAX: {np.max(self.Q)}, MIN: {np.min(self.Q)}")

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.Q, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
