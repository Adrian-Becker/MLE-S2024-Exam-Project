import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, find_distance_to_coin, ACTION_TO_INDEX, breadth_first_search, \
    determine_next_action, prepare_field_coins

from .history import TransitionHistory, Transition

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
MOVED_TOWARDS_COIN_EVENT = "Moved Towards Coin"
MOVED_AWAY_FROM_COIN_EVENT = "Moved Away from Coin"

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95

GAME_REWARDS = {
    e.COIN_COLLECTED: 100,
    e.KILLED_OPPONENT: 100,
    e.KILLED_SELF: -300,
    e.GOT_KILLED: -150,
    e.INVALID_ACTION: -10,
    e.WAITED: -5,
    e.BOMB_DROPPED: 5,
    e.CRATE_DESTROYED: 40,
    e.COIN_FOUND: 20,
    MOVED_TOWARDS_COIN_EVENT: 10,
    MOVED_AWAY_FROM_COIN_EVENT: -7
}


def setup_training(self):
    """
    Initialise agent for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.transitions = TransitionHistory(TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    Updates the Q-table based on how well the agent did in the current round.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    features_old = state_to_features(old_game_state)
    action_old = ACTION_TO_INDEX[self_action]
    features_new = state_to_features(new_game_state)
    action_new = ACTION_TO_INDEX[determine_next_action(new_game_state, self.Q)]

    if e.COIN_COLLECTED not in events:
        # no coin collected => agent might have moved closer to coin
        distance_old = breadth_first_search(old_game_state['self'][3], *prepare_field_coins(old_game_state))
        distance_new = breadth_first_search(new_game_state['self'][3], *prepare_field_coins(new_game_state))
        if distance_old > distance_new:
            events.append(MOVED_TOWARDS_COIN_EVENT)
        elif distance_new > distance_old:
            events.append(MOVED_AWAY_FROM_COIN_EVENT)

    rewards = reward_from_events(self, events)

    self.Q[features_old, action_old] += LEARNING_RATE * (
            rewards + DISCOUNT_FACTOR * self.Q[features_new, action_new] - self.Q[features_old, action_old])

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(features_old, action_old, features_new, rewards))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of the round, saves the current state of the Q-table so that it can be restored after training.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(
    #    Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("q-table.pt", "wb") as file:
        pickle.dump(self.Q, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculated the rewards for a given action based on its event list.
    Rewards are defined in the global variable GAME_REWARDS.
    """
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
