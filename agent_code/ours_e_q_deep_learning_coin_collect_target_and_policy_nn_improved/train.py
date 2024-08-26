from collections import namedtuple, deque

import pickle
import random
from typing import List

import numpy as np
import torch
from torch import nn

import events as e
from .callbacks import state_to_features, ACTION_TO_INDEX
from .helper_functions import distance_to_nearest_coins_from_position

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10000  # keep only ... last transitions
MINI_BATCH_SIZE = 128
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

STEPS_BEFORE_SYNC = 10

LEARNING_RATE_OPTIMIZER = 1e-6
DISCOUNT_FACTOR = 0.9
TAU = 0.005

# Events
MOVED_CLOSER_EVENT = "MOVED_CLOSER"
MOVED_FURTHER_AWAY_EVENT = "MOVED_FURTHER_AWAY"
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
    self.round = 0
    self.loss = nn.MSELoss()
    self.optimizer = torch.optim.RMSprop(self.policy_net.parameters()) #torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE_OPTIMIZER, amsgrad=True)


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
    if len(old_game_state['coins']) > 0:
        if self_action == 'WAIT' or old_game_state['self'][3] == old_game_state['self'][3]:
            events.append(WAITED_EVENT)
        if distance_to_nearest_coins_from_position(old_game_state['self'][3], old_game_state) < \
                distance_to_nearest_coins_from_position(new_game_state['self'][3], new_game_state):
            events.append(MOVED_FURTHER_AWAY_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(state_to_features(old_game_state), torch.Tensor([ACTION_TO_INDEX[self_action]]).to(torch.int64),
                   state_to_features(new_game_state),
                   reward_from_events(self, events)))

    self.iteration += 1

    # training
    if len(self.transitions) >= MINI_BATCH_SIZE:
        optimize_network(self, random.sample(self.transitions, MINI_BATCH_SIZE))

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)


def optimize_network(self, batch):
    batch = Transition(*zip(*batch))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                         if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(MINI_BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    if self.iteration % 300 == 0:
        print(f"Loss: {loss}")

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
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
    self.round += 1
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(
    #    Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    print(f"Agent scored {last_game_state['self'][1]} points.")

    # Store the model
    if self.round % 50 == 0:
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.policy_net, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        MOVED_CLOSER_EVENT: 10,
        WAITED_EVENT: -3,
        MOVED_FURTHER_AWAY_EVENT: 0
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return torch.Tensor([reward_sum])
