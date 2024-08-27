from ..ours_l_q_deep_learning_reduced_features.train import add_custom_events
from ..ours_l_q_deep_learning_reduced_features.callbacks import state_to_features


def setup_training(self):
    pass


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    features_old = state_to_features(old_game_state)
    features_new = state_to_features(new_game_state)
    add_custom_events(self, old_game_state, self_action, new_game_state, events, features_old, features_new)
    print(f"Events: {events}")
    print(f"New State: {features_new}")


def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str,
                               enemy_game_state: dict, enemy_events: list[str]):
    pass


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    pass
