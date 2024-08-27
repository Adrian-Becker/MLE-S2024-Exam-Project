from ..ours_k_q_learning_crates_reduced_enemy_information.train import add_custom_events
from features import state_to_features_for_q_learning as state_to_features


def setup_training(self):
    pass


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    features_old = state_to_features(old_game_state)
    features_new = state_to_features(new_game_state)
    add_custom_events(self, old_game_state, self_action, new_game_state, events, features_old, features_new)
    print(f"Events: {events}")
    print(f"New State: {features_new}")
    #for bomb in new_game_state['bombs']:
    #    print(bomb)


def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str,
                               enemy_game_state: dict, enemy_events: list[str]):
    pass


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    pass
