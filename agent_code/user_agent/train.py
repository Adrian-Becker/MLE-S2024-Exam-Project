#from features import state_to_features_for_q_learning as state_to_features, determine_explosion_timer, \
#    determine_coin_value_scored, determine_crate_value_scored, prepare_escape_path_fields
from ..ours_p_q_learning_crates_no_mode_reduced.callbacks import state_to_features
from ..ours_p_q_learning_crates_no_mode_reduced.train import add_custom_events, reward_from_events

def setup_training(self):
    pass


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    features_old = state_to_features(old_game_state)
    features_new = state_to_features(new_game_state)
    add_custom_events(self, old_game_state, self_action, new_game_state, events, features_old, features_new)
    #add_custom_events(self, old_game_state, self_action, new_game_state, events, features_old, features_new)
    print(f"Events: {events}")
    print(f"Old State: {features_old}")
    print(f"New State: {features_new}")
    print(f"Rewards: {reward_from_events(self, events)}")

    #explosion_timer = determine_explosion_timer(new_game_state)
    #print(f"Coin direction: {determine_coin_value_scored(x, y, new_game_state, explosion_timer)}")
    #print(f"Crate direction: {determine_crate_value_scored(x, y, new_game_state, explosion_timer)}")





def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str,
                               enemy_game_state: dict, enemy_events: list[str]):
    pass


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    pass
