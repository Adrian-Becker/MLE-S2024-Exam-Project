from ..ours_k_q_learning_crates_reduced_enemy_information.callbacks import state_to_features


def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    print("=== Turn ===")
    print(state_to_features(game_state))
    return game_state['user_input']
