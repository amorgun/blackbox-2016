import interface as bbox
import numpy as np

# initial greedy:  37815.867188
# grredy without repeats: 60347.917969
# grredy without repeats and lookahead for 1: -362.757812
# grredy without repeats and lookahead for 200: 73009.976562
# grredy without repeats and lookahead for 400: 71082.304688
# grredy without repeats and lookahead for 300: 75061.179688
# grredy without repeats and lookahead for 10: 14053.540039
# grredy without repeats and lookahead for 300 and magic: 97046.109375
# see "greedy_states_no_repeat_300fwd2.npy"
n_features = 36
n_actions = 4
max_time = -1


states, actions, rewards = None, None, None
current_idx = 0


def calc_best_action_using_checkpoint():
    checkpoint_id = bbox.create_checkpoint()

    best_action = -1
    best_score = -1e9

    rewards_for_actions = []

    for action in range(n_actions):
        bbox.do_action(action)
        best_next_score = -1e9
        next_id = bbox.create_checkpoint()
        for next_action in range(n_actions):
            for _ in range(300):
                bbox.do_action(next_action)
            best_next_score = max(best_next_score, bbox.get_score())
            bbox.load_from_checkpoint(next_id)

        if best_next_score > best_score:
            best_score = best_next_score
            best_action = action

        rewards_for_actions.append(best_next_score)

        bbox.load_from_checkpoint(checkpoint_id)

    bbox.clear_all_checkpoints()
    return best_action, [i - bbox.get_score() for i in rewards_for_actions]


def prepare_bbox():
    global n_features, n_actions, max_time, states, actions, rewards
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
        states = np.empty((max_time + 100, n_features), dtype=np.float)
        actions = np.empty(max_time + 100, dtype=np.int)
        rewards = np.empty((max_time + 100, n_actions), dtype=np.float)


def run_bbox():
    has_next = 1
    
    prepare_bbox()
    global current_idx, states, actions, rewards
    current_idx = 0
    while has_next:
        best_act, next_rewards = calc_best_action_using_checkpoint()
        # for _ in range(100):
        #     st = bbox.get_state()
        #     states[current_idx] = bbox.get_state()
        #     has_next = bbox.do_action(best_act)
        #     actions[current_idx] = best_act
        #     current_idx += 1

        st = bbox.get_state()
        states[current_idx] = bbox.get_state()
        has_next = bbox.do_action(best_act)
        actions[current_idx] = best_act
        for action in range(n_actions):
            rewards[current_idx, action] = next_rewards[action]
        current_idx += 1

        if bbox.get_time() % 10000 == 0:
            print ("time = %d, score = %f" % (bbox.get_time(), bbox.get_score()))

    print(current_idx)
                                                                    np.save("greedy_states_no_repeathtop", states[:current_idx, :])
    np.save("greedy_actions_no_repeat_300fwd2.npy", actions[:current_idx])
    np.save("greedy_rewards_no_repeat_300fwd2.npy", rewards[:current_idx])
 
    bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
    run_bbox()