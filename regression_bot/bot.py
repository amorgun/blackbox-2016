import time

import interface as bbox
import numpy as np

# initial: Level score= 2308.362061   14s
# Optimized: Level score= 2308.362061 6.542621850967407s

# test level
# baseline: 2221.729980
# best: 2246.279541
# best_coefs_score=2560.100830078125_sigma=0.004999999888241291.txt: 2158.130615
# best_coefs_score=2964.60009765625_sigma=0.0010000000474974513.txt: 2259.347900
# star3 - subfit_best_coefs_score=2621.0400390625_sigma=0.009999999776482582.txt: 2621.040039
# star 4-subfit_best_coefs_score=2738.301513671875_sigma=0.009999999776482582.txt: 2738.301514
# star 5-best_coefs_score=2966.489501953125_sigma=0.009999999776482582_level=train_level: 2422.259033
# star 6-best_coefs_score=2964.60009765625_sigma=0.10000000149011612_level=train_level: 2259.347900
# star 7-best_coefs_score=2994.271240234375_sigma=0.009999999776482582_level=train_level:
# star 8-best_coefs_score=2992.164794921875_sigma=0.0010000000474974513_level=train_level:
# star 9-best_coefs_score=3017.848388671875_sigma=0.0010000000474974513_level=train_level: 2389.348633
# star 10-best_coefs_score=2972.124267578125_sigma=9.999999747378752e-05_level=train_level.txt: 2257.179688
# star 13-best_coefs_score=2980.401123046875_sigma=0.0010000000474974513_level=train_level.txt:

def get_action_by_state(state, coefs):
    return np.argmax(np.dot(coefs, state))

n_features = 36
n_actions = 4
max_time = -1

 
def prepare_bbox():
    global n_features, n_actions, max_time
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/test_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
 
def load_regression_coefs(filename):
    coefs = np.loadtxt(filename)
    return coefs

 
def run_bbox():
    
    start_time = time.time()
    
    has_next = 1
    
    prepare_bbox()
    coefs = load_regression_coefs("star 13-best_coefs_score=2980.401123046875_sigma=0.0010000000474974513_level=train_level.txt")
    state = np.ones(n_features + 1)
 
    while has_next:
        state[:-1] = bbox.get_state()
        action = get_action_by_state(state, coefs)
        has_next = bbox.do_action(action)
 
    bbox.finish(verbose=1)

    end_time = time.time()
    print(end_time - start_time)
 
 
if __name__ == "__main__":
    run_bbox()