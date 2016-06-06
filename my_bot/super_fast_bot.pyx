# cython: profile=False
# cython: wraparound=False
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False

import time

import interface as bbox
cimport interface as bbox
import numpy as np
cimport numpy as np

# initial: Level score= 2308.362061   14s
# Optimized: Level score= 2308.362061 6.542621850967407s
# Cython: Level score= 2308.362061 1.6427972316741943s
# Cython + flags: Level score= 2308.362061 1.6883609294891357s
# Cython + flags + float pointer: Level score= 2308.362061 0.38703465461730957s


# History:
# (1): train level, sigma = 0.01, 10000 iter, score=2851.02905, 2246.279541 val=2291
# (2): (1) + train level, sigma = 0.001, 10000 iter score=2964.600097, 2259.347900 val=2323
# (3): (1) + test level, sigma = 0.01, 10000 iter score=2787.132812, 2621.040039 val=2276
# (4): (2) + test level, sigma = 0.01, 10000 iter score=2668.353760, 2738.301514
# (5): (3) + train level, sigma = 0.01, 10000 iter score=2966.48950195, 2422.259033
# (6): (2) + train level, sigma = 0.1, 10000 iter score=2964.600097, 2259.347900 (same as (2))
# (7): (4) + train level, sigma = 0.01, 10000 iter score=2994.271240234375, 2478.290283 val=2291
# (8): (5) + train level, sigma = 0.001, 1000 iter score=2992.164795, 2397.651123 val=2256
# (9): (8) + train level, sigma = 0.001, 10000 iter score=3017.848388671875, 2389.348633 val=2256
# (10): (2) + train level, sigma = 0.0001, 10000 iter score=2972.124267578125, 2257.179688 val=2324
# (11): (10) + train level, sigma = 0.001, 10000 iter
# (12): (10) + train level, sigma = 0.005, 10000 iter
# (13): (10) + train level, sigma = 0.00001, 10000 iter score=2980.401123046875, 2254.822754

np.random.seed(111)

cdef:
    int n_features = 36
    int n_actions = 4


cdef int get_action_by_state(float* state, float[:, :] coefs):
    cdef:
        int best_act = 1
        float best_score = -1e9
        int i, j
        float curr_score

    for i in range(n_actions):
        curr_score = 0
        for j in range(n_features + 1):
            curr_score += coefs[i, j] * state[j]
        if curr_score > best_score:
            best_score = curr_score
            best_act = i
    return best_act


cdef prepare_bbox(level):
    global n_features, n_actions

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level(level, verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()


cdef float[:, :] load_regression_coefs(filename):
    coefs = np.loadtxt(filename, dtype=np.float32).reshape(
        n_actions, n_features + 1)
    return coefs


cdef float run_sim(float[:, :] coefs):
    cdef:
        float[37] state
        int i
        float *next_state
        int action
        int has_next = 1

    state[36] = 1
    while has_next:
        next_state = bbox.c_get_state()
        for i in range(n_features):
            state[i] = next_state[i]
        action = get_action_by_state(state, coefs)
        has_next = bbox.c_do_action(action)

    return bbox.c_get_score()


cdef float[:, :] generate_next_coefs(float[:, :] coefs, float sigma):
    shifts = np.random.rand(coefs.shape[0], coefs.shape[1]).astype(np.float32) * sigma
    return np.add(shifts, coefs)

cpdef run_bbox():
    start_time = time.time()

    level = "../levels/train_level.data"
    prepare_bbox(level)

    start = bbox.create_checkpoint()

    cdef:
        float[:, :] coefs = np.loadtxt("star 10-best_coefs_score=2972.124267578125_sigma=9.999999747378752e-05_level=train_level.txt", dtype=np.float32)
        # float[:, :] coefs = load_regression_coefs("reg_coefs.txt")
        float[:, :] best_coefs = coefs
        float[:, :] new_coefs
        float lambda_
        float best_score = run_sim(coefs)
        float new_score
        int n_iters = 10000
        int i
        float sigma = 0.001

    for i in range(n_iters):
        bbox.load_from_checkpoint(start)
        new_coefs = generate_next_coefs(best_coefs, sigma)
        new_score = run_sim(new_coefs)
        print(new_score, best_score)
        if new_score > best_score:
            best_score = new_score
            best_coefs = new_coefs

    bbox.finish(verbose=1)

    end_time = time.time()
    print(end_time - start_time)
    print(best_score)
    np.savetxt("star 13-best_coefs_score={}_sigma={}_level={}.txt".format(best_score, sigma, level[10:-5]),
               best_coefs)


if __name__ == "__main__":
    run_bbox()