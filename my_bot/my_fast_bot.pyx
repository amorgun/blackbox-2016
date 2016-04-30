from sklearn.linear_model import SGDRegressor
import numpy as np
cimport numpy as np
from cpython cimport bool
import interface as bbox
cimport interface as bbox
from libcpp.vector cimport vector

#  without step: Level score= 2308.362061


ctypedef float STATE_ELEMENT
ctypedef np.float64_t NP_STATE_ELEMENT
ctypedef STATE_ELEMENT[:] STATE
ctypedef int ACTION
ctypedef float SCORE
ctypedef int ACTION_RESULT


cdef class Predictor:

    cdef:
        int n_features, n_actions, max_time
        object action_predictors

    def __init__(self):
        self.n_features = 36
        self.n_actions = 4
        self.max_time = -1

        if bbox.is_level_loaded():
            bbox.reset_level()
        else:
            bbox.load_level("../levels/train_level.data", verbose=1)
            self.n_features = bbox.get_num_of_features()
            self.n_actions = bbox.get_num_of_actions()
            self.max_time = bbox.get_max_time()

        self.action_predictors = [
            SGDRegressor(warm_start=True, random_state=i)
            for i in range(self.n_actions)]


    cdef cast_to_numpy_array(self, STATE state):
        return np.frombuffer(state, dtype=STATE_ELEMENT, count=self.n_features)


    cdef void load(self, filename):
        coefs = np.loadtxt(filename).reshape(self.n_actions, self.n_features + 1)
        for idx, clf in enumerate(self.action_predictors):
            clf.coef_ = coefs[idx,:-1]
            clf.intercept_ = coefs[idx,-1].reshape(1, -1)

    cdef ACTION_RESULT make_step(self, STATE old_state, ACTION action):
        cdef:
            SCORE old_score = bbox.c_get_score()
            ACTION_RESULT result = bbox.c_do_action(action)
            SCORE new_score = bbox.c_get_score()
        score_diff = new_score - old_score
        self.action_predictors[action].partial_fit(self.cast_to_numpy_array(old_state), [score_diff])
        return result

    cdef calc_reg_for_action(self, action, state):
        clf = self.action_predictors[action]
        return clf.predict(state)

    cdef get_action_by_state(self, state):
        cdef:
            ACTION best_act = -1
            SCORE best_val = -1e9
            ACTION act

        for act in range(self.n_actions):
            val = self.calc_reg_for_action(act, state)
            if val > best_val:
                best_val = val
                best_act = act

        return best_act

    cdef public run(self):
        cdef:
            bool has_next = True
            ACTION action
            STATE state
            vector[STATE_ELEMENT] state_storage = vector[STATE_ELEMENT](self.n_elements, 0)
            STATE_ELEMENT* next_state_ptr
            int i
        self.load("reg_coefs.txt")
        while has_next:
            next_state_ptr = bbox.c_get_state()
            for i in range(self.n_elements):
                state_storage[i] = next_state_ptr[i]
            state = state_storage
            action = self.get_action_by_state(state)
            has_next = self.make_step(state, action)
        bbox.finish(verbose=1)


def run_bbox():
    predictor = Predictor()
    predictor.run()

if __name__ == "__main__":
    run_bbox()
