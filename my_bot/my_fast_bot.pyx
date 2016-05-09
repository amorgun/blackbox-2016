# cython: profile=False
# cython: wraparound=False
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
import cython

from sklearn.linear_model import SGDRegressor
import numpy as np
cimport numpy as np
from cpython cimport bool
import interface as bbox
cimport interface as bbox
from libc.stdlib cimport rand, RAND_MAX
from scipy.linalg cimport cython_blas as blas

# without step: Level score= 2308.362061
# with batch sgd: Level score= -9512.807617
# predict score: Level score= -13255.810547
# predict diff of score: Level score= -10842.875000
# with delay (gamma = 0.9, n_seps = 10): -11945.239258
# with delay (gamma = 0.9, n_seps = 20): -10932.197266
# with delay (gamma = 0.9, n_seps = 100): -9800.141602
# with delay (gamma = 0.9, n_seps = 200): -10463.736328
# with delay (gamma = 0.9, n_seps = 500): -11083.313477
# with delay (gamma = 0.95, n_seps = 100): -10427.183594
# with delay (gamma = 0.95, n_seps = 500): -10407.957031
# with delay (gamma = 0.8, n_seps = 100): -11011.115234


ctypedef float STATE_ELEMENT
ctypedef np.float_t NP_STATE_ELEMENT_t
ctypedef STATE_ELEMENT[:] STATE
ctypedef int ACTION
ctypedef np.int_t NP_ACTION_t
ctypedef float SCORE
ctypedef np.float_t NP_SCORE_t
ctypedef int ACTION_RESULT

NP_STATE_ELEMENT = np.float
NP_SCORE = np.float
NP_ACTION = np.int


cdef class Predictor:

    cdef:
        # Enviroment info
        int n_actions, max_time, env_state_size

        # Algorithm config
        int n_steps_in_state, state_size
        int train_batch_size # Размер batch-а для обучения классификатора
        float gamma # Cкорость затухания Q
        int n_steps_delay # Колиество последних действий, по которым считается Q
        int history_size # Backlog для восстановления предыдущих состоний
        float eps # Параметр эпсилон-жадной стратегии

        # Algorithm implementation
        object action_predictors

        NP_STATE_ELEMENT_t[:,:] state_history
        NP_SCORE_t[:] reward_history
        NP_ACTION_t[:] action_history

        NP_STATE_ELEMENT_t[:] current_state
        int current_state_num

        int global_state_num

        NP_STATE_ELEMENT_t[:,:,:] actions_training_examples
        NP_SCORE_t[:, :]  actions_training_results
        np.int_t[:] actions_training_examples_positions

    def __init__(self):
        # Algorithm config
        self.n_steps_in_state = 4
        self.gamma = 0.9
        self.n_steps_delay = 100
        self.train_batch_size = 100
        self.eps = 0.2
        self.global_state_num = 0

        self.env_state_size, self.n_actions, self.max_time = \
            self.get_env_info()

        self.action_predictors = [
            SGDRegressor(warm_start=True, random_state=i)
            for i in range(self.n_actions)]

        self.state_size = self.n_steps_in_state * self.env_state_size

        self.history_size = self.n_steps_delay + self.n_steps_in_state
        self.state_history = np.zeros(
            (self.history_size, self.state_size),
            dtype=NP_STATE_ELEMENT)
        self.reward_history = np.empty(self.history_size, dtype=NP_SCORE)
        self.action_history = np.empty(self.history_size, dtype=NP_ACTION)

        self.actions_training_examples= np.empty(
            (self.n_actions, self.train_batch_size, self.state_size),
            dtype=NP_STATE_ELEMENT)
        self.actions_training_results = np.empty(
            (self.n_actions, self.train_batch_size),
            dtype=NP_SCORE)
        self.actions_training_examples_positions = \
            np.zeros(self.n_actions, dtype=np.int)

        self.current_state_num = self.history_size - 1

        self.load_coeffs("reg_coefs.txt")

    cdef get_env_info(self):
        cdef:
            int env_state_size = 36
            int n_actions = 4
            int max_time = -1
        if bbox.is_level_loaded():
            bbox.reset_level()
        else:
            bbox.load_level("../levels/train_level.data", verbose=1)
            env_state_size = bbox.get_num_of_features()
            n_actions = bbox.get_num_of_actions()
            max_time = bbox.get_max_time()
        return env_state_size, n_actions, max_time

    cdef void load_coeffs(self, str filename):
        cdef np.ndarray[np.float_t, ndim=2] coefs = \
            np.loadtxt(filename).reshape(
                self.n_actions, self.env_state_size + 1)
        for idx, clf in enumerate(self.action_predictors):
            clf.coef_ = np.zeros(self.state_size)
            clf.coef_[0:self.env_state_size] = coefs[idx,:-1]
            clf.coef_[self.env_state_size:] = 0
            clf.intercept_ = coefs[idx,-1].reshape(1, -1)

    cdef ACTION_RESULT make_step(self, ACTION action):
        cdef:
            SCORE old_score = bbox.c_get_score()
            ACTION_RESULT result = bbox.c_do_action(action)
            SCORE new_score = bbox.c_get_score()
            SCORE score_diff = new_score - old_score
        self.remember_action_result(action, score_diff)
        self.add_training_example()
        return result

    cdef void add_training_example(self):
        if self.global_state_num <= self.n_steps_delay:
            # Not enough data to create next data row
            return
        cdef:
            int delayed_position = (self.current_state_num
                                    + self.history_size
                                    - self.n_steps_delay) % self.history_size
            ACTION action = self.action_history[delayed_position]
            int batch_position =\
                self.actions_training_examples_positions[action]
            int next_batch_position = \
                (batch_position + 1) % self.train_batch_size
            NP_STATE_ELEMENT_t[:] batch_row = \
                self.actions_training_examples[action, batch_position]
            NP_STATE_ELEMENT_t[:] state = self.state_history[delayed_position]
            NP_SCORE_t[:] action_results = self.actions_training_results[action]
            int i
        for i in range(self.state_size):
            batch_row[i] = state[i]
        action_results[batch_position] = self.calculate_delayed_reward(
            self.current_state_num)
        if next_batch_position == 0:
            # Batch full
            self.train_predictor_for_action(action)
        self.actions_training_examples_positions[action] = next_batch_position

    cdef void remember_action_result(self, ACTION action, SCORE result):
        self.action_history[self.current_state_num] = action
        self.reward_history[self.current_state_num] = result

    cdef SCORE calculate_delayed_reward(self, int current_state_num):
        cdef:
            int i
            SCORE result = 0
        for i in range(self.n_steps_delay):
            result = result * self.gamma + self.reward_history[
                (current_state_num + self.history_size - i) % self.history_size
            ]
        return result

    cdef void train_predictor_for_action(self, ACTION action):
        cdef:
            predictor = self.action_predictors[action]
            NP_STATE_ELEMENT_t[:, :] train_features = \
                self.actions_training_examples[action]
            NP_SCORE_t[:] train_results = self.actions_training_results[action]
        # print("Train predictor #{}".format(action))
        X = np.frombuffer(
            train_features, dtype=NP_STATE_ELEMENT,
            count=(self.state_size * self.train_batch_size)).reshape(
            (self.train_batch_size, self.state_size))
        y = np.frombuffer(
            train_results, dtype=NP_SCORE, count=self.train_batch_size)
        try:
            predictor.partial_fit(X, y)
        except ValueError:
            print(X, y)
            print(np.max(X), np.min(X), np.max(y), np.min(y))
            raise

    cdef SCORE calc_reg_for_action(self, ACTION action):
        cdef:
            clf = self.action_predictors[action]
            np.float_t[:] coef = clf.coef_
            np.float_t intercept = clf.intercept_
            SCORE result = 0
            int i = 0
        for i in range(self.state_size):
            result += coef[i] * self.current_state[i]
        return result + intercept

    cdef get_action(self):
        cdef:
            ACTION best_act = -1
            SCORE best_val = -1e9
            ACTION act
            SCORE val

        for act in range(self.n_actions):
            val = self.calc_reg_for_action(act)
            if val > best_val:
                best_val = val
                best_act = act
        return best_act

    cdef void get_next_state(self):
        cdef:
            STATE_ELEMENT* next_state_ptr = bbox.c_get_state()
            int next_state_num = (
                (self.current_state_num + 1) % self.history_size)
            int i
        for i in range(self.state_size - self.env_state_size):
            self.state_history[next_state_num, i + self.env_state_size] = \
                self.state_history[self.current_state_num, i]
        for i in range(self.env_state_size):
            self.state_history[next_state_num, i] = next_state_ptr[i]
        self.current_state_num = next_state_num
        self.global_state_num += 1
        self.current_state = self.state_history[self.current_state_num]

    cdef public run(self):
        cdef:
            int has_next = 1
            ACTION action
        while has_next:
            self.get_next_state()
            action = self.get_action()
            has_next = self.make_step(action)
        bbox.finish(verbose=1)


def run_bbox():
    predictor = Predictor()
    predictor.run()

if __name__ == "__main__":
    run_bbox()
