import interface as bbox
from sklearn.linear_model import SGDRegressor
import numpy as np


class Predictor:
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


    def load(self, filename):
        coefs = np.loadtxt(filename).reshape(self.n_actions, self.n_features + 1)
        self.reg_coefs = coefs[:,:-1]
        self.free_coefs = coefs[:,-1]
        
    def calc_reg_for_action(self, action, state):
        return np.dot(self.reg_coefs[action], state) + self.free_coefs[action]
    
    def get_action_by_state(self, state):
        best_act = -1
        best_val = -1e9
     
        for act in range(self.n_actions):
            val = self.calc_reg_for_action(act, state)
            if val > best_val:
                best_val = val
                best_act = act
     
        return best_act
     
    def run(self):
        has_next = True
        self.load("reg_coefs.txt")
        while has_next:
            state = bbox.get_state()
            action = self.get_action_by_state(state)
            has_next = bbox.do_action(action)
     
        bbox.finish(verbose=1)


 
 
if __name__ == "__main__":
    predictor = Predictor()
    predictor.run()
