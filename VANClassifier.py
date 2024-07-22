
from base_utils import *
import copy
import random

def set_seed(seed = 231):
    np.random.seed(seed)
    random.seed(seed)

def getVANRes(X_split,N_features, N_clients, MAX_DEPTH, K_max):
    client_list = [Client(x) for x in X_split]
    server = Server(N_features=N_features, N_clients=N_clients, MAX_DEPTH=MAX_DEPTH)
    while server.depth<MAX_DEPTH:
        for n_idx in range(len(server.curr_layer)):
            ancestor_feature_list = findAncestorFeatures(server.curr_node)
            server.excludeFeature(ancestor_feature_list)
            for f_idx in range(min(server.N_features-len(ancestor_feature_list), K_max+1)):
                sub_info_list = [client_list[c_idx].submitFeatureInfo() for c_idx in server.client_idx_list]
                valid_node_flag = server.collectClientInfo(sub_info_list)
                if f_idx == 0:
                    server.excludeClient()
            best_feature_idx = server.findBestFeatureWithKFeatures()
            
            not_sub_client_idx_list = np.where(~server.clientSetForEachFeature[best_feature_idx,:])[0]
            sub_info_list = [client_list[c_idx].submitFeatureInfo(best_feature_idx) for c_idx in not_sub_client_idx_list]
            server.collectClientInfo(sub_info_list, not_sub_client_idx_list)
            split_info = server.updateCurrNode(best_feature_idx)
            for c in client_list:
                c.growTree(split_info)
        
        server.updateCurrLayer()
    recorder = copy.deepcopy(server.recorder)
    return recorder
    

class VANClassifier():
    def __init__(self, max_depth, N_clients=5):
        self.max_depth = max_depth
        self.N_clients = N_clients
    
    def fit(self, x_train, y_train):
        X_total = np.hstack([x_train, y_train.reshape([-1,1])])
        N_samples_total = len(X_total)
        N_features = X_total.shape[1]-1
        set_seed(seed=123) 
        np.random.shuffle(X_total)
        split_points = np.sort(np.random.randint(N_samples_total, size=self.N_clients-1))
        X_split = np.split(X_total, split_points)
        self.recorder = getVANRes(X_split, N_features, self.N_clients, self.max_depth, K_max=N_features-1)
        
    def predict(self, x):
        return self.recorder.predict(x)