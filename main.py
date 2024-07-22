import logging
import numpy as np
import random
import pickle
import time
import random
import argparse
from utils import *


def splitDataByGroup(X_train, columns, groups):
    base = 1
    val = 0
    n_samples = X_train.shape[0]
    for c in columns:
        val = val + X_train[:, c]*base
        base = base*2
    unique_val = np.unique(val)
    n_splits = len(groups)
    X_splits = []
    for g in groups:
        indx = np.zeros(n_samples, dtype=np.bool_)
        for v in g:
            indx = np.logical_or(indx, val==v)
        X_splits.append(X_train[indx,:])
    return X_splits

N_classes = 2

parser = argparse.ArgumentParser(description='FLDT')

parser.add_argument('-d', '--dataset', type=str, default=None, help='dataset name')
parser.add_argument('-m', '--max_depth', type=int, default=5, help='depth of tree')
args = parser.parse_args()

data_set = args.dataset
MAX_DEPTH= args.max_depth
N_clients = 5

for seed_num in [123, 321, 213]:
# for seed_num in [ 123, 213]:
    np.random.seed(seed_num)
    random.seed(seed_num)
    data = np.load(f"data/{data_set}.npy", allow_pickle=True).item()
    X_total, X_test = data["train"], data["test"]

    N_samples_total = len(X_total)
    N_samples_test = len(X_test)
    N_features = X_total.shape[1]-1
    X, y = X_total[:,:-1], X_total[:,-1]
    X_test, y_test = X_test[:,:-1], X_test[:,-1]

    # X_split = splitDataByGroup(X_total, columns=[0,1,2,3], 
    #         groups=[[6, 2, 9], [7, 0, 4], [3], [1,8], [ 5]])
    while True:
        np.random.shuffle(X_total)
        split_points = np.sort(np.random.randint(N_samples_total, size=N_clients-1))
        X_split = np.split(X_total, split_points)
        len_list = np.array([len(x) for x in X_split])
        if all(len_list>0): break
        else: print(data_set)
    # np.random.shuffle(X_total)
    # split_points = np.sort(np.random.randint(N_samples_total, size=N_clients-1))
    # X_split = np.split(X_total, split_points)
    log_file = f"{data_set}_Nc_{N_clients}_Nf_{N_features}_Ns_{N_samples_total}_IID_{int(time.time())}.log"
    logging.basicConfig(filename=f"log/{log_file}", level=logging.DEBUG,
                    format='%(asctime)s --- %(message)s', 
                        datefmt='%m-%d %H:%M:%S')


    client_list = [Client(x, N_classes=N_classes) for x in X_split]
    server = Server(N_features=N_features, N_clients=N_clients, N_classes=N_classes, MAX_DEPTH=MAX_DEPTH)
    start_tm = time.time()
    while server.depth<MAX_DEPTH and server.curr_layer:
        for n_idx in range(len(server.curr_layer)):
            node_bound_list = []
            ancestor_feature_list = findAncestorFeatures(server.curr_node)
            server.excludeFeature(ancestor_feature_list)
            for f_idx in range(server.N_features-len(ancestor_feature_list)):
    #         for f_idx in tqdm(range(2)):
                sub_info_list = [client_list[c_idx].submitFeatureInfo() for c_idx in server.client_idx_list]
                valid_node_flag = server.collectClientInfo(sub_info_list)
                if not valid_node_flag: break
                if f_idx==0:
                    server.excludeClient()
    #             server.calLowerUpperBounds()
                calLowerUpperBounds(server)
                node_bound_list.append(server.boundsForEachFeature.copy())
                best_feature_idx = server.findBestFeature()
                if best_feature_idx>=0:
    #                 c.growTree(best_feature_idx)
                    break
            not_sub_client_idx_list = np.where(~server.clientSetForEachFeature[best_feature_idx,:])[0]
            sub_info_list = [client_list[c_idx].submitFeatureInfo(best_feature_idx) for c_idx in not_sub_client_idx_list]
            server.collectClientInfo(sub_info_list, not_sub_client_idx_list)
            split_info = server.updateCurrNode(best_feature_idx)
            for c in client_list:
                c.growTree(split_info)
        server.updateCurrLayer()
    end_tm = time.time()
    
    save_data = {"client_list": client_list, "recorder": server.recorder, "time":end_tm-start_tm}
    with open(f'results/{data_set}_{seed_num}_{int(time.time())}.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    
