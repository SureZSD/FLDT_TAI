import numpy as np
import cvxpy as cvx
import logging
from termcolor import colored

np_eps=np.finfo(np.float32).eps

def calBoundFrom4Tuples(tup):
    N, N0, Nl, Nl0 = tup
    left = Nl0**2/Nl if Nl>0 else 0
    Nr, Nr0 = N-Nl, N0-Nl0
    right = Nr0**2/Nr if Nr>0 else 0
    return left+right

class TreeNode:
    def __init__(self, data_idx=None, left=None, right=None, feature=None, parent=None, label=None):
        self.data_idx = data_idx
        self.left = left
        self.right = right
        self.feature = feature
        self.parent = parent
        self.label = label
        
        
def calGiniFromTuples(tup):
    # tup: [N_t1, N_t2, ..., N_tk, N_tl1, ..., N_tlK]
    n_elem = len(tup)//2
    N_t_vec = tup[:n_elem]
    N_tl_vec = tup[n_elem:]
    N_tr_vec = N_t_vec - N_tl_vec
    N_t = np.sum(N_t_vec)
    N_tl = np.sum(N_tl_vec)
    N_tr = N_t - N_tl
    
    left = np.sum(N_tl_vec**2)/N_tl if N_tl>0 else 0
    right = np.sum(N_tr_vec**2)/N_tr if N_tr>0 else 0
    return 1-1/N_t*(left+right) if N_t>0 else 0

def calBoundFromTuples(tup):
    # tup: [N_t1, N_t2, ..., N_tk, N_tl1, ..., N_tlK]
    n_elem = len(tup)//2
    N_t_vec = tup[:n_elem]
    N_tl_vec = tup[n_elem:]
    N_tr_vec = N_t_vec - N_tl_vec
    N_t = np.sum(N_t_vec)
    N_tl = np.sum(N_tl_vec)
    N_tr = N_t - N_tl
    
    left = np.sum(N_tl_vec**2)/N_tl if N_tl>0 else 0
    right = np.sum(N_tr_vec**2)/N_tr if N_tr>0 else 0
    return left+right

def findAncestorFeatures(node):
    feature_list = []
    while node.parent:
        feature_list.append(node.parent.feature)
        node = node.parent
    return feature_list

def getFeatureFlagWithMaxBound(self):
    max_lower_bound = np.max(self.boundsForEachFeature[1,:])
    flags = np.abs(self.boundsForEachFeature[1,:] - max_lower_bound)<1E-4
    return max_lower_bound, flags

def greaterWithMargin(a, b, eps=1E-4):
    return a>b+eps

class LogRecorder:
    def __init__(self):
        # ------ Updated in collectClientInof() ------
        # The feature index selected by each client in each round.
        self.feature_idx_from_each_client=[]
        self.client_idx_list = []
        # The samples in each node
        self.node_info = []
        # Whether the node contains any samples
        self.complete_tree_list = []
        # ----------------------------------------------
        
        
        # ------ Updated in calLowerUpperBounds() ------
        # The lower and upper bound of Gini for each feature in each round
        self.bounds_for_each_feature = []
        # ----------------------------------------------
        
        # ------ This should be updated in updateInfo() ------
        # The clients number that have submit info about feature f when f_best is determined
        self.feature_cnt = []
        # The number of features submitted by each client when f_best is determined
        self.submit_feature_no_list = []
        # The best feature to split the node
        self.split_feature_idx_list = []
        self.tree = []
        self.addOneItem()
        
    def addOneItem(self):
        self.bounds_for_each_feature.append([])
        self.feature_idx_from_each_client.append([])
        self.client_idx_list.append([])
    
    @property
    def depth(self):
        return int(np.log2(len(self.split_feature_indx_list)+1))
    
    def predict(self, X):
        y_pred = []
        for sample_idx in range(len(X)):
            node = self.tree[0]
            while node.label is None:
                split_feature = node.feature
                feature_value = X[sample_idx, split_feature]
                node = node.left if feature_value<0.5 else node.right
            y_pred.append(node.label)
        return np.array(y_pred)
    
    
def estLowerBoundsForAll(self):
    # feature_selected_flag: whether at least a client has submit a given feature
    # boundsForEachFeature: 0-row==>upper bounds, 1-row==> lower bounds
    # feature_flag_list: whether the given feature is valid for current node (if it is used by the ancestor, it is invalid)
    # sampleForEachFeature: upper half consists of distributions of samples over each class in parent node, lower half 
    #                       consists of samples in left node
    feature_selected_flag = np.sum(self.clientSetForEachFeature, axis=1)>0
    unselect_feature_cal_flag = False
    
    prev_max_lower_bound = np.max(self.boundsForEachFeature[1,:])
    
    for feature_idx in np.where(self.feature_flag_list)[0]:
        if feature_selected_flag[feature_idx] or not unselect_feature_cal_flag:    
            # Number of clients not submit this feature
            n_client_not_def = np.sum(~self.clientSetForEachFeature[feature_idx])
            idx_client_not_def = np.where(~self.clientSetForEachFeature[feature_idx])[0]
            
            if n_client_not_def==0:
                lower_bounds = calBoundFromTuples(self.sampleForEachFeature[:, feature_idx])
                self.boundsForEachFeature[1, feature_idx] = lower_bounds
                continue

            # Define Necessary Variables
            # left_dist_not_def: variables that represents the sample number of each class at each client in left node
            # left_sample_no_not_def: sample number at each client in left node
            # left_sample_total_not_def: sample number of all clients in left node
            # left_sample_each_class_not_def: sample number of all clients for each class in left node
            left_dist_not_def = [cvx.Variable(self.N_classes, integer=True) for idx in range(n_client_not_def)]
            left_sample_no_not_def = [cvx.sum(x) for x in left_dist_not_def]
            left_dist_def = self.sampleForEachFeature[self.N_classes:, feature_idx]
            left_sample_total_not_def = cvx.sum(left_sample_no_not_def)
            left_sample_total = left_sample_total_not_def+np.sum(left_dist_def)
            left_sample_each_class_not_def = [cvx.sum([x[idx] for x in left_dist_not_def]) for idx in range(self.N_classes)]

            right_dist_not_def = [self.sampleForEachClient[:,c_idx]-x for c_idx, x in zip(idx_client_not_def, left_dist_not_def)]
            right_sample_no_not_def = [cvx.sum(x) for x in right_dist_not_def]
            right_dist_def = self.sampleForEachFeature[:self.N_classes, feature_idx]-left_dist_def
            right_sample_total_not_def = cvx.sum(right_sample_no_not_def)
            right_sample_total = right_sample_total_not_def+np.sum(right_dist_def)
            right_sample_each_class_not_def = [cvx.sum([x[idx] for x in right_dist_not_def]) for idx in range(self.N_classes)]

            naive_bounds = self.sampleForEachClient[:, idx_client_not_def]
            gini_bounds = self.boundsForEachClient[idx_client_not_def]

            # ------------ Lower Bound Estimation --------------
            cons = [ left_dist_not_def[idx]>=0 for idx in range(n_client_not_def)] + \
                   [ left_dist_not_def[idx]<=naive_bounds[:, idx] for idx in range(n_client_not_def)]+\
                   [ cvx.quad_over_lin(left_dist_not_def[idx], left_sample_no_not_def[idx]+np_eps) \
                        + cvx.quad_over_lin(right_dist_not_def[idx], right_sample_no_not_def[idx]+np_eps) \
                      <= gini_bounds[idx] for idx in range(n_client_not_def)
                   ]
            # quad_over_lin does not accept an expression as the input
            obj = cvx.Minimize(cvx.sum([cvx.quad_over_lin(x_not_def+x_def, left_sample_total+np_eps ) for x_not_def, x_def in zip(left_sample_each_class_not_def, left_dist_def)]
                            +[cvx.quad_over_lin(x_not_def+x_def, right_sample_total+np_eps ) for x_not_def, x_def in zip(right_sample_each_class_not_def, right_dist_def)]))
            
            prob = cvx.Problem(obj, cons)
            prob.solve(reoptimize=True, solver=cvx.GUROBI)
            lower_bound = prob.value
#             print(f"Feature {feature_idx}: {[x.value for x in left_sample_each_class_not_def]}")
            self.boundsForEachFeature[1, feature_idx] = lower_bound
            if not feature_selected_flag[feature_idx]:
                unselect_feature_cal_flag = True
                lower_bound_for_unselected_feature = lower_bound
        else:
            self.boundsForEachFeature[1, feature_idx] = lower_bound_for_unselected_feature
    print(f"Submitted Feature: {self.curr_feature_no}: ----- Lower Bound Estimation Finished -----")
    max_lower_bound = np.max(self.boundsForEachFeature[1,:])
    return greaterWithMargin(max_lower_bound, prev_max_lower_bound,eps=-1E-4)


def estUpperBoundsForAll(self, max_lower_bound,feature_flags_with_max_lower_bound, larger_than_previous=False):
    feature_selected_flag = np.sum(self.clientSetForEachFeature, axis=1)>0
    unselect_feature_cal_flag = False
    continue_flag = True
    canditate_feature_idx_list = np.where(self.feature_flag_list)[0]
    # tqdm_iterator = tqdm(canditate_feature_idx_list, leave=False)
#     for feature_idx in tqdm(np.where(self.feature_flag_list)[0], leave=False):
    for num_features, feature_idx in enumerate(canditate_feature_idx_list):
        if larger_than_previous and self.optimalCondFlags[feature_idx]:
            continue
        if feature_selected_flag[feature_idx] or not unselect_feature_cal_flag:    
            # Number of clients not submit this feature
            n_client_not_def = np.sum(~self.clientSetForEachFeature[feature_idx])
            idx_client_not_def = np.where(~self.clientSetForEachFeature[feature_idx])[0]
            if n_client_not_def==0:
                upper_bounds = calBoundFromTuples(self.sampleForEachFeature[:, feature_idx])
                self.boundsForEachFeature[0, feature_idx] = upper_bounds
                continue

            # Define Necessary Variables
            left_dist_not_def = [cvx.Variable(self.N_classes, integer=True) for idx in range(n_client_not_def)]
            left_sample_no_not_def = [cvx.sum(x) for x in left_dist_not_def]
            left_dist_def = self.sampleForEachFeature[self.N_classes:, feature_idx]
            left_sample_total_not_def = cvx.sum(left_sample_no_not_def)
            left_sample_total = left_sample_total_not_def+np.sum(left_dist_def)
            left_sample_each_class_not_def = [cvx.sum([x[idx] for x in left_dist_not_def]) for idx in range(self.N_classes)]

            right_dist_not_def = [self.sampleForEachClient[:,c_idx]-x for c_idx, x in zip(idx_client_not_def, left_dist_not_def)]
            right_sample_no_not_def = [cvx.sum(x) for x in right_dist_not_def]
            right_dist_def = self.sampleForEachFeature[:self.N_classes, feature_idx]-left_dist_def
            right_sample_total_not_def = cvx.sum(right_sample_no_not_def)
            right_sample_total = right_sample_total_not_def+np.sum(right_dist_def)
            right_sample_each_class_not_def = [cvx.sum([x[idx] for x in right_dist_not_def]) for idx in range(self.N_classes)]

            naive_bounds = self.sampleForEachClient[:, idx_client_not_def]
            gini_bounds = self.boundsForEachClient[idx_client_not_def]
            search_interval_upper_bound = int(np.sum(naive_bounds))
            
            
            total_dist = np.sum(self.sampleForEachClient, axis=1)
            upper_bound_list = []
            sub_feat_flag = feature_idx not in self.curr_collect_feature_set
            
            cons = [ left_dist_not_def[idx]>=0 for idx in range(n_client_not_def)] + \
                   [ left_dist_not_def[idx]<=naive_bounds[:, idx] for idx in range(n_client_not_def)]+\
                   [ cvx.quad_over_lin(left_dist_not_def[idx], left_sample_no_not_def[idx]+np_eps) \
                        + cvx.quad_over_lin(right_dist_not_def[idx], right_sample_no_not_def[idx]+np_eps) \
                      <= gini_bounds[idx] for idx in range(n_client_not_def)
                   ]
            
            cons.append(left_sample_total_not_def==0)
            # ----------- Upper Bound Estimation ---------------
            for unknown_samples_in_left_node in range(search_interval_upper_bound+1):
                if  larger_than_previous and sub_feat_flag and self.subFeatureCheckFlag[feature_idx, unknown_samples_in_left_node]: 
#                     upper_bound_list.append(self.boundsForEachFeature[0, feature_idx])
                    upper_bound_list.append(self.subFeatureCheckVal[feature_idx, unknown_samples_in_left_node])
                    continue
                if not sub_feat_flag:
                    self.subFeatureCheckFlag[feature_idx, :] = False
                if unknown_samples_in_left_node==0:
                    upper_bound = calBoundFromTuples(np.concatenate([total_dist, left_dist_def]))
                elif unknown_samples_in_left_node==search_interval_upper_bound:
                    upper_bound = calBoundFromTuples(np.concatenate([total_dist, left_dist_def+np.sum(naive_bounds, axis=1)]))
                else:
                    _ = cons.pop()
                    cons.append(left_sample_total_not_def==unknown_samples_in_left_node)
                    obj = cvx.Minimize(left_sample_each_class_not_def[0])
                    prob_quad_low = cvx.Problem(obj, cons)
                    try:
                        prob_quad_low.solve(reoptimize=True, solver=cvx.GUROBI)
                        low_value = prob_quad_low.value
                        low_value = calBoundFromTuples(np.concatenate([total_dist, left_dist_def+np.array([low_value, unknown_samples_in_left_node-low_value])]))
                    except:
                        low_value = 0.    
                    
                    obj = cvx.Maximize(left_sample_each_class_not_def[0])
                    prob_quad_up = cvx.Problem(obj, cons)
                    try:
                        prob_quad_up.solve(reoptimize=True, solver=cvx.GUROBI)
                        up_value = prob_quad_up.value
                        up_value = calBoundFromTuples(np.concatenate([total_dist, left_dist_def+np.array([up_value, unknown_samples_in_left_node-up_value])]))
                    except:
                        up_value = 0.
                    
                    upper_bound = max(low_value, up_value)
                self.subFeatureCheckVal[feature_idx, unknown_samples_in_left_node] = upper_bound
                upper_bound_list.append(upper_bound)
                if greaterWithMargin(upper_bound, max_lower_bound):
                    if feature_flags_with_max_lower_bound[feature_idx] and \
                      (num_features<1 or 
                        greaterWithMargin(max_lower_bound, np.max(self.boundsForEachFeature[0, canditate_feature_idx_list[:num_features]]))):
                        pass
                    else:
                        continue_flag = False
                        logging.info(colored(f"Early Stop at Feature {feature_idx} for fix value {unknown_samples_in_left_node}",'blue'))
                        break
                else:
                    self.subFeatureCheckFlag[feature_idx, unknown_samples_in_left_node] = True
            max_upper_bound = np.max(upper_bound_list)
            self.optimalCondFlags[feature_idx] = greaterWithMargin(max_lower_bound, max_upper_bound)
            self.boundsForEachFeature[0, feature_idx] = max_upper_bound
            if not feature_selected_flag[feature_idx]:
                unselect_feature_cal_flag = True
                upper_bound_for_unselected_feature = max_upper_bound
        else:
            self.boundsForEachFeature[0, feature_idx] = upper_bound_for_unselected_feature
        if not continue_flag:
            # tqdm_iterator.container.close()
            break
    logging.info(f"Submitted Feature: {self.curr_feature_no}: ----- Lower Bound Estimation Finished -----")
#     print(f"Submitted Feature: {self.curr_feature_no}: ----- Upper Bound Estimation Finished -----")
    return True


def calLowerUpperBounds(self):
    # For the first-round communication, no computation would be performed,
    # as there is no sufficient information to find the best splitting feature.
#     if self.N_samples_for_curr_node==25:
#         import ipdb
#         ipdb.set_trace()
    if self.curr_feature_no<=1:
        self.subFeatureCheckFlag = np.zeros((self.N_features, int(self.N_samples_for_curr_node+1)), dtype=np.bool_)
        self.subFeatureCheckVal = np.zeros((self.N_features, int(self.N_samples_for_curr_node+1)))
        logging.info(colored(f"Samples for current node: {self.N_samples_for_curr_node}",'red'))
        print("Please update information for new features")
        return
    larger_flag = estLowerBoundsForAll(self)
    max_lower_bound, feature_flags_with_max_lower_bound = getFeatureFlagWithMaxBound(self)
    flag = estUpperBoundsForAll(self, max_lower_bound, feature_flags_with_max_lower_bound, larger_flag)
    self.recorder.bounds_for_each_feature[-1].append(self.boundsForEachFeature.copy())
    
    
class Server:
    def __init__(self, N_features, N_clients, N_classes, MAX_DEPTH):
        self.N_features = N_features
        self.N_clients = N_clients
        self.N_classes = N_classes
        self.MAX_DEPTH = MAX_DEPTH
        self.root = TreeNode()
        self.curr_layer = [self.root]
        self.next_layer = []
        self.curr_node_idx = 0
        self.depth = 0
        self.split_node_no = 0
        self.curr_node = self.curr_layer[self.curr_node_idx]
        self.recorder = LogRecorder()
        self._resetStatus()

    def _resetStatus(self):
        self.curr_feature_no = 0
        self.feature_flag_list = np.ones([1, self.N_features]).astype(np.bool_).squeeze()
        self.client_idx_list = np.arange(self.N_clients)
        self.N_samples_for_curr_node = 0
        self.N0_samples_for_curr_node = 0
        self.giniLowerBoundForEachClient = np.zeros(self.N_clients)
        # N_features x N_clients
        # i-th row: which client has submitted the information about i-th feature
        self.clientSetForEachFeature = np.zeros((self.N_features, self.N_clients), dtype=np.bool_)
        # N_classesxN_clients --  ---Nt1---; ---Nt2---; ...; ---NtK---;
        self.sampleForEachClient = np.zeros((self.N_classes, self.N_clients))
        # 4xN_features -- 1st row: Nt; 2nd row: Nt0; 3rd row: Nl, 4th row: Nl0 
        self.sampleForEachFeature = np.zeros((2*self.N_classes, self.N_features))
        # Nl0^2/Nl+Nr0^2/Nr for each client
        self.boundsForEachClient = np.zeros(self.N_clients)
        self.boundsForEachFeature = np.zeros((2, self.N_features))
        self.boundsForEachFeature[0,:] = np.inf
        self.boundsForEachFeature[1,:] = -0.1
        self.optimalCondFlags = np.zeros(self.N_features, dtype=np.bool_)
        self.searchIntervalForEachFeature = np.zeros((2, self.N_features))
        self.submitFeatureNoEachClient=np.zeros((self.N_clients,), dtype=np.int_)
    
    def excludeFeature(self, ancestor_feature_list):
        self.feature_flag_list[ancestor_feature_list] = False
        
    def excludeClient(self):
        client_total_sample = np.sum(self.sampleForEachClient, axis=0)
        self.client_idx_list = np.where(client_total_sample!=0)[0]
        zero_sample_clent_set = np.where(client_total_sample==0)[0]
        self.clientSetForEachFeature[:,zero_sample_clent_set] = True
        self.recorder.node_info.append([self.N0_samples_for_curr_node, self.N_samples_for_curr_node-self.N0_samples_for_curr_node])
    
    def collectClientInfo(self, sub_info, client_idx_list=None):
        self.curr_feature_no = self.curr_feature_no+1
        if client_idx_list is None: client_idx_list = self.client_idx_list
        else: self.recorder.feature_cnt.append(np.sum(self.clientSetForEachFeature, axis=1))
        feature_idx_list = []
        for idx, info in enumerate(sub_info):
            client_idx = client_idx_list[idx]
            feature_idx = info[0]
            feature_idx_list.append(feature_idx)
            self.submitFeatureNoEachClient[client_idx]+=1
            self.clientSetForEachFeature[info[0]][client_idx] = True
#             self.clientSetForEachFeature[info[0]].add(client_idx)
            self.giniLowerBoundForEachClient[client_idx] = calGiniFromTuples(info[1])
            N_elem = len(info[1])//2
            self.sampleForEachClient[:, client_idx] = info[1][:N_elem]
            self.sampleForEachFeature[:, feature_idx]+= info[1]
            self.boundsForEachClient[client_idx] = calBoundFromTuples(info[1]) # N, N0, Nl, Nl0
        self.curr_collect_feature_set = set(feature_idx_list)
        self.N_samples_for_curr_node = np.sum(self.sampleForEachClient)
        self.recorder.feature_idx_from_each_client[-1].append(feature_idx_list.copy())
        self.recorder.client_idx_list.append(client_idx_list.copy())
        valid_node_flag = self.N_samples_for_curr_node>0
        return valid_node_flag
    
    def isLeaf(self, best_feature, node):
        if node == node.parent.left:
            N_samples = self.sampleForEachFeature[self.N_classes:, best_feature]
        else:
            N_samples = self.sampleForEachFeature[:self.N_classes, best_feature] - self.sampleForEachFeature[self.N_classes:, best_feature]
        flag = np.sum(N_samples>0)<=1 or self.depth==self.MAX_DEPTH-1
        if flag: node.label = np.argmax(N_samples)
        return flag
    

    def getFeatureFlagWithMaxBound(self, boundsForEachFeature):
        max_lower_bound = np.max(boundsForEachFeature[1,:])
        flags = np.abs(boundsForEachFeature[1,:] - max_lower_bound)<1E-4
        return max_lower_bound, flags

    # def findBestFeature(self):
    #     boundsForEachFeature = self.boundsForEachFeature[:, self.feature_flag_list]
    #     feature_idx_list = np.where(self.feature_flag_list)[0]
    #     max_low_idx = np.argmax(boundsForEachFeature[1,:])
    #     max_low_value = boundsForEachFeature[1,max_low_idx]
    #     max_up_value = np.max(np.concatenate([boundsForEachFeature[0,:max_low_idx], boundsForEachFeature[0,max_low_idx+1:]]))
    #     idx = feature_idx_list[max_low_idx] if max_low_value>=max_up_value else -1
    #     return idx
    def findBestFeatureWithKFeatures(self):
        gini_list = [calBoundFrom4Tuples(self.sampleForEachFeature[:, idx]) for idx in range(self.N_features)]
        best_idx = np.where(self.feature_flag_list)[0][0]
        best_val = gini_list[best_idx]
#         best_idx, best_val = 0, gini_list[0]
        for idx in range(len(gini_list)):
            if self.feature_flag_list[idx] and gini_list[idx]>=best_val:
                best_idx = idx
                best_val = gini_list[idx]
        return best_idx #np.argmax(gini_list)

    def findBestFeature(self):
        boundsForEachFeature = self.boundsForEachFeature[:, self.feature_flag_list]
        feature_idx_list = np.where(self.feature_flag_list)[0]

        max_lower_bound, max_lower_flags = self.getFeatureFlagWithMaxBound(boundsForEachFeature)
        max_lower_bound_idx_list = np.where(max_lower_flags)[0]
        if len(max_lower_bound_idx_list)==len(feature_idx_list):
            max_upper_bound = np.max(boundsForEachFeature[0,:])
        else:
            max_upper_bound = np.max(np.delete(boundsForEachFeature, max_lower_bound_idx_list, axis=1)[0])
        flags = boundsForEachFeature[0, max_lower_bound_idx_list]<max_lower_bound+1E-5
        idx = feature_idx_list[max_lower_bound_idx_list[flags][0]] \
            if max_lower_bound>=max_upper_bound-1E-5 and np.any(flags) else -1
        return idx
    
    def updateCurrNode(self, best_feature):
        curr_node = self.curr_node
        curr_node.feature = best_feature
        curr_node.left = TreeNode(parent=curr_node)
        curr_node.right = TreeNode(parent=curr_node)
        if not self.isLeaf(best_feature, curr_node.left):
            self.next_layer.append(curr_node.left)
        if not self.isLeaf(best_feature, curr_node.right):
            self.next_layer.append(curr_node.right)
        self.curr_node_idx += 1
        if self.curr_node_idx<len(self.curr_layer):
            self.curr_node = self.curr_layer[self.curr_node_idx]
        self.split_node_no += 1
        self.recorder.submit_feature_no_list.append(self.submitFeatureNoEachClient)
        self.recorder.split_feature_idx_list.append(best_feature)
        self._resetStatus()
        self.recorder.addOneItem()
        return best_feature, curr_node.left.label, curr_node.right.label
    
    def updateCurrLayer(self):
        self.recorder.tree.extend(self.curr_layer)
        self.curr_layer = self.next_layer
        self.next_layer = []
        self.depth += 1
        self.curr_node_idx = 0
        if self.curr_layer:
            self.curr_node = self.curr_layer[self.curr_node_idx]
    
    
class Client:
    def __init__(self, x, N_classes):
        self.x = x
        self.N_classes = N_classes
        self.N_features = self.x.shape[1]-1
        root = TreeNode(data_idx=np.arange(len(x)))
        self.curr_node = root
        self.curr_node_idx = 0
        self.submit_feature_no_list = []
        self.tree = [root]
        self._resetStatus()
        
    def __len__(self):
        return len(self.x)

    def _getDataDist(self, y):
        if len(y)>0:
            return np.array([np.sum(y==k) for k in range(self.N_classes)])
        else:
            return np.zeros(self.N_classes)
    
    def _resetStatus(self):
        self.curr_submit_idx = 0
        self.curr_report_idx = 0
        self.feature_flag_list = np.ones([1, self.N_features]).astype(np.bool_).squeeze()
    
    # def update(self):
    #     self.curr_node = node
    #     self.curr_report_idx = self.feature_flag_list

    def submitFeatureInfo(self, feature_idx=None, exclude=True):
        if feature_idx is None:
            if self.curr_submit_idx==0:
                self._calGiniListForCurrNode()
                if exclude:
                    ancestor_feature_list = findAncestorFeatures(self.curr_node)
                    self.feature_flag_list[ancestor_feature_list] = False
                self.gini_sort_indx = np.argsort(self.gini_list[self.feature_flag_list])
                # self.gini_sort_indx = np.lexsort([np.arange(np.sum(self.feature_flag_list)), self.gini_list[self.feature_flag_list]])
                self.feature_idx_list = np.where(self.feature_flag_list)[0]
            # while self.curr_report_idx < len(self.gini_sort_indx):#self.N_features:
            #     feature_idx = self.gini_sort_indx[self.curr_report_idx]
            #     self.curr_report_idx += 1
            #     if self.feature_flag_list[feature_idx]:
            #         break
            feature_idx = self.feature_idx_list[self.gini_sort_indx[self.curr_submit_idx]]
        self.curr_submit_idx += 1
        return feature_idx, self.split_info_list[feature_idx] 
    
    def _calGini(self, y):
        if len(y)==0: return 0
        dist = self._getDataDist(y)
        return 1-np.sum((dist/len(y))**2)
    
    def _calGiniForFeature(self, data, feature_idx):
        
        N_total = len(data)
        if N_total==0:return 0, np.zeros(self.N_classes)
        left_data = data[data[:,feature_idx]==0]
        left_gini = self._calGini(left_data[:,-1])
        N_left = len(left_data)
        
        left_dist = self._getDataDist(left_data[:,-1])
        
        right_data = data[data[:,feature_idx]==1]
        right_gini = self._calGini(right_data[:,-1])
        N_right = len(right_data)
        gini = N_left/N_total*left_gini+N_right/N_total*right_gini
        
        return gini, left_dist
    
    def _calGiniListForCurrNode(self):
        gini_list = []
        split_info_list = []
        curr_node = self.curr_node
        x = self.x[curr_node.data_idx]
        N_total = self._getDataDist(x[:,-1])
        for i in range(self.N_features):
            info = self._calGiniForFeature(x, feature_idx=i)
            gini_list.append(info[0])
            split_info_list.append(np.concatenate([N_total, info[1]]))
        self.gini_sort_indx = np.argsort(gini_list)
        self.gini_list = np.array(gini_list)
        self.split_info_list = split_info_list
    
    def growTree(self, split_info):
        feature_idx, left_label, right_label = split_info
        curr_node = self.curr_node
        x_idx_curr_node = curr_node.data_idx
        
        feature_values = self.x[x_idx_curr_node][:, feature_idx]
        left_idx = np.where(feature_values==0)[0]
        right_idx = np.where(feature_values==1)[0]
        left_node = TreeNode(data_idx=x_idx_curr_node[left_idx],
                            parent=curr_node, label=left_label)
        right_node = TreeNode(data_idx=x_idx_curr_node[right_idx],
                            parent=curr_node, label=right_label)
        curr_node.feature = feature_idx
        if left_label is None:
            self.tree.append(left_node)
        if right_label is None:
            self.tree.append(right_node)
        self.curr_node_idx += 1
        if self.curr_node_idx<len(self.tree):
            self.curr_node = self.tree[self.curr_node_idx]
        self.submit_feature_no_list.append(self.curr_submit_idx)
        self._resetStatus()
    
    def counts(self):
        return np.unique(self.x, return_counts=True)