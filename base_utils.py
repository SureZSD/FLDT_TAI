import cvxpy as cvx
import numpy as np
class TreeNode:
    def __init__(self, data_idx=None, left=None, right=None, feature=None, parent=None, label=None):
        self.data_idx = data_idx
        self.left = left
        self.right = right
        self.feature = feature
        self.parent = parent
        self.label = label
        
def calGiniFrom4Tuples(tup):
    N, N0, Nl, Nl0 = tup
    left = Nl0**2/Nl if Nl>0 else 0
    Nr, Nr0 = N-Nl, N0-Nl0
    right = Nr0**2/Nr if Nr>0 else 0
    return 2/N*(N0-left-right) if N>0 else 0

def calBoundFrom4Tuples(tup):
    N, N0, Nl, Nl0 = tup
    left = Nl0**2/Nl if Nl>0 else 0
    Nr, Nr0 = N-Nl, N0-Nl0
    right = Nr0**2/Nr if Nr>0 else 0
    return left+right

def findAncestorFeatures(node):
    feature_list = []
    while node.parent:
        feature_list.append(node.parent.feature)
        node = node.parent
    return feature_list

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
    
np_eps=np.finfo(np.float32).eps
class Server:
    def __init__(self, N_features, N_clients, MAX_DEPTH):
        self.N_features = N_features
        self.N_clients = N_clients
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
        # 2xN_clients --  1st row: Nt for each client; 2nd row Nt0 for each client.
        self.sampleForEachClient = np.zeros((2, self.N_clients))
        # 4xN_features -- 1st row: Nt; 2nd row: Nt0; 3rd row: Nl, 4th row: Nl0 
        self.sampleForEachFeature = np.zeros((4, self.N_features))
        # Nl0^2/Nl+Nr0^2/Nr for each client
        self.boundsForEachClient = np.zeros(self.N_clients)
        self.boundsForEachFeature = np.zeros((2, self.N_features))
        self.boundsForEachFeature[1,:] = -0.1
        self.searchIntervalForEachFeature = np.zeros((2, self.N_features))
        self.submitFeatureNoEachClient=np.zeros((self.N_clients,), dtype=np.int_)
    
    def excludeFeature(self, ancestor_feature_list):
        self.feature_flag_list[ancestor_feature_list] = False
        
    def excludeClient(self):
        self.client_idx_list = np.where(self.sampleForEachClient[0,:]!=0)[0]
        zero_sample_clent_set = np.where(self.sampleForEachClient[0,:]==0)[0]
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
            self.giniLowerBoundForEachClient[client_idx] = calGiniFrom4Tuples(info[1])
            N, N0, Nl, Nl0 = info[1]
            self.sampleForEachClient[0, client_idx] = N #info[1][0]
            self.sampleForEachClient[1, client_idx] = N0 #info[1][1]
            self.sampleForEachFeature[0, feature_idx]+= N #info[1][2]
            self.sampleForEachFeature[1, feature_idx]+= N0 #info[1][3]
            self.sampleForEachFeature[2, feature_idx]+= Nl #info[1][2]
            self.sampleForEachFeature[3, feature_idx]+= Nl0 #info[1][3]
            self.boundsForEachClient[client_idx] = calBoundFrom4Tuples(info[1]) # N, N0, Nl, Nl0
        self.N_samples_for_curr_node, self.N0_samples_for_curr_node = np.sum(self.sampleForEachClient, axis=1)
        self.recorder.feature_idx_from_each_client[-1].append(feature_idx_list.copy())
        self.recorder.client_idx_list.append(client_idx_list.copy())
        valid_node_flag = self.N_samples_for_curr_node>0
        return valid_node_flag
    
    def isLeaf(self, best_feature, node):
        if node == node.parent.left:
            N_samples = self.sampleForEachFeature[2, best_feature]
            N0_samples = self.sampleForEachFeature[3, best_feature]
        else:
            N_samples = self.sampleForEachFeature[0, best_feature] - self.sampleForEachFeature[2, best_feature]
            N0_samples = self.sampleForEachFeature[1, best_feature] - self.sampleForEachFeature[3, best_feature]
        flag = N_samples==N0_samples or N0_samples==0 or self.depth==self.MAX_DEPTH-1
        if flag: node.label = 0 if N0_samples>=N_samples-N0_samples else 1
        return flag

    def findBestFeature(self):
        boundsForEachFeature = self.boundsForEachFeature[:, self.feature_flag_list]
        feature_idx_list = np.where(self.feature_flag_list)[0]
        max_low_idx = np.argmax(boundsForEachFeature[1,:])
        max_low_value = boundsForEachFeature[1,max_low_idx]
        max_up_value = np.max(np.concatenate([boundsForEachFeature[0,:max_low_idx], boundsForEachFeature[0,max_low_idx+1:]]))
        idx = feature_idx_list[max_low_idx] if max_low_value>=max_up_value else -1
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
    
    def calLowerUpperBounds(self):
        # For the first-round communication, no computation would be performed,
        # as there is no sufficient information to find the best splitting feature.
        if self.curr_feature_no<=1:
            print(f"Samples for current node: {self.N_samples_for_curr_node}")
            print("Please update information for new features")
            return
        
        # The lower and upper bounds for none-client-selected features are same.
        # As such, these values will be computed for only once.
        feature_selected_flag = np.sum(self.clientSetForEachFeature, axis=1)>0
        unselect_feature_cal_flag = False
        bounds_for_unselected_feature = 0.
        # The - sign in Gini Index is omitted. Hence, the criteria to find the best feature is to 
        # find the one with the largest lower bound
        max_lower_bound = np.max(self.boundsForEachFeature[1,:])

        for feature_idx in np.where(self.feature_flag_list)[0]:
            logging.info(f"Computing bounds for feature {feature_idx} at node {self.split_node_no}")
            # if the upper bound for this feature is lower than the max low bound,
            # it is not the best feature.
            if self.boundsForEachFeature[0, feature_idx]<=max_lower_bound: continue
            if feature_selected_flag[feature_idx] or not unselect_feature_cal_flag:
                n_client_not_def = np.sum(~self.clientSetForEachFeature[feature_idx]) # |u_k|
                
                N_def, N0_def, Nl_def, Nl0_def = self.sampleForEachFeature[:, feature_idx]
                Nr_def, Nr0_def = N_def-Nl_def, N0_def-Nl0_def
                if n_client_not_def==0:
                    bounds = calBoundFrom4Tuples((N_def, N0_def, Nl_def, Nl0_def))
                    self.boundsForEachFeature[1, feature_idx] = bounds
                    self.boundsForEachFeature[0, feature_idx] = bounds
                    continue
                idx_client_not_def = np.where(~self.clientSetForEachFeature[feature_idx])[0]
                xl_not_def = cvx.Variable(n_client_not_def, name="Nl0", integer=True)
                xr_not_def = cvx.Variable(n_client_not_def, name="Nr0", integer=True)
                yl_not_def = cvx.Variable(n_client_not_def, name="Nl",  integer=True)
                yr_not_def = cvx.Variable(n_client_not_def, name="Nr",  integer=True)
                constraints = [
                    xl_not_def>=0,
                    xr_not_def>=0,
                    xl_not_def<=yl_not_def,
                    xr_not_def<=yr_not_def,
                    xl_not_def+xr_not_def==self.sampleForEachClient[1,idx_client_not_def],
                    yl_not_def+yr_not_def==self.sampleForEachClient[0,idx_client_not_def],
                ]
                not_def_idx = 0
                for client_idx, flag in enumerate(self.clientSetForEachFeature[feature_idx]):
                    if not flag:
                        constraints.append(
                            cvx.quad_over_lin(xl_not_def[not_def_idx],yl_not_def[not_def_idx]+np_eps) \
                                +cvx.quad_over_lin(xr_not_def[not_def_idx],yr_not_def[not_def_idx]+np_eps) \
                                <=self.boundsForEachClient[client_idx]
                        )
                        not_def_idx += 1
#                 N_def, N0_def, Nl_def, Nl0_def = self.sampleForEachFeature[:, feature_idx]
#                 Nr_def, Nr0_def = N_def-Nl_def, N0_def-Nl0_def

                obj = cvx.Minimize(cvx.quad_over_lin(cvx.sum(xl_not_def)+Nl0_def, cvx.sum(yl_not_def)+Nl_def+np_eps) \
                                   +cvx.quad_over_lin(cvx.sum(xr_not_def)+Nr0_def, cvx.sum(yr_not_def)+Nr_def+np_eps))
                prob = cvx.Problem(obj, constraints)
                prob.solve(reoptimize=True, solver=cvx.GUROBI)
                self.boundsForEachFeature[1, feature_idx] = prob.value
                
                obj = cvx.Minimize(cvx.sum(yl_not_def))
                prob_intvl_low = cvx.Problem(obj, constraints)
                prob_intvl_low.solve(reoptimize=True, solver=cvx.GUROBI)
                self.searchIntervalForEachFeature[0, feature_idx] = prob_intvl_low.value
                
                obj = cvx.Maximize(cvx.sum(yl_not_def))
                prob_intvl_up = cvx.Problem(obj, constraints)
                prob_intvl_up.solve(reoptimize=True, solver=cvx.GUROBI)
                self.searchIntervalForEachFeature[1, feature_idx] = prob_intvl_up.value
                constraints.append(cvx.sum(yl_not_def)==0)
                search_value_list = []
                for samples_in_left_node in range(int(prob_intvl_low.value), int(prob_intvl_up.value+1)):
#                     Nl_total, Nl0_total = Nl_def, Nl0_def
                    
                    _ = constraints.pop()
                    constraints.append(cvx.sum(yl_not_def)==samples_in_left_node)
                    obj = cvx.Minimize(cvx.sum(xl_not_def))
                    prob_quad_low = cvx.Problem(obj, constraints)
                    try:
                        prob_quad_low.solve(reoptimize=True, solver=cvx.GUROBI)
                        low_value = prob_quad_low.value
                        t = (self.N_samples_for_curr_node, self.N0_samples_for_curr_node, Nl_def+samples_in_left_node, Nl0_def+low_value)
                        low_value = calBoundFrom4Tuples(t)
                    except:
                        low_value = 0.
                    
                    obj = cvx.Maximize(cvx.sum(xl_not_def))
                    prob_quad_up = cvx.Problem(obj, constraints)
                    try:
                        prob_quad_up.solve(reoptimize=True, solver=cvx.GUROBI)
                        up_value = prob_quad_up.value
                        t = (self.N_samples_for_curr_node, self.N0_samples_for_curr_node, Nl_def+samples_in_left_node, Nl0_def+up_value)
                        up_value = calBoundFrom4Tuples(t)
                    except:
                        up_value = 0.
                    search_value_list.append(max(low_value, up_value))
                self.boundsForEachFeature[0, feature_idx] = np.max(search_value_list)
                    
                if not feature_selected_flag[feature_idx]:
                    unselect_feature_cal_flag = True
                    low_bound_for_unselected_feature = prob.value
                    up_bound_for_unselected_feature = np.max(search_value_list)
                    search_intvl_low_bound = prob_intvl_low.value
                    search_intvl_up_bound = prob_intvl_up.value
                    
            else:
                self.boundsForEachFeature[1, feature_idx] = low_bound_for_unselected_feature
                self.boundsForEachFeature[0, feature_idx] = up_bound_for_unselected_feature
                self.searchIntervalForEachFeature[0, feature_idx] = search_intvl_low_bound
                self.searchIntervalForEachFeature[1, feature_idx] = search_intvl_up_bound
                
        self.recorder.bounds_for_each_feature[-1].append(self.boundsForEachFeature.copy())
        
        
class Client:
    def __init__(self, x):
        self.x = x
        self.N_features = self.x.shape[1]-1
        root = TreeNode(data_idx=np.arange(len(x)))
        self.curr_node = root
        self.curr_node_idx = 0
        self.submit_feature_no_list = []
        self.tree = [root]
        self._resetStatus()
        
    def __len__(self):
        return len(self.x)
    
    def _resetStatus(self):
        self.curr_submit_idx = 0
        self.curr_report_indx = 0
        self.feature_flag_list = np.ones([1, self.N_features]).astype(np.bool_).squeeze()
    
    def submitFeatureInfo(self, feature_idx=None):
        if feature_idx is None:
            if self.curr_submit_idx==0:
                self._calGiniListForCurrNode()
                ancestor_feature_list = findAncestorFeatures(self.curr_node)
                self.feature_flag_list[ancestor_feature_list] = False
                self.gini_sort_indx = np.argsort(self.gini_list[self.feature_flag_list])
                self.feature_idx_list = np.where(self.feature_flag_list)[0]
            feature_idx = self.feature_idx_list[self.gini_sort_indx[self.curr_submit_idx]]
        self.curr_submit_idx += 1
        return feature_idx, self.split_info_list[feature_idx]

#     def submitFeatureInfo(self, feature_idx=None):
#         if feature_idx is None:
#             if self.curr_submit_idx==0:
#                 self._calGiniListForCurrNode()
#                 ancestor_feature_list = findAncestorFeatures(self.curr_node)
#                 self.feature_flag_list[ancestor_feature_list] = False
#             while self.curr_report_indx < self.N_features:
#                 feature_idx = self.gini_sort_indx[self.curr_report_indx]
#                 self.curr_report_indx += 1
#                 if self.feature_flag_list[feature_idx]:
#                     break
#         self.curr_submit_idx += 1
#         return feature_idx, self.split_info_list[feature_idx] 
    
    def _calGini(self, y):
        if len(y)==0: return 0
        p = np.sum(y)/len(y)
        return 2*p*(1-p)
    
    def _calGiniForFeature(self, data, feature_idx):
        
        N_total = len(data)
        if N_total==0:return 0, (0, 0)
        
        left_data = data[data[:,feature_idx]==0]
        left_gini = self._calGini(left_data[:,-1])
        N_left = len(left_data)
        N_left_zero = np.sum(left_data[:,-1]==0)
        
        right_data = data[data[:,feature_idx]==1]
        right_gini = self._calGini(right_data[:,-1])
        N_right = len(right_data)
        gini = N_left/N_total*left_gini+N_right/N_total*right_gini
        
        return gini, (N_left, N_left_zero)
    
    def _calGiniListForCurrNode(self):
        gini_list = []
        split_info_list = []
        curr_node = self.curr_node
        x = self.x[curr_node.data_idx]
        N_total = len(x)
        N_zero = np.sum(x[:,-1]==0)
        for i in range(self.N_features):
            info = self._calGiniForFeature(x, feature_idx=i)
            gini_list.append(info[0])
            split_info_list.append((N_total, N_zero)+info[1])
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
        curr_node.left = left_node
        curr_node.right = right_node
        if left_label is None:
            self.tree.append(left_node)
        if right_label is None:
            self.tree.append(right_node)
        self.curr_node_idx += 1
        if self.curr_node_idx<len(self.tree):
            self.curr_node = self.tree[self.curr_node_idx]
        self.submit_feature_no_list.append(self.curr_submit_idx)
        self._resetStatus()
    
    def checkLeafforCurrNode(self):
        node_info = self.split_info_list[0]
        return node_info[0]==node_info[1] or node_info[1]==0
               
    def submitNodeInfo(self):
        info = []
        for idx in range(self.curr_node_indx, len(self.tree)):
            node = self.tree[idx]
            x_indx = node.indx
            label = self.x[x_indx][:,-1]
            info.append([np.sum(label==0), np.sum(label==1)])
        return np.vstack(info)
        
    def counts(self):
        return np.unique(self.x, return_counts=True)
    
def layerOrderTraversal(root):
    layer_node_cnt = [1]
    curr_layer = [root]
    next_layer = []
    while len(curr_layer)>0:
        for node in curr_layer:
            if node.left is not None:
                next_layer.append(node.left)
            if node.right is not None:
                next_layer.append(node.right)
        curr_layer = next_layer
        layer_node_cnt.append(len(curr_layer))
        next_layer = []
    return layer_node_cnt