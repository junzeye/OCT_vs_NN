from pyomo.environ import *
import numpy as np

# REMEMBER TO DISTINGUISH L_T FROM l_T!!
class OCT_classifier:
    '''
    Optimal classification tree, without hyperplanes.
    '''
    def __init__(self, max_depth=3, min_samples_split=2, alpha=0.1, warmstart=True, timelimit=600, output=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.alpha = alpha
        self.warmstart = warmstart
        self.timelimit = timelimit
        self.output = output
        self.model = None


    def _model_add_sets(self, dataset_length, dim_x, class_labels: list):
        '''
        Adds sets to `self.model`; helper method to be invoked within `self.fit()`.
        
        dataset_length: number of data points in the dataset
        num_x: dimension of predictor variables
        '''
        m = self.model
        m.T = RangeSet(2 ** self.max_depth - 1)
        m.T_B = RangeSet(2 ** (self.max_depth - 1) - 1)
        m.T_L = RangeSet(2 ** (self.max_depth - 1), 2 ** self.max_depth - 1)
        m.I = RangeSet(0, dataset_length - 1) # index set of each data point
        m.P = RangeSet(dim_x) # index set of the dimensions of the predictor variable
        m.K = Set(initialize = class_labels) # class labels need not be numeric
    
    def _model_add_decision_var(self):
        '''
        Adds decision variables to `self.model`; helper -method to be invoked within `self.fit()`.
        '''
        m = self.model
        m.z_IT = Var(m.I, m.T, domain = Binary)
        m.d_T = Var(m.T_B, domain = Binary)
        m.l_T = Var(m.T_L, domain = Binary) # indicator for if leaf node t is empty
        m.L_T = Var(m.T_L, domain = NonNegativeReals) # optimal misclassification cost in leaf node t
        m.a_JT = Var(m.P, m.T_B, domain = Binary)
        m.b_T = Var(m.T_B, domain = NonNegativeReals, bounds = (0,1))
        m.N_KT = Var(m.K, m.T_L, domain = NonNegativeIntegers) # number of class-k points in node t
        m.N_T = Var(m.T_L, domain = NonNegativeIntegers) # number of points assigned to node t
        m.c_KT = Var(m.K, m.T_L, domain = Binary)

    def _model_add_param(self, L_hat, y):
        m = self.model
        m.L_hat = Param(initialize = L_hat)
        m.N_min = Param(initialize = self.min_samples_split)

        def init_Y_IK(model, i, k):
            if y[i] == k:
                return 1
            else:
                return -1
        m.Y_IK = Param(m.I, m.K, initialize = init_Y_IK)

    def _model_add_objective(self):  
        '''
        Adds objective function to `self.model`; helper method to be invoked within `self.fit()`.
        
        L_hat: scaling factor
        '''
        m = self.model
        m.obj = Objective(expr = sum(m.L_T[t] / m.L_hat for t in m.T_L) +
            self.alpha * sum(m.d_T[t] for t in m.T_B ))

    # ¼ÇµÃ¼ì²é
    def _model_add_constr(self, x, Epsilon):
        '''
        Adds constraints to `self.model`; helper method to be invoked within `self.fit()`.

        Epsilon: numpy array / python list
        '''
        m = self.model
        n = len(m.I) # number of data points in the dataset

        # (20), (21), (22)
        m.misclass_cost = ConstraintList()
        for t in m.T_L:
            for k in m.K:
                m.misclass_cost.add( m.L_T[t] >= m.N_T[t] - m.N_KT[k,t]
                        - n * (1 - m.c_KT[k, t]) )  # (20)
                m.misclass_cost.add( m.L_T[t] <= m.N_T[t] - m.N_KT[k,t]
                        + n * m.c_KT[k,t] )  # (21)
        # (22) is already satisfied by restricting the domain of L_T to NonNegativeReals

        # (15), (16), (18)
        m.point_leaf_match = ConstraintList()
        for t in m.T_L:
            for k in m.K:
                m.point_leaf_match.add( m.N_KT[k,t] == 0.5 * 
                        sum( (1 + m.Y_IK[i,k]) * m.z_IT[i,t] for i in m.I) )  # (15)
            m.point_leaf_match.add( m.N_T[t] == sum(m.z_IT[i,t] for i in m.I) )  # (16)
            m.point_leaf_match.add( sum(m.c_KT[k,t] for k in m.K) == m.l_T[t] )  # (18)

        # (13), (14)
        m.splits = ConstraintList()
        for i in m.I:
            for t in m.T_L:
                curr = t
                parent = curr // 2
                while parent > 0:
                    if parent * 2 == curr: # (13)
                        m.splits.add( sum(m.a_JT[p, parent] * (x[i][p] + Epsilon[p - 1])
                            for p in m.P) <= m.b_T[parent] + (1 + max(Epsilon)) * (1- m.z_IT[i, t]) )
                        # Epsilon[p-1] is indexed (p-1) i/o p to accommodate pyomo list indexing
                    else: # (14)
                        m.splits.add( sum(m.a_JT[p, parent] * x[i][p] for p in m.P) >= m.b_T[parent]
                            - (1- m.z_IT[i, t]) )

        # (8)
        m.one_leaf = ConstraintList()
        for i in m.I:
            m.one_leaf.add( sum(m.z_IT[i, t] for t in m.T_L) == 1 )

        # (6), (7)
        m.leaf_min_samples = ConstraintList()
        for t in m.T_L:
            for i in m.I:
                m.leaf_min_samples.add( m.z_IT[i,t] <= m.l_T[t] )  # (6)
            m.leaf_min_samples.add( sum(m.z_IT[i,t] for i in m.I) >= m.N_min * m.l_T[t] )  # (7)
        
        # (2), (3)
        m.splits_activation = ConstraintList()
        for t in m.T_B:
            m.splits_activation.add( sum(m.a_JT[p, t] for p in m.P) == m.d_T[t] )  # (2)
            m.splits_activation.add( m.b_T[t] <= m.d_T[t] )  # (3)
        
        # (5)
        m.dt_cascade = ConstraintList()
        for t in m.T_B:
            if t == 1:
                continue
            m.dt_cascade.add(m.d_T[t] <= m.d_T[t // 2])


    # Calculate baseline accuracy: accuracy of simply predicting the most popular class of the dataset
    def _get_baseline(self, y):
        return np.bincount(y).max() / len(y)

    # TO BE COMPLETED
    def _get_Epsilon(self, x):
        m = self.model
        Epsilon = [0 for j in m.P]

        for j in m.P:
            x_j_i = []
            for i in m.I:
                x_j_i.append(x[i][j])
            x_j_i.sort()  # sort the values of the j-th feature
        
            diff = []
            for i in m.I:
                if i == len(m.I)-1:
                    continue
                if x_j_i[j+1] - x_j_i[j] != 0:
                    diff.append(x_j_i[j+1] - x_j_i[j])

            Epsilon[j] = min(diff)

        return Epsilon

    def fit(self, x, y):
        self.model = ConcreteModel()

        # Get data size
        dataset_length, dim_x = x.shape
        # Get unique class labels
        class_labels = list( np.unique(y) )

        if self.output:
            print('Training data include {} instances, {} features.'.format(dataset_length, dim_x))

        L_hat = self._get_baseline(y)
        Epsilon = self._get_Epsilon(x)

        self._model_add_sets(dataset_length, dim_x, class_labels)
        self._model_add_decision_var()
        self._model_add_objective()
        self._model_add_param(L_hat, y)
        self._model_add_constr(x, Epsilon)
