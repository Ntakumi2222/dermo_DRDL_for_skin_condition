import copy
import numpy as np
import matplotlib.pyplot as plt


class DLPreProcess:
    """
    ここでのx_train, y_trainは全て転置したものを使用する。
    """

    def __init__(self, x_train, y_train, tau=5, lmd=0.1, mu=0.0001, tol=0.1, k=100, count=100) -> None:
        # Kは入力要素数以下でないければならない
        self._K = k
        self._x_train = x_train.T
        self._y_train = y_train.T
        self._train_num = self._x_train.shape[1]
        _rand_list = np.random.choice(
            range(self._train_num), self._K, replace=False)
        self._d_high = self._x_train[:, _rand_list]+1
        self._C = np.zeros((self._K, self._train_num))+1
        self._D1 = self._d_high - 1
        self._C1 = self._C - 1
        self._tau = tau
        self._lmd = lmd
        self._mu = mu
        self._tol = tol
        self._count = count
        self._d_loss = []
        self._c_loss = []
        self.process()
        
    def process(self):
        print('Process preprocessing dl')
        _count = 0
        while np.max(np.abs(self._D1-self._d_high)) > self._tol or np.max(np.abs(self._C1-self._C)) > self._tol:
            self._d_high = copy.deepcopy(self._D1)
            self._C = copy.deepcopy(self._C1)
            self._C1 = np.zeros((self._K, self._train_num))
            dist_x_train = np.tile(
                np.sum(np.power(self._x_train, 2), axis=0).T, (self._K, 1)).T
            #  Since numpy does not have the notion of (1,n), the inversion is happening in D
            dist_d_low = np.tile(
                np.sum(np.power(self._d_high, 2), axis=0), (self._train_num, 1))
            dist = dist_x_train + dist_d_low - \
                2 * np.dot(self._x_train.T, self._d_high)
            neighbor = np.argsort(dist, 1) 
            for i in range(self._train_num):
                opt_indices = neighbor[i, :self._tau]  # Get up to the tau-th
                temp_d = self._d_high[:, opt_indices]
                g = temp_d - np.tile(self._x_train[:, i].T, (self._tau, 1)).T
                g = np.dot(g.T, g)
                temp_c = np.linalg.solve(
                    (g + np.diag(self._lmd * np.diag(g)) +
                     self._mu * np.eye(self._tau)),
                    np.ones((self._tau, 1)))
                # One-dimensional transposition in python does not work with .T
                # Full transposition of linear equations.
                temp_c = temp_c / np.sum(temp_c)
                self._C1[opt_indices, i] = temp_c[:, 0]
            self._P1 = np.dot(self._x_train, self._C1.T)
            self._P2 = np.dot(self._x_train, np.power(self._C1, 2).T)
            self._P3 = np.dot(self._C1, self._C1.T)
            _diagele = np.diag(self._P3)
            self._P3 = self._P3-np.diag(_diagele)
            _usecol = np.any(self._C1, axis=1)
            P_usecoled = (self._P1[:, _usecol]-np.dot(self._d_high, self._P3[:, _usecol]) +
                          self._lmd*self._P2[:, _usecol])
            diagele_usecoled = np.diag(1/_diagele[_usecol])
            self._D1[:, _usecol] = np.dot(
                P_usecoled, diagele_usecoled)/(1+self._lmd)
            _count = _count+1
            print(f'count:{_count}')
            print(f"D_diff{np.max(np.abs(self._D1-self._d_high))}")
            print(f"C_diff{np.max(np.abs(self._C1-self._C))}")
            self._d_loss.append(np.max(np.abs(self._D1-self._d_high)))
            self._c_loss.append(np.max(np.abs(self._C1-self._C)))
            if _count >= self._count:
                break
        # fix the coefficient, learn the low dimensional diction %%%%%
        #  The square matrix of code C is often singular, so it is pseudo-inverse.
        self._d_high = self._D1
        self._d_low = np.dot(np.dot(self._y_train, self._C.T),
                            np.linalg.pinv((np.dot(self._C, self._C.T))))

    def get_d_low_d(self):
        return self._d_low, self._d_high
    
    def get_d_loss_c_loss(self):
        return self._d_loss, self._c_loss