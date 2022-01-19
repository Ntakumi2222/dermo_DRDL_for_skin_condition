import numpy as np


class DLReconstruction():
    def __init__(self, d, d_low, x_tests, tau, lmd, mu, K):
        self.K = K
        self.d = d
        self.d_low = d_low
        self.y_tests = x_tests
        self.c_lows = []
        self.test_num = x_tests.shape[1]
        self.tau = tau
        self.lmd = lmd
        self.mu = mu
        self.process()

    def process(self):
        # d_lowはテスト数*d_low。後々転置してるから行列はcと整合性がある、そのまま
        # c_lowはテスト数*入力画像枚数、係数は入力画像をどれぐらいの分配で表現するかなので。
        self.c_lows = np.zeros((self.K, self.y_tests.shape[1]))

        dist_x_test = np.tile(
            np.sum(np.power(self.y_tests, 2), axis=0).T, (self.K, 1)).T
        # numpyには(1,nという概念がないため、Dの方で逆転が起こっている)
        dist_d_low = np.tile(
            np.sum(np.power(self.d_low, 2), axis=0), (self.test_num, 1))
        dist = dist_x_test + dist_d_low - \
            2 * np.dot(self.y_tests.T, self.d_low)
        neighbor = np.argsort(dist, 1)  # 最適解をsortで暫定的に求めている
        for i in range(self.test_num):
            opt_indices = neighbor[i, :self.tau]  # tau番目までを取得
            temp_d = self.d_low[:, opt_indices]
            g = temp_d - np.tile(self.y_tests[:, i].T, (self.tau, 1)).T
            g = np.dot(g.T, g)
            temp_c = np.linalg.solve(
                (g + np.diag(self.lmd * np.diag(g)) +
                    self.mu * np.eye(self.tau)),
                np.ones((self.tau, 1)))
            # pythonでは一次元の転置が.Tではかからない
            # 線型方程式の全転置をしている
            temp_c = temp_c / np.sum(temp_c)
            self.c_lows[opt_indices, i] = temp_c[:, 0]
        self.recons = np.dot(self.d, self.c_lows)
        pass

    def get_recons(self):
        return self.recons
