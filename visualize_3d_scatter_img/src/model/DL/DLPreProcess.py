import copy
import numpy as np
import matplotlib.pyplot as plt
import os
from src.settings import get_LOSS_PLOT_OUTPUT_DIR, get_DATA_NAME, get_TIME_STAMP, get_DR_TYPE, get_IS_USE_PREPROCESSED_DICTIONARY, get_PREPROCESSED_DICTIONARY_PATH


class DLPreProcess:
    """
    ここでのx_train, y_trainは全て転置したものを使用する。
    """

    def __init__(self, x_train, y_train, tau=5, lmd=0.1, mu=0.0001, tol=0.1, k=100, count=100) -> None:
        # Kは入力要素数以下でないければならない
        self.K = k
        self.x_train = x_train.T
        self.y_train = y_train.T
        self.train_num = self.x_train.shape[1]
        rand_list = np.random.choice(
            range(self.train_num), self.K, replace=False)
        self.d_high = self.x_train[:, rand_list]+1
        self.C = np.zeros((self.K, self.train_num))+1
        self.D1 = self.d_high - 1
        self.C1 = self.C - 1
        self.tau = tau
        self.lmd = lmd
        self.mu = mu
        self.tol = tol
        self.count = count
        self.d_loss = []
        self.c_loss = []
        if get_IS_USE_PREPROCESSED_DICTIONARY():
            self.load_d_low_d()
        else:
            self.process()
            self.save_d_c_loss_fig()

    def process(self):
        print('Process preprocessing dl')
        count = 0
        while np.max(np.abs(self.D1-self.d_high)) > self.tol or np.max(np.abs(self.C1-self.C)) > self.tol:
            # ここで更新しているDとCの値が極端に大きい場合は更新させないのはどうだろうか
            self.d_high = copy.deepcopy(self.D1)
            self.C = copy.deepcopy(self.C1)
            self.C1 = np.zeros((self.K, self.train_num))
            dist_x_train = np.tile(
                np.sum(np.power(self.x_train, 2), axis=0).T, (self.K, 1)).T
            # numpyには(1,nという概念がないため、Dの方で逆転が起こっている)
            dist_d_low = np.tile(
                np.sum(np.power(self.d_high, 2), axis=0), (self.train_num, 1))
            dist = dist_x_train + dist_d_low - \
                2 * np.dot(self.x_train.T, self.d_high)
            neighbor = np.argsort(dist, 1)  # 最適解をsortで暫定的に求めている
            for i in range(self.train_num):
                opt_indices = neighbor[i, :self.tau]  # tau番目までを取得
                temp_d = self.d_high[:, opt_indices]
                g = temp_d - np.tile(self.x_train[:, i].T, (self.tau, 1)).T
                g = np.dot(g.T, g)
                temp_c = np.linalg.solve(
                    (g + np.diag(self.lmd * np.diag(g)) +
                     self.mu * np.eye(self.tau)),
                    np.ones((self.tau, 1)))
                # pythonでは一次元の転置が.Tではかからない
                # 線型方程式の全転置をしている
                temp_c = temp_c / np.sum(temp_c)
                self.C1[opt_indices, i] = temp_c[:, 0]
            self.P1 = np.dot(self.x_train, self.C1.T)
            self.P2 = np.dot(self.x_train, np.power(self.C1, 2).T)
            self.P3 = np.dot(self.C1, self.C1.T)
            diagele = np.diag(self.P3)
            self.P3 = self.P3-np.diag(diagele)
            usecol = np.any(self.C1, axis=1)
            P_usecoled = (self.P1[:, usecol]-np.dot(self.d_high, self.P3[:, usecol]) +
                          self.lmd*self.P2[:, usecol])
            diagele_usecoled = np.diag(1/diagele[usecol])
            self.D1[:, usecol] = np.dot(
                P_usecoled, diagele_usecoled)/(1+self.lmd)
            count = count+1
            print(f'count:{count}')
            print(f"D_diff{np.max(np.abs(self.D1-self.d_high))}")
            print(f"C_diff{np.max(np.abs(self.C1-self.C))}")
            self.d_loss.append(np.max(np.abs(self.D1-self.d_high)))
            self.c_loss.append(np.max(np.abs(self.C1-self.C)))
            if count >= self.count:
                break
        # fix the coefficient, learn the low dimensional diction %%%%%
        # code Cの正方行列が特異行列になってしまっていることが多多あるため疑似逆行列にしている
        self.d_high = self.D1
        self.d_low = np.dot(np.dot(self.y_train, self.C.T),
                            np.linalg.pinv((np.dot(self.C, self.C.T))))

    def save_d_c_loss_fig(self):
        plt.plot(range(len(self.d_loss)), self.d_loss,
                 marker="o", color="red", linestyle="--", label='D loss')
        plt.plot(range(len(self.c_loss)), self.c_loss,
                 marker="v", color="blue", linestyle=":", label='C loss')
        plt.title('d_loss and c_loss per epoch')
        plt.xlabel("epoch")
        plt.ylabel("loss(log)")
        plt.yscale('log')
        plt.legend()
        os.makedirs(get_LOSS_PLOT_OUTPUT_DIR(), exist_ok=True)
        plt.savefig(os.path.join(get_LOSS_PLOT_OUTPUT_DIR(),
                    f'{get_DATA_NAME()}_{get_DR_TYPE().name}_{get_TIME_STAMP()}.png'), format="png", dpi=300)
        plt.cla()

    # TODO:絶対に良くない。ここで保存しても次元削減の結果は毎回変わるので無意味
    def load_d_low_d(self):
        print('loading npz')
        temp_npz_data = np.load(get_PREPROCESSED_DICTIONARY_PATH())
        self.d_low = temp_npz_data['d_low']
        self.d_high = temp_npz_data['d_high']
        temp_npz_data.close()

    def get_d_low_d(self):
        return self.d_low, self.d_high
