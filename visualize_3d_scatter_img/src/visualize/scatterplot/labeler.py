import cv2
import pylab as plt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.widgets import TextBox
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D


class Labeler3D(object):
    def __init__(self, data, embedding, labels):
        warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
        self.fig = plt.figure(figsize=(200, 100), dpi=60)
        self.data = data
        self.embedding = embedding
        self.labels = labels
        self.selected_labels = []
        self.ind = 0
        self.norm_text = None

    def start(self):
        plt.rcParams['font.family'] = 'Hiragino Sans'
        cmap = plt.cm.rainbow
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                                 width_ratios=[3, 1])

        # top
        ax1 = self.fig.add_subplot(spec[0], projection=Axes3D.name)

        for idx, label in enumerate(np.unique(self.labels)):
            indices = np.where(self.labels == label)[0]
            c = cmap(idx / (len(np.unique(self.labels)) - 1)) if len(
                np.unique(self.labels)) > 1 else cmap(
                idx / (len(np.unique(self.labels))))
            ax1.plot(self.embedding[indices, 0], self.embedding[indices, 1],
                     self.embedding[indices, 2], 'o', color=c,
                     label=label, alpha=0.7)
            [ax1.text(self.embedding[ind, 0], self.embedding[ind, 1],
                      self.embedding[ind, 2], str(ind), size=14,
                      color=c) for ind in indices]
        ax1.legend(loc="upper left")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        # bottom
        self.ax2 = self.fig.add_subplot(spec[1])
        self.fig.subplots_adjust(bottom=0.2)
        self.ax2.tick_params(labelbottom=False,
                             labelleft=False,
                             labelright=False,
                             labeltop=False)
        axbox1 = self.fig.add_axes([0.1, 0.1, 0.8, 0.05])
        axbox2 = self.fig.add_axes([0.1, 0.05, 0.8, 0.05])
        text_box = TextBox(axbox1, "Label")
        text_box_compare = TextBox(axbox2, "Compare")
        text_box.on_submit(self.submit)
        text_box_compare.on_submit(self.submit_compare)
        plt.show()

    def submit(self, expression):
        self.ax2.cla()
        label = eval(expression)
        self.ind = int(label)
        self.ax2.tick_params(labelbottom=False,
                             labelleft=False,
                             labelright=False,
                             labeltop=False)
        image = np.clip(self.data[self.ind], a_min=0, a_max=255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.ax2.imshow(image)
        self.ax2.text(0, 0.3, str(self.embedding[self.ind]), size=12,
                      horizontalalignment="left")
        self.selected_labels.append(self.ind)

    def submit_compare(self, expression):
        compare_index = int(eval(expression))
        if self.norm_text == None:
            self.norm_text = self.ax2.text(0, 20, str(
                np.linalg.norm(
                    self.embedding[self.ind] - self.embedding[compare_index])),
                                           size=12,
                                           horizontalalignment="left")
        else:
            self.norm_text.remove()
            self.norm_text = self.ax2.text(0, 20,
                                           str(np.linalg.norm(
                                               self.embedding[self.ind] -
                                               self.embedding[compare_index])),
                                           size=12, horizontalalignment="left")
        self.selected_labels.append(self.ind)


class Labeler2D(object):
    def __init__(self, data, embedding, labels):
        warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
        self.fig = plt.figure(figsize=(200, 100), dpi=60)
        self.data = data
        self.embedding = embedding
        self.labels = labels
        self.selected_labels = []
        self.ind = 0
        self.norm_text = None

    def start(self):
        plt.rcParams['font.family'] = 'Hiragino Sans'
        cmap = plt.cm.jet
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                                 width_ratios=[3, 1])

        # top
        ax1 = self.fig.add_subplot(spec[0])

        for idx, label in enumerate(np.unique(self.labels)):
            indices = np.where(self.labels == label)[0]
            c = cmap(idx / (len(np.unique(self.labels)) - 1)) if len(
                np.unique(self.labels)) > 1 else cmap(
                idx / (len(np.unique(self.labels))))
            ax1.plot(self.embedding[indices, 0], self.embedding[indices, 1],
                     'o', color=c,
                     label=label, alpha=0.7)
            [ax1.text(self.embedding[ind, 0], self.embedding[ind, 1], str(ind),
                      size=14,
                      color=c) for ind in indices]
        ax1.legend(loc="upper left")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # bottom
        self.ax2 = self.fig.add_subplot(spec[1])
        self.fig.subplots_adjust(bottom=0.2)
        self.ax2.tick_params(labelbottom=False,
                             labelleft=False,
                             labelright=False,
                             labeltop=False)
        axbox1 = self.fig.add_axes([0.1, 0.1, 0.8, 0.05])
        axbox2 = self.fig.add_axes([0.1, 0.05, 0.8, 0.05])
        text_box = TextBox(axbox1, "Label")
        text_box_compare = TextBox(axbox2, "Compare")
        text_box.on_submit(self.submit)
        text_box_compare.on_submit(self.submit_compare)
        plt.show()

    def submit(self, expression):
        self.ax2.cla()
        label = eval(expression)
        self.ind = int(label)
        self.ax2.tick_params(labelbottom=False,
                             labelleft=False,
                             labelright=False,
                             labeltop=False)
        image = np.clip(self.data[self.ind], a_min=0, a_max=255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.ax2.imshow(image)
        self.ax2.text(0, 0.3, str(self.embedding[self.ind]), size=12,
                      horizontalalignment="left")
        self.selected_labels.append(self.ind)

    def submit_compare(self, expression):
        compare_index = int(eval(expression))
        if self.norm_text == None:
            self.norm_text = self.ax2.text(0, 20, str(
                np.linalg.norm(
                    self.embedding[self.ind] - self.embedding[compare_index])),
                                           size=12,
                                           horizontalalignment="left")
        else:
            self.norm_text.remove()
            self.norm_text = self.ax2.text(0, 20,
                                           str(np.linalg.norm(
                                               self.embedding[self.ind] -
                                               self.embedding[compare_index])),
                                           size=12, horizontalalignment="left")
        self.selected_labels.append(self.ind)
