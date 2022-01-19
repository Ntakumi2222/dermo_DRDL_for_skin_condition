import plotly.express as px
import plotly.offline as offline
import pandas as pd
from ipywidgets import HTML, Image, Layout, HBox, VBox, interactive
from src.utils.Utils import *


class PlotlyLabeler3D(object):
    def __init__(self, data, embedding, labels, paths, data_type, dr_type):
        self.data = data
        self.embedding = embedding
        self.labels = labels
        self.paths = paths
        self.data_type = data_type
        self.dr_type = dr_type
        self.size = 1
        temp_embeding_label_list = [
            [self.labels[i], self.embedding[i][0], self.embedding[i][1], self.embedding[i][2],
             self.size, self.paths[i], self.data, i]
            for i in range(len(self.labels))]
        self.label_df = pd.DataFrame(temp_embeding_label_list,
                                     columns=['label', 'x', 'y', 'z', 'size', 'path', 'data', 'index'])
        self.details = HTML()
        self.image_widget = Image(
            value=compress_to_bytes(self.data[0], 'png'),
            layout=Layout(height='600px', width='400px')
        )
        self.opacity_slider = interactive(self.set_opacity,
                                          opacity=(0.0, 1.0, 0.01),
                                          size=(1, 10, 0.25))

    def set_opacity(self, opacity, size):
        self.fig.marker.opacity = opacity
        self.fig.marker.size = size

    def start(self):
        self.fig = px.scatter_3d(self.label_df, x='x', y='y', z='z',
                                 color='label', opacity=0.7, size='size', size_max=5, hover_name="index",
                                 hover_data=["label", "path"])

        offline.plot(self.fig, filename=f'../result/html/{self.data_type}_{self.dr_type}_{get_TIME_STAMP()}.html',
                     auto_open=True)

    def hover_fn(self, points, state):
        ind = points.point_inds[0]

        self.fig.add_layout_image(
            dict(
                source=self.label_df.data[ind],
                xref="x",
                yref="y",
                x=0,
                y=16,
                sizex=4,
                sizey=15,
                sizing="stretch")
        )
        self.fig.update()
        self.details.value = self.label_df.label[ind].to_frame().to_html()
        self.image_widget.value = compress_to_bytes(self.label_df.data[ind], 'png')
