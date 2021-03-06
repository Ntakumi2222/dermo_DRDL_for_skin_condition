import numpy as np
import os
import pandas as pd
import scipy.stats as stats


class QuantitativeEvaluation:
    def __init__(self, mapped_points, image_labels, image_paths, new_image_labels, additional_embeddings, new_reprocess_image_labels, USER_PREF, neighbors=10):
        self.mapped_points = mapped_points
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.new_image_labels = new_image_labels
        self.new_reprocess_image_labels = new_reprocess_image_labels
        self.additional_embeddings = additional_embeddings
        self.neighbors = neighbors - 1
        self.score_list = []
        self.mean_score = []
        self.USER_PREF = USER_PREF
        # TODO: Make it reflect a generic score.
        self.score_df = pd.read_excel(
            self.USER_PREF.CSV_DATA_FILE_FOR_QE, sheet_name=1, index_col=0, engine='openpyxl')
        self._process()

    def _calc_k_point_value(self, new_point):
        dst_list = np.asarray([np.linalg.norm((new_point - mapped_point))
                              for mapped_point in self.mapped_points[:-len(self.new_image_labels)]])
        top_indexs = np.argsort(dst_list)[0:self.neighbors]
        self.score_list = []
        self.mean_score = []
        for image_path in self.image_paths[top_indexs]:
            self.score_list.append(
                self.score_df[self.score_df[f'実ファイル名({self.USER_PREF.DATA_PHOTO_TYPE.value})'] == image_path].values[0][
                    1:len(self.USER_PREF.SCORE_ITEMS) + 1])
        score_list = np.asarray(self.score_list).T.astype('float64')
        weights_list = np.asarray(
            1/(dst_list[top_indexs]+1e-6)).astype('float64')
        self.mean_score = np.average(
            a=score_list, weights=weights_list, axis=1)
        self.mode_score = stats.mode(score_list, axis=1)[0]
        return self.mean_score, self.mode_score

    def _process(self):
        columns = np.append('実ファイル名', self.USER_PREF.SCORE_ITEMS)
        mean_score_items = []
        mode_score_items = []
        for index, additional_embedding in enumerate(self.additional_embeddings):
            image_path = self.image_paths[
                (index+len(self.new_reprocess_image_labels)) - (len(self.new_image_labels))]
            mean_scores, mode_scores = self._calc_k_point_value(
                additional_embedding)
            mean_score_items.append(np.append(image_path, mean_scores))
            mode_score_items.append(np.append(image_path, mode_scores))
        mean_score_items = np.asarray(mean_score_items)
        mode_score_items = np.asarray(mode_score_items)
        mean_score_df = pd.DataFrame(mean_score_items, columns=columns)
        mode_score_df = pd.DataFrame(mode_score_items, columns=columns)

        os.makedirs(self.USER_PREF.SCORE_OUTPUT_DIR, exist_ok=True)
        mean_score_df.to_csv(os.path.join(self.USER_PREF.SCORE_OUTPUT_DIR,
                                          f'mean_{self.USER_PREF.DATA_NAME}_{self.USER_PREF.DR_TYPE.name}_{self.USER_PREF.TIME_STAMP}.csv'))
        mode_score_df.to_csv(os.path.join(self.USER_PREF.SCORE_OUTPUT_DIR,
                                          f'mode_{self.USER_PREF.DATA_NAME}_{self.USER_PREF.DR_TYPE.name}_{self.USER_PREF.TIME_STAMP}.csv'))
