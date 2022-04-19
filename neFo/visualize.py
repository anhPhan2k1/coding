'''


'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VisualizeClassification():
    def __init__(self, path_to_df, graph_title):
        df = pd.read_csv(path_to_df)
        print(df.head(2))
        self.graph_title = graph_title # DB name or Model name
        self.thresh = df['thresh'].to_list()
        self.TP = df['TP'].to_list()
        self.FP = df['FP'].to_list()
        self.TN = df['TN'].to_list()
        self.FN = df['FN'].to_list()
        self.ACC = df['Accuracy'].to_list()
        self.F1 = df['F1-Score'].to_list()
        self.FAR = df['FAR'].to_list()
        self.FRR = df['FRR'].to_list()
        
    def plot_far_frr(self):
        EER = [abs(self.FAR[i] - self.FRR[i]) for i in range(len(self.thresh))]
        self.eer_index = EER.index(min(EER))
        self.optimized_f1 = max(self.F1)
        self.optimized_f1_idx = self.F1.index(self.optimized_f1)

        t = np.array(self.thresh)
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_title(self.graph_title, size= 20)
        ax1.set_xlabel('thr')
        ax1.set_ylabel('FAR', color=color)
        ax1.plot(t, self.FAR, 'o-', color=color)
        ax1.plot(t, self.F1, 'g^-')
        ax1.plot(t, self.ACC, 'yv-')
        ax1.text(0.01, 0.8,
                 f'Optimized \n'
                 f'F1 {self.optimized_f1:.4f} at thr: {self.thresh[self.optimized_f1_idx]: .3f} \n'
                 f'ACC {self.ACC[self.eer_index]: .4f} at thr: {self.thresh[self.eer_index]: .3f} \n'
                 f'FAR {self.FAR[self.eer_index]: .4f} at thr: {self.thresh[self.eer_index]: .3f} \n'
                 f'FRR {self.FRR[self.eer_index]: .4f} at thr: {self.thresh[self.eer_index]: .3f}',
                 size=10,
                 ha="left", va="center",
                 bbox=dict(boxstyle="round",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           )
                 )

        ax1.legend(['FAR', 'F1', 'ACC'], loc='upper left')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('FRR', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, self.FRR, '*-', color=color)
        ax2.legend(['FRR'], loc='upper right')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_size_inches((18, 9), forward=False)
        # plt.savefig(f'/media/hungdd21/DATA/HungDo/Data/Mask/benchmark_data/D26_all_mask/results_benchmark_d26mask/{self.graph_title}.jpg', dpi=250)
        
    def plot_confusion(self, target_thresh):
        return 0


class VisualizeDetection():
    def __init__(self, path_to_df, graph_title):
        df = pd.read_csv(path_to_df)
        print(df.head(2))
        self.graph_title = graph_title # DB name or Model name
        self.thresh = df['thresh'].to_list()
        self.precision = df['Precision'].to_list()
        self.recall = df['Recall'].to_list()


if __name__ == '__main__':
    # # Classification
    # VC = VisualizeClassification('./example_classify.csv', 'testDB')
    # VC.plot_far_frr()

    # Detection
    VD = VisualizeDetection('./example_detection.csv', 'testDB')
    plt.show()
