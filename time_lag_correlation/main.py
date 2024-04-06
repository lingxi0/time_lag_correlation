import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from numpy import newaxis, concatenate, linspace, array, min, max
from scipy.interpolate import make_interp_spline
from scipy.stats import pearsonr, spearmanr, kendalltau
from matplotlib import rc


def create_data(dataset, window_len: int, step_len: int):
    scaled = dataset.values

    x = []
    for z in range((scaled.shape[0] - window_len) // step_len):
        x.append(scaled[step_len * z:(step_len * z + window_len), :])
    x = array(x, dtype='float32')

    return x


class CreateCorrelation:
    def __init__(self, start: int, end: int, interval: int, correlation: str):
        """
        :param start: The starting position of lag
        :param end: The ending position of lag
        :param interval: The interval of lag
        :param interval: The method of correlation
        """
        # get all files in data file
        all_data_file = os.listdir('./data/')
        all_csv = [i for i in all_data_file if i.endswith('.csv')]
        if len(all_csv) == 0:
            raise 'Please add CSV files in the data folder'

        # read all csv files in data file
        self.data = [read_csv('./data/' + i, index_col=0, header=0) for i in all_csv]
        self.variables_num = self.data[0].shape[1]
        self.variables_name = self.data[0].columns

        if start >= end:
            raise 'The starting position must be smaller than the ending position'
        self.start = start
        self.end = end
        self.interval = interval

        if correlation not in ['pearsonr', 'spearmanr', 'kendalltau']:
            raise 'The calculation method for correlation coefficient is only: pearsonr/spearmanr/kendalltau'
        self.method = correlation

        self.x = None
        self.cc = None

    def add_lag(self, target: int):
        """
        Add different lag values for the base variable to each piece of data
        :param target: Which variable is used as the base for lag calculation
        """
        name = self.data[0].columns[target]
        for tmp in self.data:
            for i in range(self.start, self.end, self.interval):
                str_tmp = name + '_' + str(i)
                d = tmp.iloc[:, target].shift(i).copy()
                tmp.loc[:, str_tmp] = d
            tmp.dropna(inplace=True)

    def creat_data(self, window_len: int, step_len: int):
        """
        Using sliding time windows to obtain data from different windows for future calculations
        """
        x = create_data(self.data[0], window_len, step_len)
        for i in range(1, len(self.data), 1):
            x1 = create_data(self.data[i], window_len, step_len)
            x = concatenate((x, x1), axis=0)
        self.x = x

    def _computer_4_pearsonr_and_kendalltau(self, i):
        if self.method == 'pearsonr':
            tmp_method = pearsonr
        else:
            tmp_method = kendalltau

        cc1 = np.zeros((i.shape[1], i.shape[1]))
        for m in range(i.shape[1]):
            for n in range(m, i.shape[1]):
                cc1[m, n] = tmp_method(i[:, m], i[:, n])[0]
        return cc1

    def compute_correlation(self, save_or_not: bool):
        tmp_cc = []
        for i in self.x:
            if self.method == 'pearsonr':
                cc1 = self._computer_4_pearsonr_and_kendalltau(i)
            elif self.method == 'spearmanr':
                cc1 = spearmanr(i, axis=0)[0]
            else:
                cc1 = self._computer_4_pearsonr_and_kendalltau(i)

            cc1 = cc1[0:self.variables_num, self.variables_num:]
            cc1 = cc1[:, newaxis, :]
            tmp_cc.append(cc1)

        cc = concatenate(tmp_cc, axis=1)
        self.cc = cc.mean(axis=1)
        if save_or_not:
            cc = cc.mean(axis=1)
            cc = DataFrame(cc)
            cc.to_csv('./result/' + self.method + '_lag_all.csv')

    def smooth_and_plot(self):
        self.cc = read_csv('./result/' + self.method + '_lag_all.csv', header=0, index_col=0).values
        b = array([i for i in range(self.start, self.end, self.interval)])
        b_new = linspace(min(b), max(b), (self.end - self.start) * 10)
        cc_new = make_interp_spline(b, self.cc[0])(b_new)
        cc_new = cc_new[:, newaxis]
        for i in range(1, self.cc.shape[0]):
            a_smooth = make_interp_spline(b, self.cc[i])(b_new)
            a_smooth = a_smooth[:, newaxis]
            cc_new = concatenate((cc_new, a_smooth), axis=1)

        rc('font', size=30, family='Times new roman')
        fig = plt.figure(figsize=(40, 20), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        for i in range(0, cc_new.shape[1]):
            ax.plot(b_new, cc_new[:, i], lw=2)
        ax.grid()
        ax.set_xticks([i for i in range(self.start, self.end, self.interval * 2)])
        ax.set_xlim(self.start - 0.5, self.end - 0.5)
        ax.set_xlabel('Different lag', fontsize='x-large')
        ax.set_ylabel('Spearman correlation coefficient', fontsize='x-large')
        ax.legend(self.variables_name)

        plt.tight_layout()
        plt.savefig('./result/' + self.method + '_lag_all.svg', format='svg')
        plt.close("all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A program for calculating time lag correlation')

    parser.add_argument('--which_correlation', type=str, default='kendalltau',
                        choices=['pearsonr', 'spearmanr', 'kendalltau'],
                        help='What correlation calculation method to choose')

    parser.add_argument('--start_position', type=int, default=-30, help='The starting position of lag')
    parser.add_argument('--ending_position', type=int, default=1, help='The ending position of lag')
    parser.add_argument('--interval', type=int, default=1, help='The interval of lag')

    parser.add_argument('--target', type=int, default=0, help='Target variable column number')

    parser.add_argument('--window_len', type=int, default=60, help='The length of the sliding window')
    parser.add_argument('--step_len', type=int, default=30, help='Moving step size of sliding window')

    parser.add_argument('--if_save', type=bool, default=True, help='Whether to save the lag correlation matrix')

    args = parser.parse_args()

    a = CreateCorrelation(args.start_position, args.ending_position, args.interval, args.which_correlation)
    a.add_lag(args.target)
    a.creat_data(args.window_len, args.step_len)
    a.compute_correlation(args.if_save)
    a.smooth_and_plot()
