import quandl
import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import multivariate_normal
import os

def load_data():
    quandl.ApiConfig.api_key = 'zMPwcfnWy9rynuPgbiyT'
    myr = quandl.get("BOE/XUDLBK66")
    krw = quandl.get("BOE/XUDLBK74")
    rub = quandl.get("BOE/XUDLBK69")
    brl = quandl.get("FED/RXI_N_B_BZ")
    eur = quandl.get("BOE/XUDLERD")
    jpy = quandl.get("BOE/XUDLJYD")
    chf = quandl.get("BOE/XUDLSFD")
    ccy_list = ['myr', 'krw', 'rub', 'brl', 'eur', 'jpy', 'chf']

    global fx_data
    fx_data = quandl.get(
        ["BOE/XUDLBK66", "BOE/XUDLBK74", "BOE/XUDLBK69", "FED/RXI_N_B_BZ", "BOE/XUDLERD", "BOE/XUDLJYD", "BOE/XUDLSFD"])

    ccy_list = ['myr', 'krw', 'rub', 'brl', 'eur', 'jpy', 'chf']
    fx_data.columns = ccy_list

    fx_data.sort_index(axis=0, ascending=False, inplace=True)
    fx_data = fx_data[dt.datetime(2016, 10, 1):dt.datetime(2009, 1, 1)]
    fx_data = fx_data.fillna(method='backfill')
    fx_data.sort_index(axis=0, ascending=False, inplace=True)
    fx_data.to_csv('fx_data.csv')



def model():
    cdir = os.getcwd()
    fx_data = pd.read_csv('fx_data.csv',index_col=['Date'], parse_dates=True)
    fx_data_sample = fx_data[dt.datetime(2016, 1, 1):]
    fx_data_sample.fillna(method='backfill')
    # fx_data_sample.sort_index(axis=0, ascending=True, inplace=True)
    fx_data_sample.fillna(method = 'pad')
    ret_data = fx_data_sample.pct_change(periods=1)

    regime_number = 5
    mean_ret = ret_data.mean()
    mean_ret = mean_ret.as_matrix()

    std_data = ret_data.rolling(window=20, center=False).std()
    mean_data = ret_data.rolling(window=20, center=False).std()
    std_data = std_data.fillna(method='backfill')
    mean_data = mean_data.fillna(method='backfill')

    fx_cov = ret_data.cov().as_matrix()
    fx_corr = ret_data.cov().as_matrix()


    #get the initial guesses

    fx_cov_seg = []
    fx_mean_seg = []
    for i in (range(5)):
        fx_mean_seg.append(
            ret_data.iloc[int(i * len(ret_data) / 5):int((i + 1) * len(ret_data) / 5)]
            .mean().as_matrix())
        fx_cov_seg.append(
            ret_data.iloc[int(i * len(ret_data) / 5):int((i + 1) * len(ret_data) / 5)]
            .cov().as_matrix())

    regime_ret = []
    regime_corr = []
    regime_cov = []
    for i in (range(1, 6)):
        regime_ret.append(fx_mean_seg[i - 1])
        cov_i = fx_cov_seg[i - 1]
        inv_diag = (np.linalg.inv(np.diagflat(np.sqrt(np.diag(cov_i)))))
        corr_i = np.dot(np.dot(inv_diag, cov_i), inv_diag)
        regime_cov.append(cov_i)
        regime_corr.append(corr_i)


    #get the condtional probabilities

    # pi = np.zeros((len(ret_data), 5))
    fx_cov = ret_data.cov().as_matrix()
    # init_prob = np.array([0.3, 0.2, 0.15, 0.25, 0.1])
    # prob_regime = init_prob


    iter_count = 0
    while (iter_count<90):

        pi = np.zeros((len(ret_data), 2))
        fx_cov = ret_data.cov().as_matrix()
        init_prob = np.ones(2) * 0.5
        prob_regime = init_prob

        # get the probability
        for idx in range(len(ret_data)):
            print (idx)
            mean_t = mean_data.iloc[idx].as_matrix()
            ret_t = ret_data.iloc[idx].as_matrix()
            for regime_no in range(2):
                prob_s = 0
                cov_regime_t = regime_cov[regime_no]
                prob_regime_t = multivariate_normal.pdf(ret_t, mean=regime_ret[regime_no], cov=cov_regime_t)
                for regime_no2 in range(2):
                    prob_s += prob_regime[regime_no2] * multivariate_normal.pdf(ret_t, mean=regime_ret[regime_no2],cov=regime_cov[regime_no2])
                pi[idx][regime_no] = prob_regime[regime_no] * prob_regime_t / prob_s

        # get the distribution for each regime
        #mean -ret
        prob_regime_hat = np.zeros(2)
        for regime_no in range(2):
            mean_regime_sum = 0
            prob_sum = 0
            for t in range(1, len(pi)):
                ret_t = ret_data.iloc[t].as_matrix()
                prob_sum += pi[t][regime_no]
                mean_regime_sum += ret_t * pi[t][regime_no]
            mean_hat = mean_regime_sum / prob_sum
            prob_regime_hat[regime_no] = prob_sum / len(pi)
            regime_ret[regime_no] = mean_hat

        prob_regime = prob_regime_hat

        # #variance
        # regime_variance = []
        # for regime_no in range(2):
        #     regime_variance.append(np.zeros((7, 7)))
        # for regime_no in range(2):
        #     variance = np.zeros((7, 7))
        #     prob_sum = 0
        #     for t in range(1, len(pi)):
        #         prob_sum += pi[t][regime_no]
        #         ret_t = ret_data.iloc[t].as_matrix()
        #         ret_t_demeaned = (ret_t - regime_ret[regime_no])
        #         variance += np.outer(ret_t_demeaned, ret_t_demeaned) * pi[t][regime_no]
        #     variance_hat = variance / prob_sum
        #     regime_variance[regime_no] = variance_hat

        #cov matrix
        for regime_no in range(2):
            matrix = np.zeros((7, 7))
            prob_sum = 0
            for t in range(1, len(pi)):
                prob_sum += pi[t][regime_no]
                ret_t = ret_data.iloc[t].as_matrix()
                ret_t_demeaned = (ret_t - regime_ret[1])
                matrix += np.outer(ret_t_demeaned, ret_t_demeaned) * pi[t][regime_no]
            cov_hat = matrix / prob_sum
            regime_cov[regime_no] = cov_hat
        #transition_matrix
        transition_matrix = np.zeros((2, 2))
        for regime_no in range(2):
            for regime_no_p in range(2):
                transition_prob = 0  # this is the estimator we want
                transition_prob_regime = 0
                regime_prob = 0
                for t in range(2, len(pi)):
                    prob0 = pi[t][regime_no]
                    probp = pi[t - 1][regime_no_p]
                    transition_prob_regime += prob0 * probp
                    regime_prob += probp
                    #             print(regime_no_p)
                    #             print(probp)
                transition_prob = transition_prob_regime / regime_prob
                transition_matrix[regime_no_p][regime_no] = transition_prob

            inv_diag = (np.linalg.inv(np.diagflat(np.sqrt(np.diag(regime_cov[regime_no])))))
            corr_hat = np.dot(np.dot(inv_diag, regime_cov[regime_no]), inv_diag)

            pd.DataFrame(regime_cov[regime_no], index=fx_data.columns, columns=fx_data.columns).to_csv(
                cdir + '/cache/iter_' + str(iter_count) + '_cov_regime' + str(regime_no) + '.csv')
            pd.DataFrame(corr_hat, index=fx_data.columns, columns=fx_data.columns).to_csv(
                cdir + '/cache/iter_' + str(iter_count) + '_corr_regime' + str(regime_no) + '.csv')
            pd.DataFrame(regime_ret[regime_no], index=fx_data.columns).to_csv(
                cdir + '/cache/iter_' + str(iter_count) + '_mean_regime' + str(regime_no) + '.csv')


        iter_count +=1


        pd.DataFrame(pi).to_csv(cdir + '/cache/number_' + str(iter_count) + '_prob' + '.csv')

        pd.DataFrame(transition_matrix).to_csv(
            cdir + '/cache/iter_' + str(iter_count) + 'transition.csv')


if __name__== '__main__':

    model()
