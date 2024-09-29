import pandas as pd
import pm4py
from pm4py.objects.log.util import sorting
import numpy as np
import glob
from scipy.stats import wasserstein_distance
from sklearn import preprocessing
import scipy.stats as st
import pathlib


def convert_log(path, timestamp):
    dataframe = pd.read_csv(path, sep=',')
    dataframe = pm4py.format_dataframe(dataframe, case_id='caseid', activity_key='task', timestamp_key=timestamp)
    log = pm4py.convert_to_event_log(dataframe)
    return log


def define_cycle(log, start_timestamp, end_timestamp):
    log = sorting.sort_timestamp_log(log)
    cycle_time_real = []
    for trace in log:
        cycle_time_real.append((trace[-1][end_timestamp]-trace[0][start_timestamp]).total_seconds())
    cycle_time_real.sort()
    return cycle_time_real


def compute_MAE(log, log1, start_timestamp, end_timestamp):
    cycle1 = define_cycle(log, start_timestamp, end_timestamp)
    cycle2 = define_cycle(log1, start_timestamp, end_timestamp)
    diff = len(cycle1) - len(cycle2)
    if diff != 0:
        cycle2 = cycle2 + [0]*diff
        cycle2.sort()
    mae = []
    for i in range(0, len(cycle1)):
        mae.append(abs(cycle1[i]-cycle2[i]))
    return np.mean(mae)


def extract_time_activity(log, task, start_timestamp, end_timestamp):
    time_activity = dict()
    for trace in log:
        for event in trace:
            if event['concept:name'] == task:
                key = event[start_timestamp].replace(minute=0, second=0)
                if key in time_activity:
                    time_activity[key] += (event[end_timestamp]-event[start_timestamp]).total_seconds()
                else:
                    time_activity[key] = (event[end_timestamp]-event[start_timestamp]).total_seconds()
    return time_activity


def extract_time_log(log, dates, start_timestamp, end_timestamp):
    time_activity = dict()
    for d in dates:
        time_activity[d] = 0

    for trace in log:
        for event in trace:
            key1 = event[start_timestamp].replace(minute=0, second=0)
            time_activity[key1] += 1
            key2 = event[end_timestamp].replace(minute=0, second=0)
            time_activity[key2] += 1
    return time_activity


def extract_set_date(log, log1, start_timestamp, end_timestamp):
    dates = set()
    for trace in log:
        for event in trace:
            start = event[start_timestamp].replace(minute=0, second=0)
            end = event[end_timestamp].replace(minute=0, second=0)
            dates.add(start)
            dates.add(end)
    for trace in log1:
        for event in trace:
            start = event[start_timestamp].replace(minute=0, second=0)
            end = event[end_timestamp].replace(minute=0, second=0)
            dates.add(start)
            dates.add(end)

    dates = list(dates)
    dates.sort()
    return dates


def normalize(times):
    values = list(times.values())
    max_v = max(values)
    for i in range(0, len(values)):
        values[i] = values[i]/max_v
    return values


def confidence_interval(data):
    n = len(data)
    C = 0.95
    alpha = 1 - C
    tails = 2
    q = 1 - (alpha / tails)
    dof = n - 1
    t_star = st.t.ppf(q, dof)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    ci_upper = x_bar + t_star * s / np.sqrt(n)
    ci_lower = x_bar - t_star * s / np.sqrt(n)

    print('CI ', ci_lower, ci_upper)
    return ci_lower, ci_upper

def evaluation_sim(NAME_EXP, type):
    file_path = str(pathlib.Path().resolve())

    if type == 'rims' or type == 'rims_plus':
        end_timestamp = 'time:timestamp'
        start_timestamp = 'start:timestamp'
        path = file_path + '/RIMS/' + NAME_EXP + '/results/' + type
        all_file = glob.glob(path + '/sim*')
        path_test = path + '/tst_' + NAME_EXP + '.csv'
    elif type == 'DDPS':
        end_timestamp = 'end_timestamp'
        start_timestamp = 'start_timestamp'
        path = file_path + '/DDPS_models_data/' + NAME_EXP
        all_file = glob.glob(path + '/sim*')
        path_test = file_path + '/DDPS_models_data/' + NAME_EXP + '/tst_' + NAME_EXP + '.csv'
    elif type == 'LSTM':
        end_timestamp = 'end_timestamp'
        start_timestamp = 'start_timestamp'
        path = file_path + '/LSTM_model_data/' + NAME_EXP
        all_file = glob.glob(path + '/gen*')
        path_test = file_path + '/LSTM_model_data/' + NAME_EXP + '/tst_' + NAME_EXP + '.csv'
    else:
        end_timestamp = 'time:timestamp'
        start_timestamp = 'start:timestamp'
        path = file_path + '/DSIM/' + NAME_EXP
        all_file = glob.glob(path + '/gen*')
        path_test = file_path + '/RIMS/' + NAME_EXP + '/results/rims/tst_' + NAME_EXP + '.csv'

    real_tst = convert_log(path_test, end_timestamp)
    MAE = dict()
    EMD_normalize = dict()
    LEN = dict()

    for idx, file in enumerate(all_file):
        sim_tst = convert_log(file, end_timestamp)

        dates = extract_set_date(sim_tst, real_tst, start_timestamp, end_timestamp)

        real = extract_time_log(real_tst, dates, start_timestamp, end_timestamp)
        sim = extract_time_log(sim_tst, dates, start_timestamp, end_timestamp)

        real = list(real.values())
        sim = list(sim.values())

        LEN[idx] = len(sim_tst)
        EMD_normalize[idx] = wasserstein_distance(preprocessing.normalize([real])[0], preprocessing.normalize([sim])[0])
        MAE[file] = compute_MAE(real_tst, sim_tst, start_timestamp, end_timestamp)

    print('MEAN MAE', np.mean(list(MAE.values())))
    if len(all_file) > 4:
        confidence_interval(list(MAE.values()))
    print('NORMALIZE emd', np.mean(list(EMD_normalize.values())))
    if len(all_file) > 4:
        confidence_interval(list(EMD_normalize.values()))

    return MAE, EMD_normalize, len(all_file)