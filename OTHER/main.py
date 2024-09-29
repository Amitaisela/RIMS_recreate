from datetime import datetime
import csv
import simpy
from checking_process import SimulationProcess
from token_LSTM import Token
from MAINparameters import Parameters
import sys, getopt
import warnings
from os.path import exists
from evaluate import *
import pm4py


def main(argv):
    opts, args = getopt.getopt(argv, "h:t:l:n:")
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -t <[rims, rims_plus]> -l <log_name> -n <total number of simulation [1, 25]>')
            sys.exit()
        elif opt == "-t":
            type = arg
        elif opt == "-l":
            NAME_EXPERIMENT = arg
        elif opt == "-n":
            N_SIMULATION = int(arg)
            if N_SIMULATION > 25:
                N_SIMULATION = 25
    print(NAME_EXPERIMENT, N_SIMULATION, type)
    log = pm4py.read_xes(f'Logs/{NAME_EXPERIMENT}.xes')

    df = pm4py.convert_to_dataframe(log)

    run_simulation(NAME_EXPERIMENT, N_SIMULATION, type, df)
    MAE, EMD_normalize, LEN = evaluation_sim(NAME_EXPERIMENT, type)
    
    if LEN > 4:
        EMD_CI_lower, EMD_CI_higher  = confidence_interval(list(EMD_normalize.values()))
        MAE_CI_lower, MAE_CI_higher  = confidence_interval(list(MAE.values()))
        
    MAE = np.mean(list(MAE.values()))
    EMD_normalize = np.mean(list(EMD_normalize.values()))
    
    with open("results.csv", "a") as f:
        # write mae, emd, CI's
        f.write(f"{NAME_EXPERIMENT},{type},{MAE_CI_lower},{MAE_CI_higher},{EMD_CI_lower},{EMD_CI_higher}\n")
def setup(env: simpy.Environment, NAME_EXPERIMENT, params, i, type):
    simulation_process = SimulationProcess(env=env, params=params)

    if type == 'rims':
        path_result ='RIMS/results/simulated_log_LSTM_' + NAME_EXPERIMENT + str(i) + '.csv'
    elif type == 'DDPS':
        path_result = 'DDPS_models_data/results/simulated_log_LSTM_' + NAME_EXPERIMENT + str(i) + '.csv'
    elif type == 'DSIM':
        path_result = 'DSIM/results/simulated_log_LSTM_' + NAME_EXPERIMENT + str(i) + '.csv'
    elif type == 'LSTM':
        path_result = 'LSTM_model_data/results/simulated_log_LSTM_' + NAME_EXPERIMENT + str(i) + '.csv'
    else:
        path_result = 'RIMS/results/simulated_log_LSTM_' + NAME_EXPERIMENT + str(i) + '.csv'

    f = open(path_result, 'w')
    writer = csv.writer(f)
    writer.writerow(['caseid', 'task', 'start:timestamp', 'time:timestamp', 'role', 'st_wip', 'st_tsk_wip', 'queue'])
    
    prev = params.START_SIMULATION
    for i in range(1, len(params.ARRIVALS) + 1):
        if type == 'rims' or type == 'rims_plus':
            next = datetime.strptime(params.ARRIVALS.loc[i - 1].at["timestamp"], '%Y-%m-%d %H:%M:%S')
        if type == 'DDPS' or type == 'LSTM':
            if type == 'LSTM' and NAME_EXPERIMENT in ["SynLoan", "ConsultaDataMining201618", "PurchasingExample", "BPI_Challenge_2017_W_Two_TS","Productions"]:
                next = datetime.strptime(params.ARRIVALS.loc[i - 1].at["start_timestamp"], '%Y-%m-%d %H:%M:%S')
            else:
                next = datetime.strptime(params.ARRIVALS.loc[i - 1].at["start_timestamp"], '%Y-%m-%d %H:%M:%S.%f')
    
        if type == 'DSIM':
            next = datetime.strptime(params.ARRIVALS.loc[i - 1].at["start:timestamp"], '%Y-%m-%d %H:%M:%S')

        interval = (next - prev).total_seconds()

        if interval < 0:
            interval = 0

        prev = next
        yield env.timeout(interval)

        if type == 'DDPS':
            env.process(Token(i, params, simulation_process, params).simulation(env, writer, type))
        else:
            env.process(Token(i, params, simulation_process, params).simulation(env, writer, type))


def run_simulation(NAME_EXPERIMENT, N_SIMULATION, type, event_log_df):
    for i in range(0, N_SIMULATION):
        params = Parameters(NAME_EXPERIMENT, i, type, event_log_df)
        env = simpy.Environment()
        env.process(setup(env, NAME_EXPERIMENT, params, i, type))

        env.run(until=params.SIM_TIME)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(sys.argv[1:])



