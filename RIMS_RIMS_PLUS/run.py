from datetime import datetime
import csv
import simpy
from checking_process import SimulationProcess
from token_LSTM import Token
from MAINparameters import Parameters
import sys
import getopt
import warnings
from os.path import exists
from evaluate import *


def main(argv):
    opts, args = getopt.getopt(argv, "h:t:l:n:")
    NAME_EXPERIMENT = 'confidential_1000'
    for opt, arg in opts:
        if opt == '-h':
            print(
                'main.py -t <[rims, rims_plus]> -l <log_name> -n <total number of simulation [1, 25]>')
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
    run_simulation(NAME_EXPERIMENT, N_SIMULATION, type)
    MAE, EMD_normalize, ln = evaluation_sim(NAME_EXPERIMENT, type)

    return MAE, EMD_normalize, ln


def setup(env: simpy.Environment, NAME_EXPERIMENT, params, i, type):
    simulation_process = SimulationProcess(env=env, params=params)

    if type == 'rims':
        path_result = NAME_EXPERIMENT + '/results/rims/simulated_log_LSTM_' + \
            NAME_EXPERIMENT + str(i) + '.csv'
    else:
        path_result = NAME_EXPERIMENT + '/results/rims_plus/simulated_log_LSTM_' + \
            NAME_EXPERIMENT + str(i) + '.csv'
    f = open(path_result, 'w')

    writer = csv.writer(f)
    writer.writerow(['caseid', 'task', 'start:timestamp',
                    'time:timestamp', 'role', 'st_wip', 'st_tsk_wip', 'queue'])
    prev = params.START_SIMULATION
    for i in range(1, len(params.ARRIVALS) + 1):
        next = datetime.strptime(
            params.ARRIVALS.loc[i - 1].at["timestamp"], '%Y-%m-%d %H:%M:%S')
        interval = (next - prev).total_seconds()
        prev = next
        yield env.timeout(interval)
        # in case of SynLoan, and L1_syn, .... L6_syn, set simulation input as: (env, writer, type aand True)
        env.process(Token(i, params, simulation_process,
                    params).simulation(env, writer, type))


def run_simulation(NAME_EXPERIMENT, N_SIMULATION, type):
    path_model = NAME_EXPERIMENT + '/' + type + '/' + NAME_EXPERIMENT
    if exists(path_model + '_diapr_meta.json'):
        FEATURE_ROLE = 'all_role'
    elif exists(path_model + '_dispr_meta.json'):
        FEATURE_ROLE = 'no_all_role'
    else:
        raise ValueError(
            f'LSTM models do not exist in the right folder\n{path_model}_dispr_meta.json')
    for i in range(0, N_SIMULATION):
        params = Parameters(NAME_EXPERIMENT, FEATURE_ROLE, i, type)
        env = simpy.Environment()
        env.process(setup(env, NAME_EXPERIMENT, params, i, type))
        env.run(until=params.SIM_TIME)


def CI(data):
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

    return ci_lower, ci_upper


# conda activate azureml_py310_sdkv2
if __name__ == "__main__":
    dataset_list_RIMS = ["BPI_Challenge_2012_W_Two_TS", "BPI_Challenge_2012_W_Two_TS_2", "BPI_Challenge_2017_W_Two_TS", "confidential_2000",
                         "SynLoan", "confidential_1000", "ConsultaDataMining201618", "cvs_pharmacy",
                         "Productions", "PurchasingExample"]

    dataset_list_RIMS_plus = ["BPI_Challenge_2012_W_Two_TS", "BPI_Challenge_2012_W_Two_TS_2", "BPI_Challenge_2017_W_Two_TS",
                              "confidential_1000", "cvs_pharmacy", "SynLoan"]
    iteration = 1
    for dataset_name in dataset_list_RIMS:
        warnings.filterwarnings("ignore")

        # Run the main function with provided arguments
        mean_absolute_error, normalized_emd, length = main(
            ["-t", "rims", "-l", dataset_name, "-n", str(iteration)])

        # If length is greater than 4, calculate confidence intervals
        if length > 4:
            ci_lower_mae, ci_upper_mae = CI(list(mean_absolute_error.values()))
            ci_lower_emd, ci_upper_emd = CI(list(normalized_emd.values()))
        else:
            ci_lower_mae, ci_upper_mae = None, None
            ci_lower_emd, ci_upper_emd = None, None

        with open("RIMS.csv", "a") as file:
            writer = csv.writer(file)

            # Write the header if the file is empty
            if file.tell() == 0:
                writer.writerow(["Dataset", "Iteration", "MAE", "CI_Lower_MAE",
                                "CI_Upper_MAE", "EMD", "CI_Lower_EMD", "CI_Upper_EMD"])

            writer.writerow([
                dataset_name, iteration,
                np.mean(list(mean_absolute_error.values())
                        ), ci_lower_mae, ci_upper_mae,
                np.mean(list(normalized_emd.values())
                        ), ci_lower_emd, ci_upper_emd
            ])

        if dataset_name in dataset_list_RIMS_plus:
            warnings.filterwarnings("ignore")

            # Run the main function with provided arguments
            mean_absolute_error, normalized_emd, length = main(
                ["-t", "rims_plus", "-l", dataset_name, "-n", str(iteration)])

            # If length is greater than 4, calculate confidence intervals
            if length > 4:
                ci_lower_mae, ci_upper_mae = CI(
                    list(mean_absolute_error.values()))
                ci_lower_emd, ci_upper_emd = CI(list(normalized_emd.values()))
            else:
                ci_lower_mae, ci_upper_mae = None, None
                ci_lower_emd, ci_upper_emd = None, None

            with open("RIMS_PLUS.csv", "a") as file:
                writer = csv.writer(file)

                # Write the header if the file is empty
                if file.tell() == 0:
                    writer.writerow(["Dataset", "Iteration", "MAE", "CI_Lower_MAE",
                                    "CI_Upper_MAE", "EMD", "CI_Lower_EMD", "CI_Upper_EMD"])

                writer.writerow([
                    dataset_name, iteration,
                    np.mean(list(mean_absolute_error.values())
                            ), ci_lower_mae, ci_upper_mae,
                    np.mean(list(normalized_emd.values())
                            ), ci_lower_emd, ci_upper_emd
                ])

            print(f"Completed processing for {dataset_name}")
