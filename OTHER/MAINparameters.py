'''
Principal parameters to run the process

DA AGGIUNGERE: configurazione risorse, tempi per ogni attivita'
'''
import json
import os
from datetime import datetime

import pandas as pd


class Parameters(object):

    def __init__(self, name_exp, iterations, type1, event_log_df):
        self.NAME_EXP = name_exp
        self.type1 = type1
        self.event_log_df = event_log_df
        datasets = ["BPI_Challenge_2012_W_Two_TS", "ConsultaDataMining201618", "PurchasingExample", "confidential_1000",
                    "cvs_pharmacy", "BPI_Challenge_2017_W_Two_TS", "Productions", "SynLoan", "confidential_2000"]

        RIMS_with_A = ["BPI_Challenge_2017_W_Two_TS", "confidential_1000", "confidential_2000",
                       "ConsultaDataMining201618", "cvs_pharmacy", "SynLoan"]
        RIMS_plus_with_A = ["BPI_Challenge_2012_W_Two_TS", "BPI_Challenge_2017_W_Two_TS", "confidential_1000",
                            "cvs_pharmacy", "SynLoan", "BPI_Challenge_2012_W_Two_TS_2"]
        RIMS_with_S = ["BPI_Challenge_2012_W_Two_TS", "Productions",
                       "PurchasingExample", "BPI_Challenge_2012_W_Two_TS_2"]
        RIMS_plus_with_S = []

        if self.type1 == 'rims_plus':
            if self.NAME_EXP in RIMS_plus_with_A:
                self.prefix = ('_diapr', '_dpiapr', '_dwiapr')
            elif self.NAME_EXP in RIMS_plus_with_S:
                self.prefix = ('_dispr', '_dpispr', '_dwispr')
        else:
            if self.NAME_EXP in RIMS_with_A:
                self.prefix = ('_diapr', '_dpiapr', '_dwiapr')
            elif self.NAME_EXP in RIMS_with_S:
                self.prefix = ('_dispr', '_dpispr', '_dwispr')

        # self.prefix = ('_dispr', '_dpispr', '_dwispr')

        if self.type1 == 'rims_plus':
            self.PATH_PETRINET = 'RIMS/' + self.NAME_EXP + \
                '/' + type1 + '/' + self.NAME_EXP + '.pnml'
            self.MODEL_PATH_PROCESSING = 'RIMS/' + self.NAME_EXP + '/' + \
                type1 + '/' + self.NAME_EXP + self.prefix[1] + '.h5'
            self.MODEL_PATH_WAITING = 'RIMS/' + self.NAME_EXP + '/' + \
                type1 + '/' + self.NAME_EXP + self.prefix[2] + '.h5'
        else:
            self.PATH_PETRINET = 'RIMS/' + self.NAME_EXP + '/rims/' + self.NAME_EXP + '.pnml'
            self.MODEL_PATH_PROCESSING = 'RIMS/' + self.NAME_EXP + \
                '/rims/' + self.NAME_EXP + self.prefix[1] + '.h5'
            self.MODEL_PATH_WAITING = 'RIMS/' + self.NAME_EXP + \
                '/rims/' + self.NAME_EXP + self.prefix[2] + '.h5'

        # self.SIM_TIME = 1460*36000000000000000  # 10 day
        # self.SIM_TIME = 1460 * 800
        self.SIM_TIME = 1460
        if self.type1 == 'DSIM':
            if self.NAME_EXP == "SynLoan":
                self.ARRIVALS = pd.read_csv(
                    'DSIM/' + self.NAME_EXP + '/gen_synthetic_log_2000_11_5_' + str(iterations + 1) + '.csv')
                self.START_SIMULATION = datetime.strptime(
                    self.ARRIVALS.loc[0].at["start:timestamp"], '%Y-%m-%d %H:%M:%S')
            else:
                self.ARRIVALS = pd.read_csv(
                    'DSIM/' + self.NAME_EXP + '/gen_' + self.NAME_EXP + '_' + str(iterations + 1) + '.csv')
                self.START_SIMULATION = datetime.strptime(
                    self.ARRIVALS.loc[0].at["start:timestamp"], '%Y-%m-%d %H:%M:%S')
        elif self.type1 == 'DDPS':
            self.ARRIVALS = pd.read_csv(
                'DDPS_models_data/' + self.NAME_EXP + '/simulation' + str(iterations) + '_' + self.NAME_EXP + '.csv')
            self.START_SIMULATION = datetime.strptime(self.ARRIVALS.loc[0].at["start_timestamp"],
                                                      '%Y-%m-%d %H:%M:%S.%f')
        elif self.type1 == 'LSTM':
            if self.NAME_EXP == "SynLoan":
                self.ARRIVALS = pd.read_csv(
                    'LSTM_model_data/' + self.NAME_EXP + '/gen_synthetic_log_2000_11_5_' + str(iterations + 1) + '.csv')
                self.START_SIMULATION = datetime.strptime(
                    self.ARRIVALS.loc[0].at["start_timestamp"], '%Y-%m-%d %H:%M:%S')

            elif self.NAME_EXP == "ConsultaDataMining201618" or self.NAME_EXP == "PurchasingExample" or self.NAME_EXP == "BPI_Challenge_2017_W_Two_TS" or self.NAME_EXP == "Productions":
                self.ARRIVALS = pd.read_csv(
                    'LSTM_model_data/' + self.NAME_EXP + '/gen_' + self.NAME_EXP + '_' + str(iterations + 1) + '.csv')
                self.START_SIMULATION = datetime.strptime(self.ARRIVALS.loc[0].at["start_timestamp"],
                                                          '%Y-%m-%d %H:%M:%S')
            else:
                self.ARRIVALS = pd.read_csv(
                    'LSTM_model_data/' + self.NAME_EXP + '/gen_' + self.NAME_EXP + '_' + str(iterations + 1) + '.csv')
                self.START_SIMULATION = datetime.strptime(self.ARRIVALS.loc[0].at["start_timestamp"],
                                                          '%Y-%m-%d %H:%M:%S.%f')
        else:
            self.ARRIVALS = pd.read_csv(
                'RIMS/' + self.NAME_EXP + '/arrivals/iarr' + str(iterations) + '.csv', sep=',')
            self.START_SIMULATION = datetime.strptime(
                self.ARRIVALS.loc[0].at["timestamp"], '%Y-%m-%d %H:%M:%S')

        self.N_TRACE = len(self.ARRIVALS)

        if self.type1 == 'rims_plus':
            self.METADATA = 'RIMS/' + self.NAME_EXP + '/' + type1 + \
                '/' + self.NAME_EXP + self.prefix[0] + '_meta.json'
            self.SCALER = 'RIMS/' + self.NAME_EXP + '/' + type1 + \
                '/' + self.NAME_EXP + self.prefix[0] + '_scaler.pkl'
            self.INTER_SCALER = 'RIMS/' + self.NAME_EXP + '/' + type1 + \
                '/' + self.NAME_EXP + self.prefix[0] + '_inter_scaler.pkl'
            self.END_INTER_SCALER = 'RIMS/' + self.NAME_EXP + '/' + type1 + '/' + self.NAME_EXP + self.prefix[
                0] + '_end_inter_scaler.pkl'
        else:
            self.METADATA = 'RIMS/' + self.NAME_EXP + '/rims/' + \
                self.NAME_EXP + self.prefix[0] + '_meta.json'
            self.SCALER = 'RIMS/' + self.NAME_EXP + '/rims/' + \
                self.NAME_EXP + self.prefix[0] + '_scaler.pkl'
            self.INTER_SCALER = 'RIMS/' + self.NAME_EXP + '/rims/' + \
                self.NAME_EXP + self.prefix[0] + '_inter_scaler.pkl'
            self.END_INTER_SCALER = 'RIMS/' + self.NAME_EXP + '/rims/' + self.NAME_EXP + self.prefix[
                0] + '_end_inter_scaler.pkl'

        self.ROLE_CAPACITY = {}
        self.INDEX_AC = {}
        self.read_metadata_file()

        if self.type1 == 'DDPS' and not self.ROLE_CAPACITY:
            # Define default role capacity for DDPS if not available
            self.ROLE_CAPACITY = {'SYSTEM': [
                1000, {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]}

    def read_metadata_file(self):
        if os.path.exists(self.METADATA):
            with open(self.METADATA) as file:
                data = json.load(file)
                self.INDEX_AC = data['ac_index']
                self.AC_WIP_INITIAL = data['inter_mean_states']['tasks']
                self.PR_WIP_INITIAL = round(data['inter_mean_states']['wip'])

                roles_table = data['roles_table']
                self.ROLE_ACTIVITY = dict()
                for elem in roles_table:
                    self.ROLE_ACTIVITY[elem['task']] = elem['role']

                self.INDEX_ROLE = {'SYSTEM': 0}
                self.ROLE_CAPACITY = {'SYSTEM': [
                    1000, {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]}
                roles = data['roles']
                for idx, key in enumerate(roles):
                    self.INDEX_ROLE[key] = idx
                    self.ROLE_CAPACITY[key] = [len(roles[key]),
                                               {'days': [0, 1, 2, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]

    def get_predefined_waiting_time(self, cid, transition, time):
        """
        Return the predefined waiting time for DDPS based on the event log (from XES).
        """
        waiting_time_row = self.event_log_df[(self.event_log_df['case:concept:name'] == cid) &
                                             (self.event_log_df['concept:name'] == transition)]
        if len(waiting_time_row) > 0:
            waiting_time = (
                waiting_time_row['time:timestamp'].values[0] - time).total_seconds()
            return waiting_time
        return 0  # Default to 0 if no data available

    def get_predefined_processing_time(self, cid, transition, time):
        """
        Return the predefined processing time for DDPS based on the event log (from XES).
        """
        processing_time_row = self.event_log_df[(self.event_log_df['case:concept:name'] == cid) &
                                                (self.event_log_df['concept:name'] == transition)]
        if len(processing_time_row) > 0:
            start_time = processing_time_row['time:timestamp'].values[0]
            end_time = processing_time_row['time:timestamp'].values[-1]
            processing_time = (end_time - start_time).total_seconds()
            return processing_time
        return 0  # Default to 0 if no data available
