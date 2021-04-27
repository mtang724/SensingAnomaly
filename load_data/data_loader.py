import pandas as pd
import numpy as np
import tsfel
from itertools import combinations
import networkx as nx


def read_data(path):
    df = pd.read_csv(path)
    return df


def divide_day(df, label):
    feature_list = []
    # print (pd.to_datetime(df.index))
    df['T'] = pd.to_datetime(df['T'])
    groupby_time_df = df.groupby(pd.Grouper(key='T', freq='10Min'))[label].apply(list)
    feature_list += groupby_time_df.values.tolist()
    return groupby_time_df.keys(), feature_list


def filter_date(df, deviceid, start_date, end_date):
    filtered_df = df.loc[df['DeviceId'] == deviceid]
    filtered_df = filtered_df[(filtered_df['T'] > start_date) & (filtered_df['T'] < end_date)]
    # filtered_df = filtered_df.set_index(pd.DatetimeIndex(pd.to_datetime(filtered_df['T'])))
    return filtered_df


if __name__ == '__main__':
    path = "D:\Clean\\"
    hr_path = path + "Smartwatch_HeartRateDatum.csv"
    hr_df = read_data(hr_path)
    filtered_df = filter_date(hr_df, "6f76c9b33efc0bdb", "2021-01-06", "2021-01-07")
    # print(day_df.head())
    HR_feature_keys, HR_feature_list = divide_day(filtered_df, "HR")
    HR_key_array = np.array(HR_feature_keys, dtype=object)
    HR_feature_array = np.array(HR_feature_list, dtype=object)


    sound_path = path + "Smartwatch_SoundDatum.csv"
    sound_df = read_data(sound_path)
    sound_filtered_df = filter_date(sound_df, "6f76c9b33efc0bdb", "2021-01-06", "2021-01-07")
    # print(day_df.head())
    sound_feature_keys, sound_feature_list = divide_day(sound_filtered_df, "Sound")
    sound_key_array = np.array(sound_feature_keys, dtype=object)
    sound_feature_array = np.array(sound_feature_list, dtype=object)


    step_path = path + "Smartwatch_StepCountDatum.csv"
    step_df = read_data(step_path)
    step_filtered_df = filter_date(step_df, "6f76c9b33efc0bdb", "2021-01-06", "2021-01-07")
    # print(day_df.head())
    step_feature_keys, step_feature_list = divide_day(step_filtered_df, "Count")
    step_key_array = np.array(step_feature_keys, dtype=object)
    step_feature_array = np.array(step_feature_list, dtype=object)

    gyro_path = path + "Smartwatch_GyroscopeDatum.csv"
    gyro_df = read_data(gyro_path)
    gyro_filtered_df = filter_date(gyro_df, "6f76c9b33efc0bdb", "2021-01-06", "2021-01-07")
    # print(day_df.head())
    gyro_feature_keys, gyro_feature_list = divide_day(gyro_filtered_df, "X")
    gyro_key_array = np.array(gyro_feature_keys, dtype=object)
    gyro_feature_array = np.array(gyro_feature_list, dtype=object)

    comp_path = path + "Smartwatch_CompassDatum.csv"
    comp_df = read_data(comp_path)
    comp_filtered_df = filter_date(comp_df, "6f76c9b33efc0bdb", "2021-01-06", "2021-01-07")
    # print(day_df.head())
    comp_feature_keys, comp_feature_list = divide_day(comp_filtered_df, "Orientation")
    comp_key_array = np.array(comp_feature_keys, dtype=object)
    comp_feature_array = np.array(comp_feature_list, dtype=object)


    all = set(HR_feature_keys).intersection(step_feature_keys).\
        intersection(sound_feature_keys).intersection(gyro_feature_keys).intersection(comp_feature_keys)
    indices_HR = [np.where(HR_key_array == x) for x in all]
    indices_sound = [np.where(sound_key_array == x) for x in all]
    indices_STEP = [np.where(step_key_array == x) for x in all]
    indices_gyro = [np.where(gyro_key_array == x) for x in all]
    indices_comp = [np.where(comp_key_array == x) for x in all]

    HR_feature = np.squeeze(HR_feature_array[indices_HR]).tolist()
    sound_feature = np.squeeze(sound_feature_array[indices_sound]).tolist()
    step_feature = np.squeeze(step_feature_array[indices_STEP]).tolist()
    gyro_feature = np.squeeze(gyro_feature_array[indices_gyro]).tolist()
    comp_feature = np.squeeze(comp_feature_array[indices_comp]).tolist()

    graph_list = []
    for i in range(len(all)):
        nodes = ['HR', 'sound', 'step', 'gyro']
        edges = combinations(nodes, 2)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        attrs = {'HR': {"attr": HR_feature[i]}, 'sound': {"attr": sound_feature[i]}, 'step': {"attr": step_feature[i]},
                 'gyro': {"attr": gyro_feature[i]}, 'comp': {"attr": comp_feature[i]}}
        nx.set_node_attributes(g, attrs)
        print(g.nodes['gyro']['attr'])
        graph_list.append(g)









