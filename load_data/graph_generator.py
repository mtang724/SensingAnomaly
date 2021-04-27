import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx
import os
import utils

file_path = "D:\Clean\\"
file_list = []
for file in os.listdir(file_path):
    if file.endswith(".csv") and file != "Smartwatch_GravityDatum.csv" and file != "Smartwatch_LinearAccelerationDatum.csv":
        file_list.append(file)

df_list = []
for file in file_list:
    try:
        df = utils.read_data(file_path + file)
        filter_df = utils.filter_date(df, "45eb179e1d93ae0d", "2021-01-13", "2021-01-14").drop(columns=['ParticipantId', 'FileCreationTime', 'DeviceId'])
        df_list.append(filter_df)
    except:
        continue

feature_key_list = []
key_array_list = []
feature_array_list = []
label_list = []
for filtered_df in df_list:
    # print(day_df.head())
    label = filtered_df.columns[0]
    feature_keys, feature_list = utils.divide_day(filtered_df, label)
    key_array = np.array(feature_keys, dtype=object)
    feature_array = np.array(feature_list, dtype=object)
    if len(feature_keys) != 0:
        feature_key_list.append(feature_keys)
        if label in label_list:
            label = label + "1"
        label_list.append(label)
        key_array_list.append(key_array)
        feature_array_list.append(feature_array)

prev_key = feature_key_list[0]
for i in range(1, len(feature_key_list)):
    key = feature_key_list[i]
    prev_key = set(prev_key).intersection(key)
    print(len(prev_key))

sensor_feature_list = {}
for label, key_array, feature_array in zip(label_list, key_array_list, feature_array_list):
    indices = [np.where(key_array == x) for x in prev_key]
    feature = np.squeeze(feature_array[indices]).tolist()
    sensor_feature_list[label] = feature

graph_list = []
for i in range(len(prev_key)):
    nodes = label_list
    edges = combinations(nodes, 2)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    attrs = {}
    for key in sensor_feature_list:
        attrs[key] = {"attr": sensor_feature_list[key][i]}
    nx.set_node_attributes(g, attrs)
    graph_list.append(g)
    nx.write_gpickle(g, "../graphs/user2_{}.gpickle".format(i))