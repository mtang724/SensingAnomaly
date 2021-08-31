import torch
import networkx as nx
import torch.nn as nn
import os
from sklearn import model_selection

def read_graph_data():
    directory = "../graphs/"
    prev_user = 1
    subject_list = []
    feature_list = []
    feature_lists = []
    subject_set = set()
    # list_name.sort(key= lambda x: float(x.strip('something')))
    filenames = [file for file in os.listdir(directory)]
    filenames.sort(key=lambda x: (float(x.split("_")[0][-1]),float(x.split("_")[1].split(".")[0])))
    start_index = 0
    end_index = 0
    for filename in filenames:
        path = os.path.join(directory, filename)
        user_no = int(filename.split("_")[0][-1])
        feature_no = int(filename.split("_")[1].split(".")[0])
        subject_set.add(user_no)
        if user_no != prev_user:
            # [人的编号，开始位置，结束位置，共多少个数据点]
            subject = [prev_user, start_index, end_index, (end_index-start_index) + 1]
            subject_list.append(subject)
            # start_index = end_index + 1
            start_index = 0
            feature_lists.append(feature_list)
            feature_list = []
        G = nx.read_gpickle(path)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        outputs = []
        for node in G.nodes:
            if node == "Type":
                continue
            input = G.nodes[node]['attr']
            input_len = len(input)
            if input_len < 19:
                input += [0] * (19 - input_len) + [len(input)]
            else:
                input = input[:19] + [len(input)]
            input_list = []
            for value in input:
                if value == "Small drink":
                    value = 20
                input_list.append(value)
            input = torch.FloatTensor(input_list)
            mlp = nn.Linear(20, 10)
            output = mlp(input)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        feat = outputs
        prev_user = user_no
        end_index += 1
        feature_list.append(feat)
    subject = [user_no, start_index, end_index, (end_index-start_index) + 1]
    subject_list.append(subject)
    train_subject = torch.FloatTensor([subject_list[0]])
    test_subject = torch.FloatTensor([subject_list[1]])
    feature_lists.append(feature_list)
    train_feats = feature_lists[0]
    train_feats = torch.stack(train_feats, dim=0)
    test_feats = feature_lists[1]
    test_feats = torch.stack(test_feats, dim=0)
    print(train_feats.shape)
    print(test_feats.shape)
    train_labels = torch.zeros(train_feats.shape[0])
    test_labels = torch.zeros(test_feats.shape[0])
    abnormal_list = torch.zeros(train_feats.shape[0] + test_feats.shape[0], train_feats.shape[1] + test_feats.shape[1])
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(feats,
    #                                                     labels,
    #                                                     test_size=0.2,
    #                                                     random_state=42)
    # return X_train, y_train, X_test, y_test, subject_list
    return train_feats, train_labels, test_feats, test_labels, train_subject, test_subject, abnormal_list


# g = dgl.from_networkx(G)
read_graph_data()