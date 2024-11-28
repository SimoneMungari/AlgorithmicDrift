from igraph import *
import networkx as nx
from os.path import join, exists
import os
import torch
import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from floweaver import *

from os import listdir
import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed
import scipy as sp
DPI = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_samples(samples_to_load=None, organic_strategy=False):


    if samples_to_load is None:
        print("No samples to load specified")
        return

    if organic_strategy:
        saving_path = "Organic/"
    else:
        saving_path = "No_strategy/"

    samples = samples_to_load

    samples_dfs = []

    for sample in samples:
        sample_path = path + folder + sample

        sample_path += "/Histories_" + sample.lower() + ".tsv"

        print(sample_path)

        sample_df = pd.read_csv(sample_path, sep="\t")
        samples_dfs.append((sample_df, sample))

    return samples_dfs, samples, saving_path


def load_graphs(graphs_path):
    graphs = []

    files = [f for f in listdir(graphs_path) if f.endswith(".tsv")]

    files = sorted(files)

    nodes_files = [f for f in files if f.endswith("_node.tsv")]
    edges_files = [f for f in files if f.endswith("_edge.tsv")]

    print("Loading", len(nodes_files), "graphs from", graphs_path, "...")

    if len(nodes_files) != len(edges_files):
        print("ERROR: nodes and edges files have not the same length")
        return

    for i in range(len(nodes_files)):

        graph_user = int(nodes_files[i].split("_")[-2])

        edge_path = graphs_path + edges_files[i]
        node_path = graphs_path + nodes_files[i]

        data_edge = pd.read_csv(edge_path, sep='\t')
        data_node = pd.read_csv(node_path, sep='\t')

        G = Graph.DataFrame(data_edge)

        data_node = data_node.reset_index()  # make sure indexes pair with number of rows
        data_node = data_node.sort_values(by=['Label'])

        count = 0

        for index, row in data_node.iterrows():
            G.vs.find(count)['label'] = row['Label']
            G.vs.find(count)['category'] = row['Category'].lower()

            # G.vs.find(count)['slant'] = row['Slant'].lower()

            count += 1

        graphs.append((G, graph_user))

        if i % 50 == 0:
            print("Loaded", i, "graphs")
            #if i == 400:
            #    break

        graphs.sort(key=lambda y: y[1])


    return graphs


def load_sample_metric(sample, model_name, metric="evaluation"):
    print("Processing sample", sample)

    if model_name == "Organic":
        return []

    sample_path = path + folder + sample

    sample_path += "/No_strategy/"

    sample_path += model_name + "/"
    sample_path += metric + ".tsv"

    print(sample_path)

    metric_df = pd.read_csv(sample_path, sep="\t")

    return (metric_df, model_name)

def get_graphs_by_users_category(graphs, synthetic_df, users_category=None, users_category_to_exclude=None, kind_of_category='Orientation'):

    if users_category is not None:
        users = synthetic_df[synthetic_df[kind_of_category]
                           == users_category]["User"].unique()
    elif users_category_to_exclude is not None:
        users = synthetic_df[synthetic_df[kind_of_category]
                           != users_category_to_exclude]["User"].unique()
    graphs_by_category = [(g, user_g)
                          for g, user_g in graphs if user_g in users]

    return graphs_by_category

def load_samples_metrics(samples_list, models_to_load=[], metric="evaluation"):
    samples_metrics = []

    n_jobs = len(models_to_load)

    samples_metrics = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(load_sample_metric)(
            sample=samples_list[0], metric=metric, model_name=model
        )
        for model in models_to_load
    )

    return samples_metrics

def load_sample_graphs(sample, model_name, weights_c):
    print("Processing sample", sample)

    sample_path = path + folder + sample
    print(sample)

    if model_name != "Organic":
        sample_path += "/No_strategy/"
    else:
        sample_path += "/Organic/"

    if model_name != "Organic":
        graphs_folder = model_name + "/graphs/"
    else:
        graphs_folder = "a_0.4/graphs/"

    if model_name != "Organic":
        graphs_folder += "topk_10/"#"topk_10/"

    graphs_path = sample_path + graphs_folder

    if model_name != "Organic":
        graphs_path += weights_c

    graphs = load_graphs(graphs_path)

    return (graphs, model_name)


def load_all_samples_graphs(samples_list, weights_c, models_to_load=[]):
    samples_graphs = []

    n_jobs = len(models_to_load)

    samples_graphs = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(load_sample_graphs)(
            sample=samples_list[0], model_name=model, weights_c=weights_c
        )
        for model in models_to_load
    )

    return samples_graphs


def modify_graph(G, percentile=95):
    inner_graph = G.copy()

    degrees = list(inner_graph.degree())
    degree_max = np.percentile(degrees, percentile)

    hubs = {i for i in range(len(degrees)) if degrees[i] > degree_max}

    out_edges = []

    for node in hubs:
        out_edges = list(inner_graph.incident(node, mode='out')).copy()

    if len(out_edges) > 0:
        inner_graph.delete_edges(out_edges)

    return inner_graph


def compute_algorithmic_drift(G):
    G1 = modify_graph(G)

    H = [x.index for x in G1.vs.select(category='harmful')]
    N = [y.index for y in G1.vs.select(category='neutral')]

    if len(H) == 0:
        return -1.

    if len(N) == 0:
        return 1.

    A = G1.get_edgelist()
    G2 = nx.DiGraph(A)
    # otherwise there could be mismatch in the next sums
    G2.add_nodes_from([n.index for n in G1.vs])

    edge_attrs = dict(zip(A, [{'Weight': v} for v in G1.es["Weight"]]))
    nx.set_edge_attributes(G2, edge_attrs)

    nstart_H = {x: (1 if x in H else 0) for x in G2.nodes()}
    pr_H = nx.pagerank(G2, alpha=0.85, nstart=nstart_H,
                       personalization=nstart_H, weight="Weight")

    nstart_N = {x: (1 if x in N else 0) for x in G2.nodes()}
    pr_N = nx.pagerank(G2, alpha=0.85, nstart=nstart_N,
                       personalization=nstart_N, weight="Weight")

    P_HH = sum([pr_H[u] for u in H])
    P_NN = sum([pr_N[u] for u in N])
    P_HN = sum([pr_H[u] for u in N])
    P_NH = sum([pr_N[u] for u in H])

    RWC = P_HH * P_NH - P_NN * P_HN

    return RWC


def get_algorithmic_drifts(graphs, synthetic_df=None, users_category=None):
    users_graphs = graphs

    if users_category is not None:
        users_graphs = get_graphs_by_users_category(graphs, synthetic_df, users_category)

    rwcs = []

    for g, user_g in users_graphs:
        rwcs.append(compute_algorithmic_drift(g))

    return rwcs


def get_sparse_adj_martrix(edgelist, weights, N):
    adjacency_matrix = csr_matrix((weights, zip(*edgelist)), shape=(N, N), dtype=np.float32)

    return adjacency_matrix


def return_adj_matrix(graph):
    """
    from graph to adj-matrix
    """

    edgelist_idx = graph.get_edgelist()

    if "weight" not in graph.es.attributes():

        weights = [1] * len(edgelist_idx)

    else:

        weights = graph.es["weight"]

    N = graph.vcount()

    adj_matrix = get_sparse_adj_martrix(edgelist_idx, weights, N)

    # return adj_matrix
    adj_matrix = normalize(adj_matrix, norm='l1', axis=1)

    return adj_matrix


def compute_drift(graph, damping_factor=True, sparse=False):
    transient_nodes = graph.vs.select(category="harmful").indices

    # [1] - transient nodes
    adj_matrix = return_adj_matrix(graph)

    # [2] - A_tt matrix
    A_tt = adj_matrix[transient_nodes, :][:, transient_nodes]

    # [3] - Damping vector
    if damping_factor:
        val_damping = graph.vcount() / graph.vcount()

        A_tt = damping_factor * A_tt + (1 - damping_factor) * val_damping

    # [4] compute inverse-matrix  = (I - A_tt)^-1

    A_tt = A_tt * -1.

    A_tt = A_tt.todense()

    for ix in range(A_tt.shape[0]):
        A_tt[ix, ix] += 1.

    F = A_tt.getI()

    z_vector = F.sum(1)

    # [5] generate final-vector for transient-nodes
    z_vector = np.array([x[0] for x in z_vector.tolist()])

    return F, z_vector


def get_absorbing_rads(graphs, synthetic_df=None, users_category=None, nodes_category=None):
    users_graphs = graphs

    if users_category is not None:
        users_graphs = get_graphs_by_users_category(graphs, synthetic_df, users_category)

    absorbing_rads = []

    for g, user_g in users_graphs:
        absorbing_rads.append(np.sum(compute_drift(g)[1]))

    return absorbing_rads


def get_harmful_rate_before_and_after_recs(graphs, synthetic_df, users_category=None, kind_of_category='Orientation'):
    users_graphs = graphs
    users = synthetic_df.groupby("User")
    videos = synthetic_df["Video"]
    labels = synthetic_df["Label"]
    labels_dict = dict(zip(videos, labels))

    if users_category is not None:
        users_graphs = get_graphs_by_users_category(
            users_graphs, synthetic_df, users_category=users_category, kind_of_category=kind_of_category)

    harmful_rate_before_recs = []
    harmful_rate_after_recs = []

    for graph, user_g in users_graphs:
        graph_harmful_nodes = [labels_dict[x] for x in graph.vs["label"] if labels_dict[x] == "harmful"]
        harmful_rate_after_recs.append(len(graph_harmful_nodes) / graph.vcount())

        temp_df = users.get_group(user_g)
        harmful_user_rate_before_recs = temp_df[temp_df["Label"] == "harmful"]["Label"]
        harmful_rate_before_recs.append(len(harmful_user_rate_before_recs) / len(temp_df))

    return np.array(harmful_rate_before_recs), np.array(harmful_rate_after_recs)


def get_harmful_rate_before_and_after_recs(graphs, synthetic_df, users_category=None):
    users_graphs = graphs
    users = synthetic_df.groupby("User")
    videos = synthetic_df["Video"]
    labels = synthetic_df["Label"]
    labels_dict = dict(zip(videos, labels))

    if users_category is not None:
        users_graphs = get_graphs_by_users_category(
            users_graphs, synthetic_df, users_category=users_category)

    harmful_rate_before_recs = []
    harmful_rate_after_recs = []

    for graph, user_g in users_graphs:
        graph_harmful_nodes = [labels_dict[x] for x in graph.vs["label"] if labels_dict[x] == "harmful"]
        harmful_rate_after_recs.append(len(graph_harmful_nodes) / graph.vcount())

        temp_df = users.get_group(user_g)
        harmful_user_rate_before_recs = temp_df[temp_df["Label"] == "harmful"]["Label"]
        harmful_rate_before_recs.append(len(harmful_user_rate_before_recs) / len(temp_df))

    return np.array(harmful_rate_before_recs), np.array(harmful_rate_after_recs)


def get_delta_harmful_rates(graphs, synthetic_df, users_category=None):
    harmful_before_recs, harmful_after_recs = get_harmful_rate_before_and_after_recs(
        graphs, synthetic_df, users_category=users_category)

    delta_harmful = harmful_after_recs - harmful_before_recs

    return np.array(delta_harmful * 100)


def save_model_statistics_for_dtc_and_ads(model, sample, df, graphs, weights_c):

    statistics_path = "../../Statistics/" + sample + "/"

    if model != "Organic":
        statistics_path += weights_c

    if not exists(statistics_path):
        os.makedirs(statistics_path)

    statistics_fn = "statistics_" + model + ".tsv"

    print("Generating {}...".format(statistics_path + statistics_fn))

    f = open(statistics_path + statistics_fn, "w")
    f.write("UserID\tUserType\tDTC\tADS\n")

    non_rad_graphs = get_graphs_by_users_category(samples_graphs, samples_dfs[0][0], "non radicalized")
    semi_rad_graphs = get_graphs_by_users_category(graphs, df, "semi-radicalized")
    rad_graphs = get_graphs_by_users_category(graphs, df, "radicalized")

    for i, (g, user_id) in enumerate(graphs):
        user_type = "NR"
        if (g, user_id) in semi_rad_graphs:
            user_type = "SR"
        elif (g, user_id) in rad_graphs:
            user_type = "R"

        dtc = get_delta_harmful_rates([graphs[i]], df)

        ads = compute_algorithmic_drift(g)

        f.write(str(user_id) + "\t" + str(user_type) + "\t" + str(round(dtc[0], 2)) +
                "\t" + str(round(ads, 2)) + "\n")

    f.close()


def save_parallel_dhc_and_ads(sample, df, samples_graphs, models_to_load):
    n_jobs = len(models_to_load)

    Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(save_model_statistics_for_dtc_and_ads)(
            model, sample, df, samples_graphs[i][0]
        )
        for i, model in enumerate(models_to_load)
    )

path = '../../data/processed/'
folder = 'SyntheticDataset/History/'

orientations = ["non radicalized", "semi-radicalized", "radicalized"]

weights_c_list = ["0.75_0.25_gamma1_0.5_sigmagamma1_0.01_gamma2_0.99_sigmagamma2_0.01_gamma3_0.75_sigmagamma3_0.01_eta_0.0/"]
models_to_load = ["RecVAE", "Organic"]

datasets_for_statistics = ["0.2_0.6_0.2"]

samples_dfs, samples, saving_path = get_samples(datasets_for_statistics)
print(samples)
print(models_to_load)

for dataset_for_stat in datasets_for_statistics:
    for weights_c in weights_c_list:
        dataset_for_stat = dataset_for_stat.replace(", ", "_")
        d = dataset_for_stat
        print(f"Generating statistics for {d}")
        samples_dfs, samples, saving_path = get_samples([d])


        samples_graphs = load_all_samples_graphs(samples_list=samples, models_to_load=models_to_load,
                                                 weights_c=weights_c)

        for i, model in enumerate(models_to_load):
            if model == "Organic":
                save_model_statistics_for_dtc_and_ads(model, samples_dfs[0][1], samples_dfs[0][0], samples_graphs[i][0], weights_c)
            #
            else:
                save_model_statistics_for_dtc_and_ads(model, samples_dfs[0][1], samples_dfs[0][0], samples_graphs[i][0], weights_c)

