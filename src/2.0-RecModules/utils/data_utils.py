import os
from os.path import exists


def get_dataset_name_and_paths(args):
    path = args["path"]
    folder = args["folder"]
    strategy = args["strategy"]
    proportions = args["proportions"]

    dataset_path = ""
    dataset_name = ""

    if folder.startswith("SyntheticDataset"):

        dataset_name = "Histories_" + proportions
        dataset_path = path + folder + proportions + "/"
        saving_path = dataset_path
        saving_path += "{}/".format(strategy)

    elif folder.startswith("movielens-1m"):

        dataset_name = "Histories"
        dataset_path = path + folder + "/"
        saving_path = dataset_path

    return dataset_path, dataset_name, saving_path


def create_folders(dataset_path, args):
    model = args["model"]
    topk = args["topk"]

    base_path = dataset_path + model + "/"

    model_checkpoint_folder = base_path + "model_checkpoint/"

    graphs_folder = base_path + "graphs/topk_" + str(topk) + "/"

    sessions_folder = base_path + "sessions/"

    if not exists(model_checkpoint_folder):
        os.makedirs(model_checkpoint_folder)

    if not exists(graphs_folder):
        os.makedirs(graphs_folder)

    if not exists(sessions_folder):
        os.makedirs(sessions_folder)


def get_parsed_args(argv):

    path = ""
    folder = ""
    model = "RecVAE"
    module = "generation"
    strategy = "No_strategy"
    SYNTHETIC = False
    name = ""

    proportions = ""
    topk = 10
    gpu_id = "0"
    c = 1.0
    gamma = [0.0, 0.0, 0.0]
    sigma = [0.0, 0.0, 0.0]
    eta_random = 0.0
    introduce_bias = True
    target = "Horror"
    influence_percentage = 0.3

    if len(argv) > 1:

        #if argv[2].startswith("SyntheticDataset"):
        _, path, folder, model, module, strategy, SYNTHETIC, proportions, name, gpu_id, c,\
            gamma, sigma, eta_random, introduce_bias, target, influence_percentage = argv

        topk = int(topk)
        gamma = [float(g) for g in gamma.split(",")]
        sigma = [float(s) for s in sigma.split(",")]

        SYNTHETIC = False if SYNTHETIC == "False" else True
        introduce_bias = True if introduce_bias == "True" else False
        influence_percentage = float(influence_percentage)

        # Remember to check paths
        print(
            path,
            folder,
            model,
            module,
            strategy,
            SYNTHETIC,
            proportions,
            name,
            topk,
            gpu_id, c, gamma, sigma, eta_random, introduce_bias, target, influence_percentage)

    keys = [
        "path",
        "folder",
        "model",
        "module",
        "strategy",
        "synthetic",
        "proportions",
        "name",
        "topk",
        "gpu_id",
        "c",
        "gamma", "sigma", "eta_random", "introduce_bias", "target", "influence_percentage"]
    values = [path, folder, model, module, strategy, SYNTHETIC, proportions, name, topk, gpu_id, c, gamma, sigma,
              eta_random, introduce_bias, target, influence_percentage]

    args = dict(zip(keys, values))

    return args
