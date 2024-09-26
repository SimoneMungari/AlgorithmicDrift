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
    module = "evaluation"
    proportions = ""
    strategy = "No_strategy"
    topk = 10
    user_count_start = 0
    user_count_end = 500  # num_users
    gpu_id = "0"
    c = 1.0
    gamma = 1.0
    eta_random = 0.01

    if len(argv) > 1:

        if argv[2].startswith("SyntheticDataset"):
            _, path, folder, model, module, proportions, strategy, user_count_start, user_count_end, gpu_id, c, gamma, eta_random = argv

            topk = int(topk)
            user_count_start = int(user_count_start)
            user_count_end = int(user_count_end)

            # Remember to check paths
            print(
                path,
                folder,
                model,
                module,
                proportions,
                strategy,
                topk,
                user_count_start,
                user_count_end,
                gpu_id, c, gamma, eta_random)

    keys = [
        "path",
        "folder",
        "model",
        "module",
        "proportions",
        "strategy",
        "topk",
        "user_count_start",
        "user_count_end",
        "gpu_id",
        "c",
        "gamma", "eta_random"]
    values = [path, folder, model, module, proportions, strategy, topk,
              user_count_start, user_count_end, gpu_id, c, gamma, eta_random]

    args = dict(zip(keys, values))

    return args
