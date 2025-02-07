import sys
import os

import numpy as np
import torch.cuda

# add 2.0-RecModules folder in sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from os.path import exists
import logging
import pandas as pd
from recbole.utils import init_logger, init_seed
from recbole.config import Config
from recbole.data.utils import *
from graph_generation import generate_graphs, generate_organic_graphs, create_T_tensor
from utils.load_data import get_dataset, create_dataset_recbole
from utils.model_utils import load_model, train_model, get_parameter_dict
from utils.data_utils import create_folders, get_dataset_name_and_paths, get_parsed_args


args = get_parsed_args(sys.argv)
print(args)

if torch.cuda.is_available() and args["gpu_id"] != "cpu":
    print("Here GPU")
    torch.cuda.set_device(int(args["gpu_id"]))
else:
    print("Not Here GPU")


SYNTHETIC = args["synthetic"]
introduce_bias = args["introduce_bias"]
target = args["target"]
influence_percentage = args["influence_percentage"]

dataset_path, dataset_name, saving_path = get_dataset_name_and_paths(args)

print(dataset_path + dataset_name, saving_path)

history_dataset, uir_dataset, utils_dicts = get_dataset(
    dataset_path + dataset_name + ".tsv")

videos_labels_dict, videos_slants_dict, reverse_users_dict, reverse_videos_dict, reverse_videos_labels_dict = utils_dicts

users = list(set([i[0] for i in uir_dataset]))

df = pd.DataFrame(
    uir_dataset,
    columns=[
        "User",
        "Item",
        "Rating",
        "Orientation",
        "Label"])

items = df["Item"].unique()

# items_labels = df[["Item", "Label"]].drop_duplicates()
# item_label_dict = dict(zip(items_labels["Item"], items_labels["Label"]))

# If synthetic: Non-Rad 0, Rad 1, Semi-Rad 2
# In general, alphabetic order
orientations = list(np.sort(df["Orientation"].unique()))

users_orientations = df[["User", "Orientation"]].drop_duplicates()
user_orientation_dict = dict(zip(users_orientations["User"], list(map(orientations.index, users_orientations["Orientation"]))))

num_interactions = len(uir_dataset)
num_users = len(users)
num_items = len(items)

# remove "orientation" and "label" columns
uir_dataset = list(np.array(uir_dataset)[:, :3])

print("Dataset interactions:", num_interactions)
print("Dataset users:", num_users)
print("Dataset items:", num_items)

if args["module"] == "recbole_dataset":
    create_dataset_recbole(args, uir_dataset)

    if args["strategy"] != "Organic":
        create_folders(saving_path, args)
        print("Recbole dataset created")
    exit(0)

if args["module"] == "training":
    create_folders(saving_path, args)

model_checkpoint_folder = saving_path + args["model"] + "/model_checkpoint/"

print("MODEL CHECKPOINT:", model_checkpoint_folder)

parameter_dict = get_parameter_dict(args, model_checkpoint_folder)

if SYNTHETIC:
    dataset = "{}".format(args["proportions"])
else:
    dataset = "{}".format(args["name"])

model = args["model"]

config = Config(
    model=model,
    dataset=dataset,
    config_dict=parameter_dict)

# SET CONFIG DEVICE
if config["gpu_id"] != "cpu":
    config["device"] = "cuda:{}".format(config['gpu_id'])
else:
    config["device"] = "cpu"

# init random seed
init_seed(config["seed"], config["reproducibility"])

# logger initialization
init_logger(config)
logger = getLogger()

# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

density = num_interactions / (num_users * num_items)
print("Dataset density:", np.around(density, decimals=3))

if args["module"] == "training":

    train_model(
        args=args,
        model_checkpoint_folder=model_checkpoint_folder,
        config=config,
        num_users=num_users,
        num_items=num_items)

elif args["module"] == "evaluation":

    print("\n Evaluation on test:")

    model, data, _, checkpoint_path, checkpoint_file, trainer = load_model(
        model_checkpoint_folder=model_checkpoint_folder,
        config=config,
        num_users=num_users, num_items=num_items,
        history_dataset=history_dataset,
        args=args)

    _, _, test_data = data

    result = trainer.evaluate(
        test_data,
        model_file=checkpoint_path,
        show_progress=False)

    print(result)

    evaluation_filename = "evaluation.tsv"
    evaluation_path = saving_path + args["model"] + "/"

    print(evaluation_path + evaluation_filename)

    df = pd.DataFrame([result], columns=result.keys())
    df.to_csv(evaluation_path + evaluation_filename, sep="\t", index=False)

elif args["module"] == "generation":
    logger.setLevel(logging.ERROR)  # just to avoid too many prints

    topk = args["topk"]
    c = float(args["c"])

    eta_random = float(args["eta_random"])

    graphs_folder = saving_path + args["model"] + "/graphs/topk_" + str(topk) + "/"

    print(graphs_folder)

    B = 50
    d = 100

    T_tensor = create_T_tensor(
        history_dataset,
        num_items=num_items)

    if args["strategy"] == "Organic":
        graphs_folder = saving_path + "graphs/"
        if not exists(graphs_folder):
            os.makedirs(graphs_folder)

        if SYNTHETIC:
            generate_organic_graphs(T_tensor, history_dataset, dataset_name, dataset_path, B, d,
                                    reverse_users_dict, reverse_videos_dict, reverse_videos_labels_dict,
                                    saving_path)
        # else:

    else:
        generate_graphs(
            T_tensor,
            history_dataset,
            dataset_name,
            B,
            d,
            topk=topk,
            num_items=num_items,
            num_users=num_users,
            users=users,
            user_orientation_dict=user_orientation_dict,
            reverse_users_dict=reverse_users_dict,
            reverse_videos_dict=reverse_videos_dict,
            model_checkpoint_folder=model_checkpoint_folder,
            config=config,
            item_label_dict=videos_labels_dict,
            reverse_videos_labels_dict=reverse_videos_labels_dict,
            graphs_folder=graphs_folder,
            args=args,
            dataset_path=dataset_path,
            c=c,
            gamma_list=args["gamma"],
            sigma_gamma_list=args["sigma"],
            eta=eta_random,
            introduce_bias=introduce_bias,
            target=target,
            influence_percentage=influence_percentage,
        )
