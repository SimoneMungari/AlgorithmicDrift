import sys
import os

# add 2.0-RecModules folder in sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import logging
import copy
import torch
from recbole.trainer.trainer import Trainer, RecVAETrainer
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
import pandas as pd
from os.path import exists


from recbole.data.utils import *

from models.recvae import RecVAE
from models.lightgcn import LightGCN
from models.ngcf import NGCF

def get_parameter_dict(args, model_checkpoint_folder):
    model = args["model"]
    path = args["path"]
    folder = args["folder"]

    parameter_dict = {
        "eval_step": 25,
        "topk": [10, 20],
        "metrics": ["Recall", "nDCG", "Hit"],
        "valid_metric": "Recall@10",
        "load_col": {"inter": ["user_id", "item_id", "rating"]},
        "data_path": os.path.join(path, folder, "recbole"),
        "checkpoint_dir": model_checkpoint_folder,
        "epochs": 100,
        "gpu_id": args['gpu_id']
    }

    if model == "RecVAE":
        parameter_dict["neg_sampling"] = None
        parameter_dict["hidden_dimension"] = 512
        parameter_dict["latent_dimension"] = 512
        parameter_dict["epochs"] = 100
    elif model == "LightGCN" or model == "NGCF":
        parameter_dict["neg_sampling"] = {"uniform": 1}
        parameter_dict["embedding_size"] = 1024
        parameter_dict["epochs"] = 200

    return parameter_dict


def convert_dataset_to_dataframe(dataset, utils_dicts, non_rad_users, semi_rad_users):
    videos_labels_dict, videos_slants_dict, _, reverse_videos_dict = utils_dicts

    new_dataset = dataset.inter_matrix()
    df = pd.DataFrame({'User': new_dataset.row, 'Item': new_dataset.col})[['User', 'Item']]
    df['Item'] = df['Item'] - 1
    df['User'] = df['User'] - 1

    df = df.astype({'User': int, 'Item': int})

    df = df.assign(Label=np.zeros(len(df['Item'])))
    df = df.assign(Orientation=np.zeros(len(df['Item'])))
    df = df.assign(Slant=np.zeros(len(df['Item'])))

    df.reset_index()
    df = df.sort_values('User')

    orientations = []
    labels = []
    slants = []

    for index, row in df.iterrows():
        item = int(row['Item'])
        labels.append(videos_labels_dict[item])
        slants.append(videos_slants_dict[reverse_videos_dict[item]])

        user = int(row['User'])
        orientation = 'radicalized'
        if user in non_rad_users:
            orientation = 'non radicalized'
        elif user in semi_rad_users:
            orientation = 'semi-radicalized'
        orientations.append(orientation)

    df['Orientation'] = orientations
    df['Slant'] = slants
    df['Label'] = labels

    return df


def get_model_structure_and_trainer(
        config,
        args,
        num_users=100,
        num_items=0,
        history_dataset=None):
    dataset = create_dataset(config)
    # logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_name = args["model"]

    if model_name == "RecVAE":

        model = RecVAE(
            config=config,
            dataset=train_data.dataset,
            num_users=num_users,
            num_items=num_items).to(
            config["device"])

        trainer = RecVAETrainer(config, model)
    elif model_name == "LightGCN" or model_name == "NGCF":

        if model_name == "LightGCN":
            model = LightGCN(config,
                             train_data.dataset,
                             num_users=num_users,
                             num_items=num_items).to(config["device"])
        elif model_name == "NGCF":
            model = NGCF(config, train_data.dataset).to(config["device"])

        trainer = Trainer(config, model)

    return model, trainer, (train_data, valid_data, test_data), history_dataset


def load_model(
        model_checkpoint_folder,
        config,
        args,
        num_users=100,
        num_items=0,
        history_dataset=None):
    model, trainer, data, history_dataset = get_model_structure_and_trainer(
        config=config, args=args,
        history_dataset=history_dataset,
        num_users=num_users, num_items=num_items)

    model_files = os.listdir(model_checkpoint_folder)
    checkpoint_file = model_files[-1]

    checkpoint_path = model_checkpoint_folder + checkpoint_file

    print(checkpoint_path)

    if torch.cuda.is_available() and args["gpu_id"] != "cpu":
        map_location = torch.device("cuda:{}".format(config.gpu_id))
    else:
        map_location = torch.device("cpu")

    # Here you can replace it by your model path.
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])

    return model, data, history_dataset, checkpoint_path, checkpoint_file, trainer


def train_model(
        args,
        model_checkpoint_folder=None,
        config=None,
        num_users=100,
        num_items=0):

    if exists(model_checkpoint_folder):
        files_to_delete = os.listdir(model_checkpoint_folder)
        for f in files_to_delete:
            if os.path.isfile(model_checkpoint_folder + f):
                os.remove(model_checkpoint_folder + f)
    else:
        os.makedirs(model_checkpoint_folder)

    model, trainer, data, _ = get_model_structure_and_trainer(
        config=config, args=args,
        num_users=num_users, num_items=num_items)

    train_data, valid_data, test_data = data

    _, score = trainer.fit(
        train_data, valid_data,
    )

    print("SCORE Validation", score)

    return model
