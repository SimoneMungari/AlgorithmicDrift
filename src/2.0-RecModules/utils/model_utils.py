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
        "data_path": path + folder + "recbole",
        "checkpoint_dir": model_checkpoint_folder,
        "epochs": 100,
        "gpu_id": args['gpu_id']
    }

    if model == "RecVAE":
        parameter_dict["neg_sampling"] = None
        parameter_dict["hidden_dimension"] = 512
        parameter_dict["latent_dimension"] = 512
        parameter_dict["epochs"] = 100

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


def rewire_train(data, utils_dicts, non_rad_users, semi_rad_users, args, saving_path=None):
    rewiring_strategy = args["sub_strategy"]
    factor = args["factor"]

    discard_percentage = float("0." + factor.split("_")[1])

    train_data, valid_data, test_data = data

    train_df = convert_dataset_to_dataframe(train_data.dataset, utils_dicts,
                                            non_rad_users, semi_rad_users)

    valid_df = convert_dataset_to_dataframe(valid_data.dataset, utils_dicts,
                                            non_rad_users, semi_rad_users)

    test_df = convert_dataset_to_dataframe(test_data.dataset, utils_dicts,
                                           non_rad_users, semi_rad_users)

    df_for_videos_to_exclude = pd.concat((valid_df, test_df)).reset_index(drop=True)
    rewired_train_df = start_rewiring(train_df, df_for_videos_to_exclude, rewiring_strategy, discard_percentage)

    rewired_history_df = pd.concat((copy.deepcopy(rewired_train_df), valid_df, test_df)).reset_index(drop=True)

    if saving_path is not None:
        videos_labels_dict, videos_slants_dict, reverse_users_dict, reverse_videos_dict = utils_dicts

        rewired_fn = "Histories_eta_{}.tsv".format(args["eta"])
        rewired_path = saving_path + rewired_fn

        reversed_rewired_history_df = rewired_history_df.copy()

        reversed_rewired_history_df["User"] = [reverse_users_dict[x] for x in reversed_rewired_history_df["User"]]
        reversed_rewired_history_df["Item"] = [reverse_videos_dict[x] for x in reversed_rewired_history_df["Item"]]

        # if not exists(rewired_path):
        reversed_rewired_history_df.to_csv(rewired_path, header=["User", "Video", "Label", "Orientation", "Slant"],
                                           sep="\t", index=False)

    history_dataset = {}

    users_group = rewired_history_df.groupby("User")
    for user, _ in users_group:
        interactions = list(users_group.get_group(user)["Item"])
        history_dataset[user] = interactions

    rewired_train_df['User'] = rewired_train_df['User'] + 1
    rewired_train_df['Item'] = rewired_train_df['Item'] + 1

    rewired_train_df = rewired_train_df.assign(item_value=np.ones(len(rewired_train_df)))
    rewired_train_df = rewired_train_df.assign(user_id=rewired_train_df['User'])
    rewired_train_df = rewired_train_df.assign(item_id=rewired_train_df['Item'])

    rewired_train_df = rewired_train_df[['user_id', 'item_id', 'item_value']]

    interaction = Interaction(rewired_train_df)

    train_data.dataset = train_data.dataset.copy(interaction)
    return train_data, history_dataset


def get_model_structure_and_trainer(
        config,
        logger,
        args,
        utils_dicts=None,
        users=None,
        mean_slant_users=None,
        items_labels=None,
        non_rad_users=None,
        semi_rad_users=None,
        num_users=100,
        num_items=0,
        transient_nodes=None,
        reverse_videos_dict=None,
        history_dataset=None,
        saving_path=None):
    dataset = create_dataset(config)
    # logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_name = args["model"]

    if model_name == "RecVAE":

        model = RecVAE(
            config=config,
            dataset=train_data.dataset,
            mean_slant_users=mean_slant_users,
            items_labels=items_labels,
            num_users=num_users,
            num_items=num_items).to(
            config["device"])

        trainer = RecVAETrainer(config, model)

    return model, trainer, (train_data, valid_data, test_data), history_dataset


def load_model(
        model_checkpoint_folder,
        config,
        logger,
        args,
        utils_dicts=None,
        users=None,
        mean_slant_users=None,
        items_labels=None,
        non_rad_users=None,
        semi_rad_users=None,
        num_users=100,
        num_items=0,
        transient_nodes=None,
        reverse_videos_dict=None,
        history_dataset=None,
        saving_path=None):
    model, trainer, data, history_dataset = get_model_structure_and_trainer(
        config=config, logger=logger, args=args, utils_dicts=utils_dicts, users=users,
        mean_slant_users=mean_slant_users, items_labels=items_labels, history_dataset=history_dataset,
        num_users=num_users, num_items=num_items, non_rad_users=non_rad_users, semi_rad_users=semi_rad_users,
        transient_nodes=transient_nodes, reverse_videos_dict=reverse_videos_dict, saving_path=saving_path)


    model_files = os.listdir(model_checkpoint_folder)
    checkpoint_file = model_files[-1]

    checkpoint_path = model_checkpoint_folder + checkpoint_file

    print(checkpoint_path)

    if torch.cuda.is_available():
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
        interaction_dict=None,
        config=None,
        users=None,
        mean_slant_users=None,
        items_labels=None,
        logger=None,
        utils_dicts=None,
        non_rad_users=None,
        semi_rad_users=None,
        num_users=100,
        num_items=0,
        saving_path=None):

    if exists(model_checkpoint_folder):
        files_to_delete = os.listdir(model_checkpoint_folder)
        for f in files_to_delete:
            if os.path.isfile(model_checkpoint_folder + f):
                os.remove(model_checkpoint_folder + f)
    else:
        os.makedirs(model_checkpoint_folder)

    model, trainer, data, _ = get_model_structure_and_trainer(
        config=config, logger=logger, args=args, users=users, utils_dicts=utils_dicts,
        mean_slant_users=mean_slant_users, items_labels=items_labels,
        num_users=num_users, num_items=num_items, non_rad_users=non_rad_users, semi_rad_users=semi_rad_users,
        saving_path=saving_path)

    train_data, valid_data, test_data = data

    _, score = trainer.fit(
        train_data, valid_data,
    )

    print("SCORE Validation", score)

    return model
