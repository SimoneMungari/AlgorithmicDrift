import sys

import torch.cuda
import numpy as np
import torch
import os
import time
from os.path import exists

import subprocess

def handle_processes(cmd, thread):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Started process {} with cmd = {}".format(thread, cmd))


    for line in p.stdout:
        line_str = line.decode("utf-8")
        print("Process{}".format(thread), line_str)

    p.wait()
    print(cmd, "Return code", p.returncode)

    f = open("log_processes/err_{}.txt".format(thread), "w")
    for line in p.stderr:
        f.write("{}\n".format(line.decode("utf-8")))
    f.close()


    print("Finished handle process {}".format(cmd[4]))

path = "../../data/processed/"
folder = "SyntheticDataset/History/"

if not exists(path):
    os.makedirs(path)

if not exists(path + folder):
    os.makedirs(path + folder)

gpu_id = "cpu"

user_count_start_args = "0"
user_count_end_args = "2000"

module = "training"  # training, evaluation, generation
dataset = None

if len(sys.argv) >= 3:
    dataset = sys.argv[1]
    module = sys.argv[2]

# "No_strategy", "Organic"
strategy = "No_strategy"

if strategy == "Organic":
    module = "generation"

proportions = "0.05_0.9_0.05"
model = "RecVAE"
c = "0.75"

gamma_non_rad = "0.5"
gamma_semi_rad = "0.99"
gamma_rad = "0.75"

sigma_gamma_non_rad = "0.01"
sigma_gamma_semi_rad = "0.01"
sigma_gamma_rad = "0.01"

eta_random = "0.0"

program_to_call = 'start/handle_modules.py'

if not exists('log_processes'):
    os.makedirs('log_processes')

process_args_1 = ["python",
              program_to_call,
              path,
              folder,
              model,
              'recbole_dataset',
              proportions,
              strategy,
              user_count_start_args,
              user_count_end_args,
              gpu_id,
              c, gamma_non_rad, gamma_semi_rad, gamma_rad, sigma_gamma_non_rad, sigma_gamma_semi_rad,
                  sigma_gamma_rad, eta_random]

process_args_2 = ["python",
              program_to_call,
              path,
              folder,
              model,
              module,
              proportions,
              strategy,
              user_count_start_args,
              user_count_end_args,
              gpu_id,
              c, gamma_non_rad, gamma_semi_rad, gamma_rad, sigma_gamma_non_rad, sigma_gamma_semi_rad,
                  sigma_gamma_rad, eta_random]

handle_processes(process_args_1, 0)
handle_processes(process_args_2, 0)
