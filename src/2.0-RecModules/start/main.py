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



SYNTHETIC = "False"

if SYNTHETIC == "True":
    path = "../../data/processed/"
    folder = "SyntheticDataset/History/"
else:
    path = "../../data/processed/"
    folder = "movielens-1m/"

if not exists(path):
    os.makedirs(path)

if not exists(path + folder):
    os.makedirs(path + folder)

gpu_id = "cpu"

module = "generation"  # training, evaluation, generation
dataset = None

if len(sys.argv) >= 3:
    dataset = sys.argv[1]
    module = sys.argv[2]

# "No_strategy", "Organic"
strategy = "No_strategy"

if strategy == "Organic":
    module = "generation"

if SYNTHETIC == "False":
    proportions = ""
    name = "movielens-1m"
else:
    proportions = "0.05_0.9_0.05"
    name = ""

model = "RecVAE"
introduce_bias = "True"
target = "Horror"
influence_percentage = "0.3"

c = "0.75"

#gamma = "0.5,0.99,0.75"
gamma = "0.05,0.05,0.05,0.05,0.05,0.05,0.05"

# sigma = "0.01,0.01,0.01"
sigma = "0.01,0.01,0.01,0.01,0.01,0.01,0.01"

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
              strategy,
              SYNTHETIC,
              proportions,
              name,
              gpu_id,
              c, gamma, sigma, eta_random, introduce_bias, target, influence_percentage]

process_args_2 = ["python",
              program_to_call,
              path,
              folder,
              model,
              module,
              strategy,
              SYNTHETIC,
              proportions,
              name,
              gpu_id,
              c, gamma, sigma, eta_random, introduce_bias, target, influence_percentage]

handle_processes(process_args_1, 0)
handle_processes(process_args_2, 0)
