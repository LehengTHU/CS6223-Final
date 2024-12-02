import torch
import json
from termcolor import colored, cprint
import pandas as pd
import chromadb
import time
import argparse
from pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_mutation', type=int, default=3,
                        help='numbers for mutation.')
    parser.add_argument('--n_generation', type=int, default=3,
                        help='numbers for generation.')
    parser.add_argument('--HB_name', type=str, default='SR-small',
                            choices=['MI', 'XSTest', 'SR-small'],)
    parser.add_argument('--target_lm', type=str, default='gpt-4o-mini',
                        help='target language model.')
    parser.add_argument('--mode', type=str, default='basic',
                        help='the mode to run the pipeline.',
                        choices=['basic', 'fuzzing'])
    parser.add_argument('--use_memory', action='store_true', 
                        help="use summary from previous turns.")
    args, _ = parser.parse_known_args()
    return args, _


args, _ = parse_args()
print(args)


pipeline = Pipeline(args)

if args.mode == 'basic':
    pipeline.run_basic_dataset()

if args.mode == 'fuzzing':
    pipeline.run()


