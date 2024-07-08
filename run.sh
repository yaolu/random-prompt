#!/usr/bin/env bash

python3 main.py --model gpt2 --num_random_draw 160 --context_shot_size 1 --corpus_size 64 --optimization_mode random_vocab --seed 1

python3 main.py --model gpt2-large --num_random_draw 160 --context_shot_size 1 --corpus_size 64 --optimization_mode random_wo_context --seed 1

