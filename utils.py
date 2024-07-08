import os
import json
import time
import torch
import openai
import random
import dataset
import numpy as np

from tqdm import tqdm
from collections import defaultdict


def check_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def balanced_sample(data, n_shot=1):
    label_dict = defaultdict(list)
    for i in range(len(data)):
        item = data.__getitem__(i, include_output=True)
        label = item["output"]
        label_dict[label].append(item)
    balanced_set = []
    for label in label_dict:
        balanced_set += random.sample(label_dict[label], n_shot)
    random.shuffle(balanced_set)
    return balanced_set


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def profile_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Function: {func.__name__}, Elapsed time: {elapsed_time} seconds")
        return result

    return wrapper


def get_context_limit(model):
    config = model.config
    if hasattr(config, "n_ctx"):  # gpt2
        context_size = config.n_ctx
    elif hasattr(config, "max_position_embeddings"):  # llama
        context_size = config.max_position_embeddings
    elif hasattr(config, "n_positions"):  # bloom
        context_size = config.n_positions
    else:
        context_size = 1024  # use 1024 as default value if none of the above is found
    context_size -= 1  # account for special tokens
    return context_size


def compute_word_overlap_score(prediction, label):
    length_prediction = 1.0 * len(set(prediction.split()))
    word_overlap_ratio = len(
        set(prediction.split()).intersection(set(label.split()))
    ) / max(length_prediction, 1e-5)
    if word_overlap_ratio > 0:
        return word_overlap_ratio
    else:
        return 0.0

def compute_accuracy(predictions, labels):
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
    return correct / len(labels)

def load_cache(cache_dir="openai_cache.json"):
    if os.path.exists(cache_dir):
        # with open(cache_dir, "r") as f:
        cache = json.load(open(cache_dir, "r"))
    else:
        cache = {}
    return cache


def save_cache(cache, cache_dir="openai_cache.json"):
    json.dump(cache, open(cache_dir, "w"))


def openai_batch_generator(prompts, model):
    use_azure = True
    if use_azure:
        openai.api_type = "azure"
        openai.api_base = "https://uclnlp.openai.azure.com/"
        openai.api_version = "2023-07-01-preview"
        openai.api_key = os.environ['MSFT_OAI_KEY']
    else:
        openai.api_key = os.environ['OPENAI_KEY']

    model_str = model.ckpt.replace("openai_", "")
    assert model_str == "gpt-35-turbo-0613", "only support gpt-35-turbo-0613 for now"

    cache = load_cache()
    if model_str not in cache:
        cache[model_str] = {}

    for prompt in tqdm(prompts):
        if prompt not in cache[model_str]:
            context = [elem for elem in prompt.split("\n\n")[:-1]]
            test_example = prompt.split("\n\n")[-1]
            labels = set([elem.split(" ")[-1] for elem in context])
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people find information. "
                    f"classify text into {labels}",
                }
            ]

            for elem in context:
                messages.append(
                    {"role": "user", "content": " ".join(elem.split(" ")[:-1])}
                )
                messages.append({"role": "assistant", "content": elem.split(" ")[-1]})
            messages.append({"role": "user", "content": test_example})

            try:
                response = openai.ChatCompletion.create(
                    engine=model_str,
                    messages=messages,  # [{"role": "system", "content": prompt}],
                    temperature=0,
                    max_tokens=4,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["\n"],
                    request_timeout=30,
                )

                assert len(response.choices) == 1, "should only have one completion"
                cache[model_str][prompt] = response.choices[0]
            except openai.InvalidRequestError as e:
                print(e)
                cache[model_str][prompt] = {
                    "finish_reason": "censored",
                    "message": {"content": ""},
                }
            except openai.error.Timeout as e:
                print(e)
                cache[model_str][prompt] = {
                    "finish_reason": "timeout",
                    "message": {"content": ""},
                }

    predictions, log_probs, dists = [], [], []
    _censored_count = 0
    for prompt in prompts:
        # azure openai service censors the output occasionally
        if cache[model_str][prompt]["finish_reason"] == "stop":
            prediction = cache[model_str][prompt]["message"]["content"]
        else:
            prediction = ""
            _censored_count += 1
            print("censored", _censored_count)
        log_prob = 1e-9
        dist = np.random.randn(5)
        predictions.append(prediction)
        log_probs.append(log_prob)
        dists.append(dist)

    save_cache(cache)
    return predictions, log_probs, torch.tensor(np.array(dists))


def load_data(dataset_name, path, template="{input_text} {separator} {output_text}"):
    dataset_classes = {
        "sst2": dataset.SST2Dataset,
        "sst5": dataset.SST5Dataset,
        "dbpedia": dataset.DBPediaDataset,
        "mr": dataset.MRDataset,
        "cr": dataset.CRDataset,
        "mpqa": dataset.MPQADataset,
        "subj": dataset.SubjDataset,
        "trec": dataset.TRECDataset,
        "agnews": dataset.AGNewsDataset,
        "rte": dataset.RTEDataset,
        "cb": dataset.CBDataset,
    }

    if dataset_name in dataset_classes:
        return dataset_classes[dataset_name](path=path, prompt_template=template)
    else:
        raise NotImplementedError(f"{dataset_name} not implemented")
