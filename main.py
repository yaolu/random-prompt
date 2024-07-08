import torch
import random
import argparse
import transformers

import utils

transformers.logging.set_verbosity_error()


def compute_accuracy(predictions, labels):
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
    return correct / len(labels)


def generate_random_vocab_separator(model, min_separator_length, max_separator_length):
    vocabulary_size = model.tokenizer.vocab_size
    separator_length = random.randint(min_separator_length, max_separator_length)
    separator_ids = random.sample(range(vocabulary_size), separator_length)
    separator_text = model.tokenizer.decode(separator_ids)
    return separator_text


def generate_random_wo_context_separator(model, min_separator_length, max_separator_length):
    separator_text = \
    model(text_inputs="", do_sample=True, max_new_tokens=random.randint(min_separator_length, max_separator_length))[0][
        'generated_text']
    return separator_text


def main(args):
    dataset_name = args.dataset
    model_name = args.model
    corpus_size = args.corpus_size
    context_shot_size = args.context_shot_size
    min_separator_length = args.min_separator_length
    max_separator_length = args.max_separator_length
    num_random_draw = args.num_random_draw
    optimization_mode = args.optimization_mode

    train_corpus = utils.load_data(
        dataset_name, path=f"data/{dataset_name}/train.jsonl"
    )  # we will use train corpus to optimize the separator

    test_corpus = utils.load_data(
        dataset_name, path=f"data/{dataset_name}/dev_subsample.jsonl"
    )

    context_corpus = utils.load_data(
        dataset_name, path=f"data/{dataset_name}/train_full.jsonl"
    )  # a separate context corpus to construct the context

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.pipeline("text-generation", model=model_name, device=device)

    context_examples = utils.balanced_sample(
        context_corpus, n_shot=context_shot_size
    )

    random.shuffle(context_examples)

    train_data = [
        train_corpus.__getitem__(i, include_output=False)
        for i in range(len(train_corpus))
    ]
    train_data = train_data[:corpus_size]

    test_data = [
        test_corpus.__getitem__(i, include_output=False)
        for i in range(len(test_corpus))
    ]

    context = "\n\n".join([elem["prompt"] for elem in context_examples])

    if optimization_mode == "random_vocab":
        random_separator_texts = [generate_random_vocab_separator(model, min_separator_length, max_separator_length) for _ in range(num_random_draw)]
    elif optimization_mode == "random_wo_context":
        random_separator_texts = [generate_random_wo_context_separator(model, min_separator_length, max_separator_length) for _ in range(num_random_draw)]
    else:
        raise NotImplementedError

    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    model.tokenizer.padding_side = "left"

    print("running random separator search over train set...")
    separator_search_result = []
    for i, separator_text in enumerate(random_separator_texts):
        text_sequences = [context + "\n\n" + train_instance["prompt"] for train_instance in train_data]
        text_sequences = [elem.replace("{separator}", separator_text) for elem in text_sequences]
        labels = [train_instance["output"] for train_instance in train_data]
        r = model(text_sequences, max_new_tokens=1, return_full_text=False, do_sample=False, batch_size=16)
        predictions = [elem[0]['generated_text'].strip() for elem in r]
        accuracy = compute_accuracy(predictions=predictions, labels=labels)
        print(f"{i + 1}/{num_random_draw} - accuracy: {accuracy}, separator: {repr(separator_text)}")
        separator_search_result.append({"separator": separator_text,
                                        "accuracy": accuracy})

    separator_search_result = sorted(separator_search_result, key=lambda x: x["accuracy"], reverse=True)

    average_accuracy = sum([elem['accuracy'] for elem in separator_search_result]) / len(separator_search_result)
    print(f"average train accuracy: {average_accuracy}")

    # select top 4 separators
    top_separators = separator_search_result[:4]
    top_accuracy = []
    for i, separator in enumerate(top_separators):
        train_accuracy = separator['accuracy']
        separator_text = separator['separator']
        text_sequences = [context + "\n\n" + test_instance["prompt"] for test_instance in test_data]
        text_sequences = [elem.replace("{separator}", separator_text) for elem in text_sequences]
        labels = [test_instance["output"] for test_instance in test_data]
        r = model(text_sequences, max_new_tokens=1, return_full_text=False, do_sample=False, batch_size=16)
        predictions = [elem[0]['generated_text'].strip() for elem in r]
        accuracy = compute_accuracy(predictions=predictions, labels=labels)
        print(
            f"top {i + 1} separator: {repr(separator_text)} - accuracy: {accuracy} - train accuracy: {train_accuracy}")
        top_accuracy.append(accuracy)
    print(f"average test accuracy: {sum(top_accuracy) / len(top_accuracy)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--model", type=str, default="gpt2-large")
    parser.add_argument(
        "--min_separator_length", type=int, default=1
    )
    parser.add_argument(
        "--max_separator_length", type=int, default=5
    )
    parser.add_argument("--num_random_draw", type=int, default=160)
    parser.add_argument(
        "--context_shot_size", type=int, default=4
    )  # ICL demonstration, context_shot_size is the number of shots we use to construct the balanced context

    parser.add_argument("--corpus_size", type=int, default=64)

    parser.add_argument(
        "--optimization_mode",
        choices=[
            "random_vocab",
            "random_wo_context"
        ],
        default="random_wo_context",
    )

    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    main(args)
