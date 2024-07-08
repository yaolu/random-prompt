import json


class TextClassificationDataset:
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        self.task_type = "classification"
        self.path = path
        self.data = self.load_data(path)
        self.prompt_template = prompt_template
        self.label_mapping = {}

    def load_data(self, path):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def __len__(self):
        return len(self.data)

    def make_prompt(self, input_text, output_text):
        return self.prompt_template.format(
            input_text=input_text, separator="{separator}", output_text=output_text
        )

    def __getitem__(self, idx, include_output=False):
        instance = self.data[idx]
        input_text = instance["sentence"]
        output_text = self.label_mapping[instance["label"]]

        if include_output:
            prompt = self.make_prompt(input_text, output_text).strip()
        else:
            prompt = self.make_prompt(input_text, "").strip()
        return {"prompt": prompt, "output": output_text}


class RTEDataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {"not_entailment": "False", "entailment": "True"}

    def __getitem__(self, idx, include_output=False):
        instance = self.data[idx]
        input_text_a = instance["sentence_1"]
        input_text_b = instance["sentence_2"]
        input_text = f"{input_text_a} {input_text_b}"  # TODO: add a separator if necessary, whitespace for now
        output_text = self.label_mapping[instance["label"]]

        if include_output:
            prompt = self.make_prompt(input_text, output_text).strip()
        else:
            prompt = self.make_prompt(input_text, "").strip()
        return {"prompt": prompt, "output": output_text}


class CBDataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "contradiction": "false",
            "entailment": "true",
            "neutral": "neither",
        }

    def __getitem__(self, idx, include_output=False):
        instance = self.data[idx]
        input_text_a = instance["premise"]
        input_text_b = instance["hypothesis"]
        input_text = f"{input_text_a} {input_text_b}"  # TODO: add a separator if necessary, whitespace for now
        output_text = self.label_mapping[instance["label"]]

        if include_output:
            prompt = self.make_prompt(input_text, output_text).strip()
        else:
            prompt = self.make_prompt(input_text, "").strip()
        return {"prompt": prompt, "output": output_text}


class SST2Dataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {"0": "negative", "1": "positive"}


class TRECDataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "description",
            "1": "entity",
            "2": "expression",
            "3": "human",
            "4": "location",
            "5": "number",
        }


class AGNewsDataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "1": "world",
            "2": "sports",
            "3": "business",
            "4": "technology",
        }


class DBPediaDataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "1": "company",
            "2": "school",
            "3": "artist",
            "4": "athlete",
            "5": "politics",
            "6": "transportation",
            "7": "building",
            "8": "nature",
            "9": "village",
            "10": "animal",
            "11": "plant",
            "12": "album",
            "13": "film",
            "14": "book",
        }


class SubjDataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "subjective",
            "1": "objective",
        }


class MRDataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "negative",
            "1": "positive",
        }


class SST5Dataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "terrible",
            "1": "bad",
            "2": "okay",
            "3": "good",
            "4": "great",
        }


class MPQADataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "negative",
            "1": "positive",
        }


class CRDataset(TextClassificationDataset):
    def __init__(self, path, prompt_template="{input_text} {separator} {output_text}"):
        super().__init__(path, prompt_template)
        self.label_mapping = {
            "0": "negative",
            "1": "positive",
        }




