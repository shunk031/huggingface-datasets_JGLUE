import json

import datasets as ds

_CITATION = """\
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}

@InProceedings{Kurihara_nlp2022,
  author = 	"栗原健太郎 and 河原大輔 and 柴田知秀",
  title = 	"JGLUE: 日本語言語理解ベンチマーク",
  booktitle = 	"言語処理学会第28回年次大会",
  year =	"2022",
  url = "https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf"
  note= "in Japanese"
}
"""

_DESCRIPTION = """\
JGLUE, Japanese General Language Understanding Evaluation, is built to measure the general NLU ability in Japanese. JGLUE has been constructed from scratch without translation. We hope that JGLUE will facilitate NLU research in Japanese.
"""

_HOMEPAGE = "https://github.com/yahoojapan/JGLUE"

_LICENSE = """\
This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
"""

_DESCRIPTION_CONFIGS = {
    "JSTS": "JSTS is a Japanese version of the STS (Semantic Textual Similarity) dataset. STS is a task to estimate the semantic similarity of a sentence pair.",
    "JNLI": "JNLI is a Japanese version of the NLI (Natural Language Inference) dataset. NLI is a task to recognize the inference relation that a premise sentence has to a hypothesis sentence.",
    "JSQuAD": "JSQuAD is a Japanese version of SQuAD (Rajpurkar+, 2016), one of the datasets of reading comprehension.",
    "JCommonsenseQA": "JCommonsenseQA is a Japanese version of CommonsenseQA (Talmor+, 2019), which is a multiple-choice question answering dataset that requires commonsense reasoning ability.",
}

_URLS = {
    "JSTS": {
        "train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsts-v1.1/train-v1.1.json",
        "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsts-v1.1/valid-v1.1.json",
    },
    "JNLI": {
        "train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jnli-v1.1/train-v1.1.json",
        "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jnli-v1.1/valid-v1.1.json",
    },
    "JSQuAD": {
        "train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsquad-v1.1/train-v1.1.json",
        "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsquad-v1.1/valid-v1.1.json",
    },
    "JCommonsenseQA": {
        "train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jcommonsenseqa-v1.1/train-v1.1.json",
        "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jcommonsenseqa-v1.1/valid-v1.1.json",
    },
}


def features_jsts() -> ds.Features:
    features = ds.Features(
        {
            "sentence_pair_id": ds.Value("string"),
            "yjcaptions_id": ds.Value("string"),
            "sentence1": ds.Value("string"),
            "sentence2": ds.Value("string"),
            "label": ds.Value("float"),
        }
    )
    return features


def features_jnli() -> ds.Features:
    features = ds.Features(
        {
            "sentence_pair_id": ds.Value("string"),
            "yjcaptions_id": ds.Value("string"),
            "sentence1": ds.Value("string"),
            "sentence2": ds.Value("string"),
            "label": ds.ClassLabel(
                num_classes=3, names=["entailment", "contradiction", "neutral"]
            ),
        }
    )
    return features


def features_jsquad() -> ds.Features:
    title = ds.Value("string")
    answers = ds.Sequence(
        {"text": ds.Value("string"), "answer_start": ds.Value("int64")}
    )
    qas = ds.Sequence(
        {
            "question": ds.Value("string"),
            "id": ds.Value("string"),
            "answers": answers,
            "is_impossible": ds.Value("bool"),
        }
    )
    paragraphs = ds.Sequence({"qas": qas, "context": ds.Value("string")})
    features = ds.Features(
        {"data": ds.Sequence({"title": title, "paragraphs": paragraphs})}
    )
    return features


def features_jcommonsenseqa() -> ds.Features:
    features = ds.Features(
        {
            "q_id": ds.Value("int64"),
            "question": ds.Value("string"),
            "choice0": ds.Value("string"),
            "choice1": ds.Value("string"),
            "choice2": ds.Value("string"),
            "choice3": ds.Value("string"),
            "choice4": ds.Value("string"),
            "label": ds.Value("int8"),
        }
    )
    return features


class JGLUE(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.1.0")
    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name="JSTS",
            version=VERSION,
            description=_DESCRIPTION_CONFIGS["JSTS"],
        ),
        ds.BuilderConfig(
            name="JNLI",
            version=VERSION,
            description=_DESCRIPTION_CONFIGS["JNLI"],
        ),
        ds.BuilderConfig(
            name="JSQuAD",
            version=VERSION,
            description=_DESCRIPTION_CONFIGS["JSQuAD"],
        ),
        ds.BuilderConfig(
            name="JCommonsenseQA",
            version=VERSION,
            description=_DESCRIPTION_CONFIGS["JCommonsenseQA"],
        ),
    ]

    def _info(self) -> ds.DatasetInfo:
        if self.config.name == "JSTS":
            features = features_jsts()
        elif self.config.name == "JNLI":
            features = features_jnli()
        elif self.config.name == "JSQuAD":
            features = features_jsquad()
        elif self.config.name == "JCommonsenseQA":
            features = features_jcommonsenseqa()
        else:
            raise ValueError(f"Invalid config name: {self.config.name}")

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        file_paths = dl_manager.download_and_extract(_URLS[self.config.name])
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "file_path": file_paths["train"],
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={
                    "file_path": file_paths["valid"],
                },
            ),
        ]

    def _generate_examples(self, file_path: str):
        with open(file_path, "r") as rf:
            for i, line in enumerate(rf):
                json_dict = json.loads(line)
                yield i, json_dict
