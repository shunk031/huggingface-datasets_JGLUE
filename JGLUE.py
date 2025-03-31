import json
import logging
import random
import string
import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import datasets as ds
import pandas as pd

logger = logging.getLogger(__name__)

_JGLUE_CITATION = """\
@inproceedings{kurihara-lrec-2022-jglue,
  title={JGLUE: Japanese general language understanding evaluation},
  author={Kurihara, Kentaro and Kawahara, Daisuke and Shibata, Tomohide},
  booktitle={Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  pages={2957--2966},
  year={2022},
  url={https://aclanthology.org/2022.lrec-1.317/}
}
@inproceedings{kurihara-nlp-2022-jglue,
  title={JGLUE: 日本語言語理解ベンチマーク},
  author={栗原健太郎 and 河原大輔 and 柴田知秀},
  booktitle={言語処理学会第28回年次大会},
  pages={2023--2028},
  year={2022},
  url={https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf},
  note={in Japanese}
}
"""

_JCOLA_CITATION = """\
@article{someya2023jcola,
  title={JCoLA: Japanese Corpus of Linguistic Acceptability}, 
  author={Taiga Someya and Yushi Sugimoto and Yohei Oseki},
  year={2023},
  eprint={2309.12676},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
@inproceedings{someya-nlp-2022-jcola,
  title={日本語版 CoLA の構築},
  author={染谷 大河 and 大関 洋平},
  booktitle={言語処理学会第28回年次大会},
  pages={1872--1877},
  year={2022},
  url={https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E7-1.pdf},
  note={in Japanese}
}
"""

_MARC_JA_CITATION = """\
@inproceedings{marc_reviews,
  title={The Multilingual Amazon Reviews Corpus},
  author={Keung, Phillip and Lu, Yichao and Szarvas, György and Smith, Noah A.},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  pages={4563--4568},
  year={2020}
}
"""

_JSTS_JNLI_CITATION = """\
@inproceedings{miyazaki2016cross,
  title={Cross-lingual image caption generation},
  author={Miyazaki, Takashi and Shimizu, Nobuyuki},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1780--1790},
  year={2016}
}
"""

_DESCRIPTION = """\
JGLUE, Japanese General Language Understanding Evaluation, \
is built to measure the general NLU ability in Japanese. JGLUE has been constructed \
from scratch without translation. We hope that JGLUE will facilitate NLU research in Japanese.\
"""

_JGLUE_HOMEPAGE = "https://github.com/yahoojapan/JGLUE"
_JCOLA_HOMEPAGE = "https://github.com/osekilab/JCoLA"
_MARC_JA_HOMEPAGE = "https://registry.opendata.aws/amazon-reviews-ml/"

_JGLUE_LICENSE = """\
This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.\
"""

_DESCRIPTION_CONFIGS = {
    "MARC-ja": "MARC-ja is a dataset of the text classification task. This dataset is based on the Japanese portion of Multilingual Amazon Reviews Corpus (MARC) (Keung+, 2020).",
    "JCoLA": "JCoLA (Japanese Corpus of Linguistic Accept010 ability) is a novel dataset for targeted syntactic evaluations of language models in Japanese, which consists of 10,020 sentences with acceptability judgments by linguists.",
    "JSTS": "JSTS is a Japanese version of the STS (Semantic Textual Similarity) dataset. STS is a task to estimate the semantic similarity of a sentence pair.",
    "JNLI": "JNLI is a Japanese version of the NLI (Natural Language Inference) dataset. NLI is a task to recognize the inference relation that a premise sentence has to a hypothesis sentence.",
    "JSQuAD": "JSQuAD is a Japanese version of SQuAD (Rajpurkar+, 2016), one of the datasets of reading comprehension.",
    "JCommonsenseQA": "JCommonsenseQA is a Japanese version of CommonsenseQA (Talmor+, 2019), which is a multiple-choice question answering dataset that requires commonsense reasoning ability.",
}

_URLS = {
    "MARC-ja": {
        "data": "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_JP_v1_00.tsv.gz",
        "filter_review_id_list": {
            "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/preprocess/marc-ja/data/filter_review_id_list/valid.txt",
        },
        "label_conv_review_id_list": {
            "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/preprocess/marc-ja/data/label_conv_review_id_list/valid.txt",
        },
    },
    "JCoLA": {
        "train": {
            "in_domain": {
                "json": "https://raw.githubusercontent.com/osekilab/JCoLA/main/data/jcola-v1.0/in_domain_train-v1.0.json",
            }
        },
        "valid": {
            "in_domain": {
                "json": "https://raw.githubusercontent.com/osekilab/JCoLA/main/data/jcola-v1.0/in_domain_valid-v1.0.json",
            },
            "out_of_domain": {
                "json": "https://raw.githubusercontent.com/osekilab/JCoLA/main/data/jcola-v1.0/out_of_domain_valid-v1.0.json",
                "json_annotated": "https://raw.githubusercontent.com/osekilab/JCoLA/main/data/jcola-v1.0/out_of_domain_valid_annotated-v1.0.json",
            },
        },
    },
    "JSTS": {
        "train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jsts-v1.2/train-v1.2.json",
        "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jsts-v1.2/valid-v1.2.json",
    },
    "JNLI": {
        "train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jnli-v1.2/train-v1.2.json",
        "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jnli-v1.2/valid-v1.2.json",
    },
    "JSQuAD": {
        "train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jsquad-v1.2/train-v1.2.json",
        "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jsquad-v1.2/valid-v1.2.json",
    },
    "JCommonsenseQA": {
        "train": "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jcommonsenseqa-v1.2/train-v1.2.json",
        "valid": "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jcommonsenseqa-v1.2/valid-v1.2.json",
    },
}


def dataset_info_jsts() -> ds.DatasetInfo:
    features = ds.Features(
        {
            "sentence_pair_id": ds.Value("string"),
            "yjcaptions_id": ds.Value("string"),
            "sentence1": ds.Value("string"),
            "sentence2": ds.Value("string"),
            "label": ds.Value("float"),
        }
    )
    return ds.DatasetInfo(
        description=_DESCRIPTION,
        citation=_JGLUE_CITATION,
        homepage=f"{_JSTS_JNLI_CITATION}\n{_JGLUE_HOMEPAGE}",
        license=_JGLUE_LICENSE,
        features=features,
    )


def dataset_info_jnli() -> ds.DatasetInfo:
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
    return ds.DatasetInfo(
        description=_DESCRIPTION,
        citation=_JGLUE_CITATION,
        homepage=f"{_JSTS_JNLI_CITATION}\n{_JGLUE_HOMEPAGE}",
        license=_JGLUE_LICENSE,
        features=features,
        supervised_keys=None,
    )


def dataset_info_jsquad() -> ds.DatasetInfo:
    features = ds.Features(
        {
            "id": ds.Value("string"),
            "title": ds.Value("string"),
            "context": ds.Value("string"),
            "question": ds.Value("string"),
            "answers": ds.Sequence(
                {"text": ds.Value("string"), "answer_start": ds.Value("int32")}
            ),
            "is_impossible": ds.Value("bool"),
        }
    )
    return ds.DatasetInfo(
        description=_DESCRIPTION,
        citation=_JGLUE_CITATION,
        homepage=_JGLUE_HOMEPAGE,
        license=_JGLUE_LICENSE,
        features=features,
        supervised_keys=None,
    )


def dataset_info_jcommonsenseqa() -> ds.DatasetInfo:
    features = ds.Features(
        {
            "q_id": ds.Value("int64"),
            "question": ds.Value("string"),
            "choice0": ds.Value("string"),
            "choice1": ds.Value("string"),
            "choice2": ds.Value("string"),
            "choice3": ds.Value("string"),
            "choice4": ds.Value("string"),
            "label": ds.ClassLabel(
                num_classes=5,
                names=["choice0", "choice1", "choice2", "choice3", "choice4"],
            ),
        }
    )
    return ds.DatasetInfo(
        description=_DESCRIPTION,
        citation=_JGLUE_CITATION,
        homepage=_JGLUE_HOMEPAGE,
        license=_JGLUE_LICENSE,
        features=features,
    )


def dataset_info_jcola() -> ds.DatasetInfo:
    features = ds.Features(
        {
            "uid": ds.Value("int64"),
            "source": ds.Value("string"),
            "label": ds.ClassLabel(
                num_classes=2,
                names=["unacceptable", "acceptable"],
            ),
            "diacritic": ds.Value("string"),
            "sentence": ds.Value("string"),
            "original": ds.Value("string"),
            "translation": ds.Value("string"),
            "gloss": ds.Value("bool"),
            "linguistic_phenomenon": {
                "argument_structure": ds.Value("bool"),
                "binding": ds.Value("bool"),
                "control_raising": ds.Value("bool"),
                "ellipsis": ds.Value("bool"),
                "filler_gap": ds.Value("bool"),
                "island_effects": ds.Value("bool"),
                "morphology": ds.Value("bool"),
                "nominal_structure": ds.Value("bool"),
                "negative_polarity_concord_items": ds.Value("bool"),
                "quantifier": ds.Value("bool"),
                "verbal_agreement": ds.Value("bool"),
                "simple": ds.Value("bool"),
            },
        }
    )
    return ds.DatasetInfo(
        description=_DESCRIPTION,
        citation=f"{_JCOLA_CITATION}\n{_JGLUE_CITATION}",
        homepage=_JCOLA_HOMEPAGE,
        features=features,
    )


def dataset_info_marc_ja() -> ds.DatasetInfo:
    features = ds.Features(
        {
            "sentence": ds.Value("string"),
            "label": ds.ClassLabel(
                num_classes=3, names=["positive", "negative", "neutral"]
            ),
            "review_id": ds.Value("string"),
        }
    )
    return ds.DatasetInfo(
        description=_DESCRIPTION,
        citation=f"{_MARC_JA_CITATION}\n{_JGLUE_CITATION}",
        homepage=_MARC_JA_HOMEPAGE,
        license=_JGLUE_LICENSE,
        features=features,
    )


@dataclass
class JGLUEConfig(ds.BuilderConfig):
    """Class for JGLUE benchmark configuration"""


@dataclass
class MarcJaConfig(JGLUEConfig):
    name: str = "MARC-ja"
    is_han_to_zen: bool = False
    max_instance_num: Optional[int] = None
    max_char_length: int = 500
    is_pos_neg: bool = True
    train_ratio: float = 0.94
    val_ratio: float = 0.03
    test_ratio: float = 0.03
    output_testset: bool = False
    filter_review_id_list_valid: bool = True
    label_conv_review_id_list_valid: bool = True

    def __post_init__(self) -> None:
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0


JcolaDomain = Literal["in_domain", "out_of_domain"]


@dataclass
class JcolaConfig(JGLUEConfig):
    name: str = "JCoLA"
    domain: JcolaDomain = "in_domain"


def get_label(rating: int, is_pos_neg: bool = False) -> Optional[str]:
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    else:
        if is_pos_neg:
            return None
        else:
            return "neutral"


def is_filtered_by_ascii_rate(text: str, threshold: float = 0.9) -> bool:
    ascii_letters = set(string.printable)
    rate = sum(c in ascii_letters for c in text) / len(text)
    return rate >= threshold


def shuffle_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    instances = df.to_dict(orient="records")
    random.seed(1)
    random.shuffle(instances)
    return pd.DataFrame(instances)


def get_filter_review_id_list(
    filter_review_id_list_paths: Dict[str, str],
) -> Dict[str, List[str]]:
    filter_review_id_list_valid = filter_review_id_list_paths.get("valid")
    filter_review_id_list_test = filter_review_id_list_paths.get("test")

    filter_review_id_list = {}

    if filter_review_id_list_valid is not None:
        with open(filter_review_id_list_valid, "r", encoding="utf-8") as rf:
            filter_review_id_list["valid"] = [line.rstrip() for line in rf]

    if filter_review_id_list_test is not None:
        with open(filter_review_id_list_test, "r", encoding="utf-8") as rf:
            filter_review_id_list["test"] = [line.rstrip() for line in rf]

    return filter_review_id_list


def get_label_conv_review_id_list(
    label_conv_review_id_list_paths: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    import csv

    label_conv_review_id_list_valid = label_conv_review_id_list_paths.get("valid")
    label_conv_review_id_list_test = label_conv_review_id_list_paths.get("test")

    label_conv_review_id_list: Dict[str, Dict[str, str]] = {}

    if label_conv_review_id_list_valid is not None:
        with open(label_conv_review_id_list_valid, "r", encoding="utf-8") as rf:
            label_conv_review_id_list["valid"] = {
                row[0]: row[1] for row in csv.reader(rf)
            }

    if label_conv_review_id_list_test is not None:
        with open(label_conv_review_id_list_test, "r", encoding="utf-8") as rf:
            label_conv_review_id_list["test"] = {
                row[0]: row[1] for row in csv.reader(rf)
            }

    return label_conv_review_id_list


def output_data(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    output_testset: bool,
    filter_review_id_list_paths: Dict[str, str],
    label_conv_review_id_list_paths: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    instance_num = len(df)
    split_dfs: Dict[str, pd.DataFrame] = {}
    length1 = int(instance_num * train_ratio)
    split_dfs["train"] = df.iloc[:length1]

    length2 = int(instance_num * (train_ratio + val_ratio))
    split_dfs["valid"] = df.iloc[length1:length2]
    split_dfs["test"] = df.iloc[length2:]

    filter_review_id_list = get_filter_review_id_list(
        filter_review_id_list_paths=filter_review_id_list_paths,
    )
    label_conv_review_id_list = get_label_conv_review_id_list(
        label_conv_review_id_list_paths=label_conv_review_id_list_paths,
    )

    for eval_type in ("valid", "test"):
        if filter_review_id_list.get(eval_type):
            df = split_dfs[eval_type]
            df = df[~df["review_id"].isin(filter_review_id_list[eval_type])]
            split_dfs[eval_type] = df

    for eval_type in ("valid", "test"):
        if label_conv_review_id_list.get(eval_type):
            df = split_dfs[eval_type]
            df = df.assign(
                converted_label=df["review_id"].map(label_conv_review_id_list["valid"])
            )
            df = df.assign(
                label=df[["label", "converted_label"]].apply(
                    lambda xs: xs["label"]
                    if pd.isnull(xs["converted_label"])
                    else xs["converted_label"],
                    axis=1,
                )
            )
            df = df.drop(columns=["converted_label"])
            split_dfs[eval_type] = df

    return {
        "train": split_dfs["train"],
        "valid": split_dfs["valid"],
    }


def preprocess_for_marc_ja(
    config: MarcJaConfig,
    data_file_path: str,
    filter_review_id_list_paths: Dict[str, str],
    label_conv_review_id_list_paths: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    try:
        import mojimoji

        def han_to_zen(text: str) -> str:
            return mojimoji.han_to_zen(text)

    except ImportError:
        warnings.warn(
            "can't import `mojimoji`, failing back to method that do nothing. "
            "We recommend running `pip install mojimoji` to reproduce the original preprocessing.",
            UserWarning,
        )

        def han_to_zen(text: str) -> str:
            return text

    try:
        from bs4 import BeautifulSoup

        def cleanup_text(text: str) -> str:
            return BeautifulSoup(text, "html.parser").get_text()

    except ImportError:
        warnings.warn(
            "can't import `beautifulsoup4`, failing back to method that do nothing."
            "We recommend running `pip install beautifulsoup4` to reproduce the original preprocessing.",
            UserWarning,
        )

        def cleanup_text(text: str) -> str:
            return text

    from tqdm import tqdm

    df = pd.read_csv(data_file_path, delimiter="\t")
    df = df[["review_body", "star_rating", "review_id"]]

    # rename columns
    df = df.rename(columns={"review_body": "text", "star_rating": "rating"})

    # convert the rating to label
    tqdm.pandas(dynamic_ncols=True, desc="Convert the rating to the label")
    df = df.assign(
        label=df["rating"].progress_apply(
            lambda rating: get_label(rating, config.is_pos_neg)
        )
    )

    # remove rows where the label is None
    df = df[~df["label"].isnull()]

    # remove html tags from the text
    tqdm.pandas(dynamic_ncols=True, desc="Remove html tags from the text")
    df = df.assign(text=df["text"].progress_apply(cleanup_text))

    # filter by ascii rate
    tqdm.pandas(dynamic_ncols=True, desc="Filter by ascii rate")
    df = df[~df["text"].progress_apply(is_filtered_by_ascii_rate)]

    if config.max_char_length is not None:
        df = df[df["text"].str.len() <= config.max_char_length]

    if config.is_han_to_zen:
        df = df.assign(text=df["text"].apply(han_to_zen))

    df = df[["text", "label", "review_id"]]
    df = df.rename(columns={"text": "sentence"})

    # shuffle dataset
    df = shuffle_dataframe(df)

    split_dfs = output_data(
        df=df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        output_testset=config.output_testset,
        filter_review_id_list_paths=filter_review_id_list_paths,
        label_conv_review_id_list_paths=label_conv_review_id_list_paths,
    )
    return split_dfs


class JGLUE(ds.GeneratorBasedBuilder):
    JGLUE_VERSION = ds.Version("1.2.0")
    JCOLA_VERSION = ds.Version("1.0.0")

    BUILDER_CONFIG_CLASS = JGLUEConfig
    BUILDER_CONFIGS = [
        MarcJaConfig(
            name="MARC-ja",
            version=JGLUE_VERSION,
            description=_DESCRIPTION_CONFIGS["MARC-ja"],
        ),
        JcolaConfig(
            name="JCoLA",
            version=JCOLA_VERSION,
            description=_DESCRIPTION_CONFIGS["JCoLA"],
        ),
        JGLUEConfig(
            name="JSTS",
            version=JGLUE_VERSION,
            description=_DESCRIPTION_CONFIGS["JSTS"],
        ),
        JGLUEConfig(
            name="JNLI",
            version=JGLUE_VERSION,
            description=_DESCRIPTION_CONFIGS["JNLI"],
        ),
        JGLUEConfig(
            name="JSQuAD",
            version=JGLUE_VERSION,
            description=_DESCRIPTION_CONFIGS["JSQuAD"],
        ),
        JGLUEConfig(
            name="JCommonsenseQA",
            version=JGLUE_VERSION,
            description=_DESCRIPTION_CONFIGS["JCommonsenseQA"],
        ),
    ]

    def _info(self) -> ds.DatasetInfo:
        if self.config.name == "JSTS":
            return dataset_info_jsts()
        elif self.config.name == "JNLI":
            return dataset_info_jnli()
        elif self.config.name == "JSQuAD":
            return dataset_info_jsquad()
        elif self.config.name == "JCommonsenseQA":
            return dataset_info_jcommonsenseqa()
        elif self.config.name == "JCoLA":
            return dataset_info_jcola()
        elif self.config.name == "MARC-ja":
            return dataset_info_marc_ja()
        else:
            raise ValueError(f"Invalid config name: {self.config.name}")

    def __split_generators_marc_ja(self, dl_manager: ds.DownloadManager):
        try:
            file_paths = dl_manager.download_and_extract(_URLS[self.config.name])
        except FileNotFoundError as err:
            logger.warning(err)
            # An error occurs because the file cannot be downloaded from _URLS[MARC-ja]['data'].
            # So, remove the 'data' key and try to download again.
            urls = _URLS[self.config.name]
            urls.pop("data")  # type: ignore[attr-defined]
            file_paths = dl_manager.download_and_extract(urls)

        filter_review_id_list = file_paths["filter_review_id_list"]
        label_conv_review_id_list = file_paths["label_conv_review_id_list"]

        try:
            split_dfs = preprocess_for_marc_ja(
                config=self.config,
                data_file_path=file_paths["data"],
                filter_review_id_list_paths=filter_review_id_list,
                label_conv_review_id_list_paths=label_conv_review_id_list,
            )
        except KeyError as err:
            from urllib.parse import urljoin

            logger.warning(err)

            base_url = "https://huggingface.co/datasets/shunk031/JGLUE/resolve/refs%2Fconvert%2Fparquet/MARC-ja/"
            marcja_parquet_urls = {
                "train": urljoin(base_url, "jglue-train.parquet"),
                "valid": urljoin(base_url, "jglue-validation.parquet"),
            }
            file_paths = dl_manager.download_and_extract(marcja_parquet_urls)
            split_dfs = {k: pd.read_parquet(v) for k, v in file_paths.items()}

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"split_df": split_dfs["train"]},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={"split_df": split_dfs["valid"]},
            ),
        ]

    def __split_generators_jcola(self, dl_manager: ds.DownloadManager):
        file_paths = dl_manager.download_and_extract(_URLS[self.config.name])

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"file_path": file_paths["train"]["in_domain"]["json"]},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={"file_path": file_paths["valid"]["in_domain"]["json"]},
            ),
            ds.SplitGenerator(
                name=ds.NamedSplit("validation_out_of_domain"),
                gen_kwargs={"file_path": file_paths["valid"]["out_of_domain"]["json"]},
            ),
            ds.SplitGenerator(
                name=ds.NamedSplit("validation_out_of_domain_annotated"),
                gen_kwargs={
                    "file_path": file_paths["valid"]["out_of_domain"]["json_annotated"]
                },
            ),
        ]

    def __split_generators(self, dl_manager: ds.DownloadManager):
        file_paths = dl_manager.download_and_extract(_URLS[self.config.name])

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={"file_path": file_paths["train"]},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={"file_path": file_paths["valid"]},
            ),
        ]

    def _split_generators(self, dl_manager: ds.DownloadManager):
        if self.config.name == "MARC-ja":
            return self.__split_generators_marc_ja(dl_manager)
        elif self.config.name == "JCoLA":
            return self.__split_generators_jcola(dl_manager)
        else:
            return self.__split_generators(dl_manager)

    def __generate_examples_marc_ja(self, split_df: Optional[pd.DataFrame] = None):
        if split_df is None:
            raise ValueError(f"Invalid preprocessing for {self.config.name}")

        instances = split_df.to_dict(orient="records")
        for i, data_dict in enumerate(instances):
            yield i, data_dict

    def __generate_examples_jcola(self, file_path: Optional[str] = None):
        if file_path is None:
            raise ValueError(f"Invalid argument for {self.config.name}")

        def convert_label(json_dict):
            label_int = json_dict["label"]
            label_str = "unacceptable" if label_int == 0 else "acceptable"
            json_dict["label"] = label_str
            return json_dict

        def convert_addntional_info(json_dict):
            json_dict["translation"] = json_dict.get("translation")
            json_dict["gloss"] = json_dict.get("gloss")
            return json_dict

        def convert_phenomenon(json_dict):
            argument_structure = json_dict.get("Arg. Str.")

            def json_pop(key):
                return json_dict.pop(key) if argument_structure is not None else None

            json_dict["linguistic_phenomenon"] = {
                "argument_structure": json_pop("Arg. Str."),
                "binding": json_pop("binding"),
                "control_raising": json_pop("control/raising"),
                "ellipsis": json_pop("ellipsis"),
                "filler_gap": json_pop("filler-gap"),
                "island_effects": json_pop("island effects"),
                "morphology": json_pop("morphology"),
                "nominal_structure": json_pop("nominal structure"),
                "negative_polarity_concord_items": json_pop("NPI/NCI"),
                "quantifier": json_pop("quantifier"),
                "verbal_agreement": json_pop("verbal agr."),
                "simple": json_pop("simple"),
            }
            return json_dict

        with open(file_path, "r", encoding="utf-8") as rf:
            for i, line in enumerate(rf):
                json_dict = json.loads(line)

                example = convert_label(json_dict)
                example = convert_addntional_info(example)
                example = convert_phenomenon(example)

                yield i, example

    def __generate_examples_jsquad(self, file_path: Optional[str] = None):
        if file_path is None:
            raise ValueError(f"Invalid argument for {self.config.name}")

        with open(file_path, "r", encoding="utf-8") as rf:
            json_data = json.load(rf)

        for json_dict in json_data["data"]:
            title = json_dict["title"]
            paragraphs = json_dict["paragraphs"]

            for paragraph in paragraphs:
                context = paragraph["context"]
                questions = paragraph["qas"]

                for question_dict in questions:
                    q_id = question_dict["id"]
                    question = question_dict["question"]
                    answers = question_dict["answers"]
                    is_impossible = question_dict["is_impossible"]

                    example_dict = {
                        "id": q_id,
                        "title": title,
                        "context": context,
                        "question": question,
                        "answers": answers,
                        "is_impossible": is_impossible,
                    }

                    yield q_id, example_dict

    def __generate_examples_jcommonsenseqa(self, file_path: Optional[str] = None):
        if file_path is None:
            raise ValueError(f"Invalid argument for {self.config.name}")

        with open(file_path, "r", encoding="utf-8") as rf:
            for i, line in enumerate(rf):
                json_dict = json.loads(line)
                json_dict["label"] = f"choice{json_dict['label']}"
                yield i, json_dict

    def __generate_examples(self, file_path: Optional[str] = None):
        if file_path is None:
            raise ValueError(f"Invalid argument for {self.config.name}")

        with open(file_path, "r", encoding="utf-8") as rf:
            for i, line in enumerate(rf):
                json_dict = json.loads(line)
                yield i, json_dict

    def _generate_examples(
        self,
        file_path: Optional[str] = None,
        split_df: Optional[pd.DataFrame] = None,
    ):
        if self.config.name == "MARC-ja":
            yield from self.__generate_examples_marc_ja(split_df)

        elif self.config.name == "JCoLA":
            yield from self.__generate_examples_jcola(file_path)

        elif self.config.name == "JSQuAD":
            yield from self.__generate_examples_jsquad(file_path)

        elif self.config.name == "JCommonsenseQA":
            yield from self.__generate_examples_jcommonsenseqa(file_path)

        else:
            yield from self.__generate_examples(file_path)
