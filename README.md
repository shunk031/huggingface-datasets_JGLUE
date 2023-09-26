---
annotations_creators:
- crowdsourced
language:
- ja
language_creators:
- crowdsourced
- found
license:
- cc-by-4.0
multilinguality:
- monolingual
pretty_name: JGLUE
size_categories: []
source_datasets:
- original
tags:
- MARC
- CoLA
- STS
- NLI
- SQuAD
- CommonsenseQA
task_categories:
- multiple-choice
- question-answering
- sentence-similarity
- text-classification
task_ids:
- multiple-choice-qa
- open-domain-qa
- multi-class-classification
- sentiment-classification
---

# Dataset Card for JGLUE

[![CI](https://github.com/shunk031/huggingface-datasets_JGLUE/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-datasets_JGLUE/actions/workflows/ci.yaml)
[![ACL2020 2020.acl-main.419](https://img.shields.io/badge/LREC2022-2022.lrec--1.317-red)](https://aclanthology.org/2022.lrec-1.317)

This dataset loading script is developed on [GitHub](https://github.com/shunk031/huggingface-datasets_JGLUE).
Please feel free to open an [issue](https://github.com/shunk031/huggingface-datasets_JGLUE/issues/new/choose) or [pull request](https://github.com/shunk031/huggingface-datasets_JGLUE/pulls).

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://github.com/yahoojapan/JGLUE
- **Repository:** https://github.com/shunk031/huggingface-datasets_JGLUE

### Dataset Summary

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE#jglue-japanese-general-language-understanding-evaluation):

> JGLUE, Japanese General Language Understanding Evaluation, is built to measure the general NLU ability in Japanese. JGLUE has been constructed from scratch without translation. We hope that JGLUE will facilitate NLU research in Japanese.

> JGLUE has been constructed by a joint research project of Yahoo Japan Corporation and Kawahara Lab at Waseda University.

### Supported Tasks and Leaderboards

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE#tasksdatasets):

> JGLUE consists of the tasks of text classification, sentence pair classification, and QA. Each task consists of multiple datasets. 

#### Supported Tasks

##### MARC-ja

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE#marc-ja):

> MARC-ja is a dataset of the text classification task. This dataset is based on the Japanese portion of [Multilingual Amazon Reviews Corpus (MARC)](https://docs.opendata.aws/amazon-reviews-ml/readme.html) ([Keung+, 2020](https://aclanthology.org/2020.emnlp-main.369/)).

##### JCoLA

From [JCoLA's README.md](https://github.com/osekilab/JCoLA#jcola-japanese-corpus-of-linguistic-acceptability)

> JCoLA (Japanese Corpus of Linguistic Accept010 ability) is a novel dataset for targeted syntactic evaluations of language models in Japanese, which consists of 10,020 sentences with acceptability judgments by linguists. The sentences are manually extracted from linguistics journals, handbooks and textbooks. JCoLA is included in [JGLUE benchmark](https://github.com/yahoojapan/JGLUE) (Kurihara et al., 2022).

##### JSTS

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE#jsts):

> JSTS is a Japanese version of the STS (Semantic Textual Similarity) dataset. STS is a task to estimate the semantic similarity of a sentence pair. The sentences in JSTS and JNLI (described below) are extracted from the Japanese version of the MS COCO Caption Dataset, [the YJ Captions Dataset](https://github.com/yahoojapan/YJCaptions) ([Miyazaki and Shimizu, 2016](https://aclanthology.org/P16-1168/)).

##### JNLI

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE#jnli):

> JNLI is a Japanese version of the NLI (Natural Language Inference) dataset. NLI is a task to recognize the inference relation that a premise sentence has to a hypothesis sentence. The inference relations are entailment, contradiction, and neutral.

##### JSQuAD

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE#jsquad):

> JSQuAD is a Japanese version of [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) ([Rajpurkar+, 2018](https://aclanthology.org/P18-2124/)), one of the datasets of reading comprehension. Each instance in the dataset consists of a question regarding a given context (Wikipedia article) and its answer. JSQuAD is based on SQuAD 1.1 (there are no unanswerable questions). We used [the Japanese Wikipedia dump](https://dumps.wikimedia.org/jawiki/) as of 20211101.

##### JCommonsenseQA

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE#jcommonsenseqa):

> JCommonsenseQA is a Japanese version of [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) ([Talmor+, 2019](https://aclanthology.org/N19-1421/)), which is a multiple-choice question answering dataset that requires commonsense reasoning ability. It is built using crowdsourcing with seeds extracted from the knowledge base [ConceptNet](https://conceptnet.io/).

#### Leaderboard

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE#leaderboard):

> A leaderboard will be made public soon. The test set will be released at that time.

### Languages

The language data in JGLUE is in Japanese ([BCP-47 ja-JP](https://www.rfc-editor.org/info/bcp47)).

## Dataset Structure

### Data Instances

When loading a specific configuration, users has to append a version dependent suffix:

#### MARC-ja

```python
from datasets import load_dataset

dataset = load_dataset("shunk031/JGLUE", name="MARC-ja")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['sentence', 'label', 'review_id'],
#         num_rows: 187528
#     })
#     validation: Dataset({
#         features: ['sentence', 'label', 'review_id'],
#         num_rows: 5654
#     })
# })
```

#### JCoLA

```python
from datasets import load_dataset

dataset = load_dataset("shunk031/JGLUE", name="JCoLA")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['uid', 'source', 'label', 'diacritic', 'sentence', 'original', 'translation', 'gloss', 'simple', 'linguistic_phenomenon'],
#         num_rows: 6919
#     })
#     validation: Dataset({
#         features: ['uid', 'source', 'label', 'diacritic', 'sentence', 'original', 'translation', 'gloss', 'simple', 'linguistic_phenomenon'],
#         num_rows: 865
#     })
#     validation_out_of_domain: Dataset({
#         features: ['uid', 'source', 'label', 'diacritic', 'sentence', 'original', 'translation', 'gloss', 'simple', 'linguistic_phenomenon'],
#         num_rows: 685
#     })
#     validation_out_of_domain_annotated: Dataset({
#         features: ['uid', 'source', 'label', 'diacritic', 'sentence', 'original', 'translation', 'gloss', 'simple', 'linguistic_phenomenon'],
#         num_rows: 685
#     })
# })
```

An example of the JCoLA dataset (validation - out of domain annotated) looks as follows:

```json
{
  "uid": 9109,
  "source": "Asano_and_Ura_2010",
  "label": 1,
  "diacritic": "g",
  "sentence": "太郎のゴミの捨て方について話した。",
  "original": "太郎のゴミの捨て方",
  "translation": "‘The way (for Taro) to throw out garbage’",
  "gloss": true,
  "simple": false,
  "linguistic_phenomenon": {
    "argument_structure": true,
    "binding": false,
    "filler_gap": false,
    "ellipsis": false,
    "island_effects": false,
    "negative_polarity_items_licensing": false,
    "quantifier": false,
    "control_raising": false,
    "verbal_agreement": false,
    "nominal_structure": false,
    "morphology": false
  }
}
```

#### JSTS

```python
from datasets import load_dataset

dataset = load_dataset("shunk031/JGLUE", name="JSTS")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['sentence_pair_id', 'yjcaptions_id', 'sentence1', 'sentence2', 'label'],
#         num_rows: 12451
#     })
#     validation: Dataset({
#         features: ['sentence_pair_id', 'yjcaptions_id', 'sentence1', 'sentence2', 'label'],
#         num_rows: 1457
#     })
# })
```

An example of the JSTS dataset looks as follows:

```json
{
  "sentence_pair_id": "691",
  "yjcaptions_id": "127202-129817-129818",
  "sentence1": "街中の道路を大きなバスが走っています。 (A big bus is running on the road in the city.)", 
  "sentence2": "道路を大きなバスが走っています。 (There is a big bus running on the road.)", 
  "label": 4.4
}
```

#### JNLI

```python
from datasets import load_dataset

dataset = load_dataset("shunk031/JGLUE", name="JNLI")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['sentence_pair_id', 'yjcaptions_id', 'sentence1', 'sentence2', 'label'],
#         num_rows: 20073
#     })
#     validation: Dataset({
#         features: ['sentence_pair_id', 'yjcaptions_id', 'sentence1', 'sentence2', 'label'],
#         num_rows: 2434
#     })
# })
```

An example of the JNLI dataset looks as follows:

```json
{
  "sentence_pair_id": "1157",
  "yjcaptions_id": "127202-129817-129818",
  "sentence1": "街中の道路を大きなバスが走っています。 (A big bus is running on the road in the city.)", 
  "sentence2": "道路を大きなバスが走っています。 (There is a big bus running on the road.)", 
  "label": "entailment"
}
```

#### JSQuAD

```python
from datasets import load_dataset

dataset = load_dataset("shunk031/JGLUE", name="JSQuAD")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'title', 'context', 'question', 'answers', 'is_impossible'],
#         num_rows: 62859
#     })
#     validation: Dataset({
#         features: ['id', 'title', 'context', 'question', 'answers', 'is_impossible'],
#         num_rows: 4442
#     })
# })
```

An example of the JSQuAD looks as follows:

```json
{
  "id": "a1531320p0q0", 
  "title": "東海道新幹線", 
  "context": "東海道新幹線 [SEP] 1987 年（昭和 62 年）4 月 1 日の国鉄分割民営化により、JR 東海が運営を継承した。西日本旅客鉄道（JR 西日本）が継承した山陽新幹線とは相互乗り入れが行われており、東海道新幹線区間のみで運転される列車にも JR 西日本所有の車両が使用されることがある。2020 年（令和 2 年）3 月現在、東京駅 - 新大阪駅間の所要時間は最速 2 時間 21 分、最高速度 285 km/h で運行されている。", 
  "question": "2020 年（令和 2 年）3 月現在、東京駅 - 新大阪駅間の最高速度はどのくらいか。", 
  "answers": {
    "text": ["285 km/h"], 
    "answer_start": [182]
  }, 
  "is_impossible": false
}
```

#### JCommonsenseQA

```python
from datasets import load_dataset

dataset = load_dataset("shunk031/JGLUE", name="JCommonsenseQA")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['q_id', 'question', 'choice0', 'choice1', 'choice2', 'choice3', 'choice4', 'label'],
#         num_rows: 8939
#     })
#     validation: Dataset({
#         features: ['q_id', 'question', 'choice0', 'choice1', 'choice2', 'choice3', 'choice4', 'label'],
#         num_rows: 1119
#     })
# })
```

An example of the JCommonsenseQA looks as follows:

```json
{
  "q_id": 3016,
  "question": "会社の最高責任者を何というか？ (What do you call the chief executive officer of a company?)",
  "choice0": "社長 (president)",
  "choice1": "教師 (teacher)",
  "choice2": "部長 (manager)",
  "choice3": "バイト (part-time worker)",
  "choice4": "部下 (subordinate)",
  "label": 0
}
```

### Data Fields

#### MARC-ja

- `sentence_pair_id`: ID of the sentence pair
- `yjcaptions_id`: sentence ids in yjcaptions (explained below)
- `sentence1`: first sentence
- `sentence2`: second sentence
- `label`: sentence similarity: 5 (equivalent meaning) - 0 (completely different meaning)

##### Explanation for `yjcaptions_id`

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE#explanation-for-yjcaptions_id), there are the following two cases:

1. sentence pairs in one image: `(image id)-(sentence1 id)-(sentence2 id)`
    - e.g., 723-844-847
    - a sentence id starting with "g" means a sentence generated by a crowdworker (e.g., 69501-75698-g103): only for JNLI
2. sentence pairs in two images: `(image id of sentence1)_(image id of sentence2)-(sentence1 id)-(sentence2 id)`
    - e.g., 91337_217583-96105-91680

#### JCoLA

From [JCoLA's README.md](https://github.com/osekilab/JCoLA#data-description) and [JCoLA's paper](https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E7-1.pdf)

- `uid`: unique id of the sentence
- `source`: author and the year of publication of the source article
- `label`: acceptability judgement label (0 for unacceptable, 1 for acceptable)
- `diacritic`: acceptability judgement as originally notated in the source article
- `sentence`: sentence (modified by the author if needed)
- `original`: original sentence as presented in the source article
- `translation`: English translation of the sentence as presentend in the source article (if any)
- `gloss`: gloss of the sentence as presented in the source article (if any)
- `simple`: [More Information Needed]
- `linguistic_phenomenon`
  - `argument_structure`: 動詞の項構造に関わる容認性判断
  - `binding`: 名詞句の束縛関係に関する容認性判断
  - `filler_gap`: 移動した構成素と移動元の空所の依存関係に関する容認性判断
  - `ellipsis`: 文中の要素の省略可能性に関する容認性判断
  - `island effects`: 島の制約に関する容認性判断
  - `negative_polarity_items_licensing`: 否定極性項目 (negative polarity items, NPIs) の出現環境に関する容認性判断
  - `quantifiers`: 数量詞 (quantifiers) の分布に関する容認性判断
  - `control_raising`: コントロール (control) や繰り上げ (raising) 構文に関する容認性判断
  - `verbal_agreement`: BLiMP では、主語と動詞の数の一致に関する例文を、Subject-verb agreement としてまとめている。JCoLA では、より一般に主語の性質が動詞の形態に反映される現象や、動詞が主語の性質に制約を与えるような現象に関する容認性判断を含む中分類として Verbal Agreement を採用した
  - `nominal_structure`: BLiMP では、限定詞 (determiner) と名詞の一致に関する例文を、Determiner-noun agreement としてまとめている。JCoLA では、より一般に名詞句の内部構造に関わる容認性判断を含む中分類として、Nominal structure を採用した
  - `morphology`: BLiMP では、動詞の過去分詞の活用が正しく行われているかに関する例文を、Irregular forms としてまとめている。JCoLA では、より幅広く形態論に関する容認性判断を含む中分類として Morphology を採用した

#### JNLI

- `sentence_pair_id`: ID of the sentence pair
- `yjcaptions_id`: sentence ids in the yjcaptions
- `sentence1`: premise sentence
- `sentence2`: hypothesis sentence
- `label`: inference relation

#### JSQuAD

- `title`: title of a Wikipedia article
- `paragraphs`: a set of paragraphs
- `qas`: a set of pairs of a question and its answer
- `question`: question
- `id`: id of a question
- `answers`: a set of answers
- `text`: answer text
- `answer_start`: start position (character index)
- `is_impossible`: all the values are false
- `context`: a concatenation of the title and paragraph

#### JCommonsenseQA

- `q_id`: ID of the question
- `question`: question
- `choice{0..4}`: choice
- `label`: correct choice id

### Data Splits

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE/blob/main/README.md#tasksdatasets):

> Only train/dev sets are available now, and the test set will be available after the leaderboard is made public.

| Task                         | Dataset        | Train   | Dev   | Test  |
|------------------------------|----------------|--------:|------:|------:|
| Text Classification          | MARC-ja        | 187,528 | 5,654 | 5,639 |
|                              | JCoLA          | 6,919   | 865&dagger; / 685&ddagger; | - |
| Sentence Pair Classification | JSTS           | 12,451  | 1,457 | 1,589 |
|                              | JNLI           | 20,073  | 2,434 | 2,508 |
| Question Answering           | JSQuAD         | 62,859  | 4,442 | 4,420 |
|                              | JCommonsenseQA | 8,939   | 1,119 | 1,118 |

> JCoLA: &dagger; in domain. &ddagger; out of domain.

## Dataset Creation

### Curation Rationale

From [JGLUE's paper](https://aclanthology.org/2022.lrec-1.317/):

> JGLUE is designed to cover a wide range of GLUE and SuperGLUE tasks and consists of three kinds of tasks: text classification, sentence pair classification, and question answering.

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

- The source language producers are users of Amazon (MARC-ja), crowd-workers of Yahoo! Crowdsourcing (JSTS, JNLI and JCommonsenseQA), writers of the Japanese Wikipedia (JSQuAD).

### Annotations

#### Annotation process

##### MARC-ja

From [JGLUE's paper](https://aclanthology.org/2022.lrec-1.317/):

> As one of the text classification datasets, we build a dataset based on the Multilingual Amazon Reviews Corpus (MARC) (Keung et al., 2020). MARC is a multilingual corpus of product reviews with 5-level star ratings (1-5) on the Amazon shopping site. This corpus covers six languages, including English and Japanese. For JGLUE, we use the Japanese part of MARC and to make it easy for both humans and computers to judge a class label, we cast the text classification task as a binary classification task, where 1- and 2-star ratings are converted to “negative”, and 4 and 5 are converted to “positive”. We do not use reviews with a 3-star rating. 

> One of the problems with MARC is that it sometimes contains data where the rating diverges from the review text. This happens, for example, when a review with positive content is given a rating of 1 or 2. These data degrade the quality of our dataset. To improve the quality of the dev/test instances used for evaluation, we crowdsource a positive/negative judgment task for approximately 12,000 reviews. We adopt only reviews with the same votes from 7 or more out of 10 workers and assign a label of the maximum votes to these reviews. We divide the resulting reviews into dev/test data. 

> We obtained 5,654 and 5,639 instances for the dev and test data, respectively, through the above procedure. For the training data, we extracted 187,528 instances directly from MARC without performing the cleaning procedure because of the large number of training instances. The statistics of MARC-ja are listed in Table 2. For the evaluation metric for MARC-ja, we use accuracy because it is a binary classification task of texts.

##### JCoLA

From [JCoLA's paper](https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E7-1.pdf):

> ### JCoLA の構築
> #### 3.1 データ収集
>言語モデルが理論言語学で重要とされている統語現象を捉えられているかを検証するため、本研究では東アジア・東南アジア言語の言語学のジャーナルとして著名な JEAL (Journal of East Asian Linguistics) に 10 年間（2006 年から 2015 年）で掲載された 133 本の論文の中で、特に日本語の統語論を扱っている論文 28 本を対象とし、その論文で提示されている全てのデータポイント（2,323 文）を抽出した。ここでの「全てのデータポイント」は、脚注や付録を含む本文の全ての日本語の例文の中で、構造分析のために提示された例文を除いたもののことである。
> #### 3.2 タイプ分類
> 単純な全データポイントに対する正解率による比較に終始することなく、個別の統語現象ごとのモデル評価を可能にするため、前節で抽出した例文を統語現象のタイプによって分類した。本研究では、全データポイントを 3 つの粒度で分類する。分類の名称については、BLiMP [14] を参考にした。まず、大分類として各データポイントが問題としている容認性判断の性質や、本文中での提示のされ方に基づいて 8 種類に分類した。次に、中分類として各データポイントがどのような統語現象を扱っているのかに基づく分類を行った (𝑝ℎ𝑒𝑛𝑜𝑚𝑒𝑛𝑜𝑛)。中分類は、基本的に BLiMP における 12 の現象に Others を加えたものであるが、今回対象とする日本語のデータに合わせてその一部を変更した（付録 A）。また、データポイントが二つ以上の現象に分類されうると判断された場合には、𝑝ℎ𝑒𝑛𝑜𝑚𝑒𝑛𝑜𝑛-2 を用意して分類した。ただし、言語モデルの統語現象ごとの評価の際には 𝑝ℎ𝑒𝑛𝑜𝑚𝑒𝑛𝑜𝑛の分類が優先される。最後に、小分類として中分類 (𝑝ℎ𝑒𝑛𝑜𝑚𝑒𝑛𝑜𝑛) よりさらに粒度の細かい、個別の統語現象ごとに 39 種類の分類を行った (𝑝𝑎𝑟𝑎𝑑𝑖𝑔𝑚)。これにより、エラー分析の際により粒度の細かい分析が可能となる。
> #### 3.3 ミニマルペアの作成
> まず、前節でタイプ分類が行われたデータセットのうち、以下の全ての条件を満たすものを抽出
する。
> • 非文として提示されている（? や*などのマーキングがされている）もの。ただし、? などのマーキングがされつつも、本文中で正例としてみなされているものは除く。
> • 大分類が variation、repeat、footnote、appendix のいずれでもないもの。
> • 中分類が others でないもの。
> 次に、言語学の論文において提示された全ての負例には、対応する正例が存在するという仮定のもと [20]、以上により抜き出された負例のぞれぞれに対応する正例を、論文の中から採用するか、本文の内容を確認しつつ筆者が作例することにより構築した。この際、解釈により容認度が変わる例は、JCoLA に含めない不適切な例として、ミニマルペア構築の対象外とした。また、重複している例文や、語彙項目が異なるのみで検証対象が同じである例文も除外した。以上の手順により、合計で 369 ペアのミニマルペアが作成された（表 2、付録 B）。

##### JSTS and JNLI

From [JGLUE's paper](https://aclanthology.org/2022.lrec-1.317/):

> For the sentence pair classification datasets, we construct a semantic textual similarity (STS) dataset, JSTS, and a natural language inference (NLI) dataset, JNLI.

> ### Overview
> STS is a task of estimating the semantic similarity of a sentence pair. Gold similarity is usually assigned as an average of the integer values 0 (completely different meaning) to 5 (equivalent meaning) assigned by multiple workers through crowdsourcing.

> NLI is a task of recognizing the inference relation that a premise sentence has to a hypothesis sentence. Inference relations are generally defined by three labels: “entailment”, “contradiction”, and “neutral”. Gold inference relations are often assigned by majority voting after collecting answers from multiple workers through crowdsourcing.

> For the STS and NLI tasks, STS-B (Cer et al., 2017) and MultiNLI (Williams et al., 2018) are included in GLUE, respectively. As Japanese datasets, JSNLI (Yoshikoshi et al., 2020) is a machine translated dataset of the NLI dataset SNLI (Stanford NLI), and JSICK (Yanaka and Mineshima, 2021) is a human translated dataset of the STS/NLI dataset SICK (Marelli et al., 2014). As mentioned in Section 1, these have problems originating from automatic/manual translations. To solve this problem, we construct STS/NLI datasets in Japanese from scratch. We basically extract sentence pairs in JSTS and JNLI from the Japanese version of the MS COCO Caption Dataset (Chen et al., 2015), the YJ Captions Dataset (Miyazaki and Shimizu, 2016). Most of the sentence pairs in JSTS and JNLI overlap, allowing us to analyze the relationship between similarities and inference relations for the same sentence pairs like SICK and JSICK.

> The similarity value in JSTS is assigned a real number from 0 to 5 as in STS-B. The inference relation in JNLI is assigned from the above three labels as in SNLI and MultiNLI. The definitions of the inference relations are also based on SNLI.

> ### Method of Construction
> Our construction flow for JSTS and JNLI is shown in Figure 1. Basically, two captions for the same image of YJ Captions are used as sentence pairs. For these sentence pairs, similarities and NLI relations of entailment and neutral are obtained by crowdsourcing. However, it is difficult to collect sentence pairs with low similarity and contradiction relations from captions for the same image. To solve this problem, we collect sentence pairs with low similarity from captions for different images. We collect contradiction relations by asking workers to write contradictory sentences for a given caption. 

> The detailed construction procedure for JSTS and JNLI is described below.
> 1. We crowdsource an STS task using two captions for the same image from YJ Captions. We ask five workers to answer the similarity between two captions and take the mean value as the gold similarity. We delete sentence pairs with a large variance in the answers because such pairs have poor answer quality. We performed this task on 16,000 sentence pairs and deleted sentence pairs with a similarity variance of 1.0 or higher, resulting in the collection of 10,236 sentence pairs with gold similarity. We refer to this collected data as JSTS-A.
> 2. To collect sentence pairs with low similarity, we crowdsource the same STS task as Step 1 using sentence pairs of captions for different images. We conducted this task on 4,000 sentence pairs and collected 2,970 sentence pairs with gold similarity. We refer to this collected data as JSTS-B.
> 3. For JSTS-A, we crowdsource an NLI task. Since inference relations are directional, we obtain inference relations in both directions for sentence pairs. As mentioned earlier,it is difficult to collect instances of contradiction from JSTS-A, which was collected from the captions of the same images,and thus we collect instances of entailment and neutral in this step. We collect inference relation answers from 10 workers. If six or more people give the same answer, we adopt it as the gold label if it is entailment or neutral. To obtain inference relations in both directions for JSTS-A, we performed this task on 20,472 sentence pairs, twice as many as JSTS-A. As a result, we collected inference relations for 17,501 sentence pairs. We refer to this collected data as JNLI-A. We do not use JSTS-B for the NLI task because it is difficult to define and determine the inference relations between captions of different images.
> 4. To collect NLI instances of contradiction, we crowdsource a task of writing four contradictory sentences for each caption in YJCaptions. From the written sentences, we remove sentence pairs with an edit distance of 0.75 or higher to remove low-quality sentences, such as short sentences and sentences with low relevance to the original sentence. Furthermore, we perform a one-way NLI task with 10 workers to verify whether the created sentence pairs are contradictory. Only the sentence pairs answered as contradiction by at least six workers are adopted. Finally,since the contradiction relation has no direction, we automatically assign contradiction in the opposite direction of the adopted sentence pairs. Using 1,800 captions, we acquired 7,200 sentence pairs, from which we collected 3,779 sentence pairs to which we assigned the one-way contradiction relation.By automatically assigning the contradiction relation in the opposite direction, we doubled the number of instances to 7,558. We refer to this collected data as JNLI-C.
> 5. For the 3,779 sentence pairs collected in Step 4, we crowdsource an STS task, assigning similarity  and filtering in the same way as in Steps1 and 2. In this way, we collected 2,303 sentence pairs with gold similarity from 3,779 pairs. We refer to this collected data as JSTS-C.

##### JSQuAD

From [JGLUE's paper](https://aclanthology.org/2022.lrec-1.317/):

> As QA datasets, we build a Japanese version of SQuAD (Rajpurkar et al., 2016), one of the datasets of reading comprehension, and a Japanese version ofCommonsenseQA, which is explained in the next section.

> Reading comprehension is the task of reading a document and answering questions about it. Many reading comprehension evaluation sets have been built in English, followed by those in other languages or multilingual ones.

> In Japanese, reading comprehension datasets for quizzes (Suzukietal.,2018) and those in the drivingdomain (Takahashi et al., 2019) have been built, but none are in the general domain. We use Wikipedia to build a dataset for the general domain. The construction process is basically based on SQuAD 1.1 (Rajpurkar et al., 2016). 

> First, to extract high-quality articles from Wikipedia, we use Nayuki, which estimates the quality of articles on the basis of hyperlinks in Wikipedia. We randomly chose 822 articles from the top-ranked 10,000 articles. For example, the articles include “熊本県 (Kumamoto Prefecture)” and “フランス料理 (French cuisine)”. Next, we divide an article into paragraphs, present each paragraph to crowdworkers, and ask them to write questions and answers that can be answered if one understands the paragraph. Figure 2 shows an example of JSQuAD. We ask workers to write two additional answers for the dev and test sets to make the system evaluation robust.

##### JCommonsenseQA

From [JGLUE's paper](https://aclanthology.org/2022.lrec-1.317/):

> ### Overview
> JCommonsenseQA is a Japanese version of CommonsenseQA (Talmor et al., 2019), which consists of five choice QA to evaluate commonsense reasoning ability. Figure 3 shows examples of JCommonsenseQA. In the same way as CommonsenseQA, JCommonsenseQA is built using crowdsourcing with seeds extracted from the knowledge base ConceptNet (Speer et al., 2017). ConceptNet is a multilingual knowledge base that consists of triplets of two concepts and their relation. The triplets are directional and represented as (source concept, relation, target concept), for example (bullet train, AtLocation, station).

> ### Method of Construction
> The construction flow for JCommonsenseQA is shown in Figure 4. First, we collect question sets (QSs) from ConceptNet, each of which consists of a source concept and three target concepts that have the same relation to the source concept. Next, for each QS, we crowdAtLocation 2961source a task of writing a question with only one target concept as the answer and a task of adding two distractors. We describe the detailed construction procedure for JCommonsenseQA below, showing how it differs from CommonsenseQA.

> 1. We collect Japanese QSs from ConceptNet. CommonsenseQA uses only forward relations (source concept, relation, target concept) excluding general ones such as “RelatedTo” and “IsA”. JCommonsenseQA similarly uses a set of 22 relations5, excluding general ones, but the direction of the relations is bidirectional to make the questions more diverse. In other words, we also use relations in the opposite direction (source concept, relation−1, target concept).6 With this setup, we extracted 43,566 QSs with Japanese source/target concepts and randomly selected 7,500 from them.
> 2. Some low-quality questions in CommonsenseQA contain distractors that can be considered to be an answer. To improve the quality of distractors, we add the following two processes that are not adopted in CommonsenseQA. First, if three target concepts of a QS include a spelling variation or a synonym of one another, this QS is removed. To identify spelling variations, we use the word ID of the morphological dictionary Juman Dic7. Second, we crowdsource a task of judging whether target concepts contain a synonym. As a result, we adopted 5,920 QSs from 7,500.
> 3. For each QS, we crowdsource a task of writing a question sentence in which only one from the three target concepts is an answer. In the example shown in Figure 4, “駅 (station)” is an answer, and the others are distractors. To remove low quality question sentences, we remove the following question sentences.
>    - Question sentences that contain a choice word(this is because such a question is easily solved).
>    - Question sentences that contain the expression “XX characters”.8 (XX is a number).
>    - Improperly formatted question sentences that do not end with “?”.
>    - As a result, 5,920 × 3 = 17,760question sentences were created, from which we adopted 15,310 by removing inappropriate question sentences.
> 4. In CommonsenseQA, when adding distractors, one is selected from ConceptNet, and the other is created by crowdsourcing. In JCommonsenseQA, to have a wider variety of distractors, two distractors are created by crowdsourcing instead of selecting from ConceptNet. To improve the quality of the questions9, we remove questions whose added distractors fall into one of the following categories:
>    - Distractors are included in a question sentence.
>    - Distractors overlap with one of existing choices.
>    - As a result, distractors were added to the 15,310 questions, of which we adopted 13,906.
> 5. We asked three crowdworkers to answer each question and adopt only those answered correctly by at least two workers. As a result, we adopted 11,263 out of the 13,906 questions.

#### Who are the annotators?

From [JGLUE's README.md](https://github.com/yahoojapan/JGLUE/blob/main/README.md#tasksdatasets):

> We use Yahoo! Crowdsourcing for all crowdsourcing tasks in constructing the datasets.

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

From [JGLUE's paper](https://aclanthology.org/2022.lrec-1.317/):

> We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

- 日本語言語理解ベンチマーク JGLUE の構築 〜 自然言語処理モデルの評価用データセットを公開しました - Yahoo! JAPAN Tech Blog https://techblog.yahoo.co.jp/entry/2022122030379907/ 

### Dataset Curators

#### MARC-ja

- Keung, Phillip, et al. "The Multilingual Amazon Reviews Corpus." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020.

#### JCoLA

- 染谷ら，"日本語版 CoLA の構築，" 言語処理学会第 28 回年次大会，2022.

#### JSTS and JNLI

- Miyazaki, Takashi, and Nobuyuki Shimizu. "Cross-lingual image caption generation." Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2016.

#### JSQuAD

The JGLUE's 'authors curated the original data for JSQuAD from the Japanese wikipedia dump.

#### JCommonsenseQA

In the same way as CommonsenseQA, JCommonsenseQA is built using crowdsourcing with seeds extracted from the knowledge base ConceptNet

### Licensing Information

#### JGLUE

From [JGLUE's README.md'](https://github.com/yahoojapan/JGLUE#license):

> This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.

#### JCoLA

From [JCoLA's README.md'](https://github.com/osekilab/JCoLA#license):

> The text in this corpus is excerpted from the published works, and copyright (where applicable) remains with the original authors or publishers. We expect that research use within Japan is legal under fair use, but make no guarantee of this.

### Citation Information

#### JGLUE

```bibtex
@inproceedings{kurihara-lrec-2022-jglue,
  title={JGLUE: Japanese general language understanding evaluation},
  author={Kurihara, Kentaro and Kawahara, Daisuke and Shibata, Tomohide},
  booktitle={Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  pages={2957--2966},
  year={2022},
  url={https://aclanthology.org/2022.lrec-1.317/}
}
```

```bibtex
@inproceedings{kurihara-nlp-2022-jglue,
  title={JGLUE: 日本語言語理解ベンチマーク},
  author={栗原健太郎 and 河原大輔 and 柴田知秀},
  booktitle={言語処理学会第 28 回年次大会},
  pages={2023--2028},
  year={2022},
  url={https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E8-4.pdf},
  note={in Japanese}
}
```

#### MARC-ja

```bibtex
@inproceedings{marc_reviews,
  title={The Multilingual Amazon Reviews Corpus},
  author={Keung, Phillip and Lu, Yichao and Szarvas, György and Smith, Noah A.},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year={2020}
}
```

#### JCoLA

```bibtex
@inproceedings{someya-nlp-2022-jcola,
  title={日本語版 CoLA の構築},
  author={染谷 大河 and 大関 洋平},
  booktitle={言語処理学会第 28 回年次大会},
  pages={1872--1877},
  year={2022},
  url={https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/E7-1.pdf},
  note={in Japanese}
}
```

### Contributions

Thanks to [Kentaro Kurihara](https://twitter.com/kkurihara_cs), [Daisuke Kawahara](https://twitter.com/daisukekawahar1), and [Tomohide Shibata](https://twitter.com/stomohide) for creating JGLUE dataset.
Thanks to [Taiga Someya](https://twitter.com/T0a8i0g9a) for creating JCoLA dataset.
