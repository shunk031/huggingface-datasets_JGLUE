import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "JGLUE.py"


@pytest.mark.parametrize(
    argnames="dataset_name, expected_num_train, expected_num_valid,",
    argvalues=(
        ("JSTS", 12451, 1457),
        ("JNLI", 20073, 2434),
        ("JSQuAD", 62859, 4442),
        ("JCommonsenseQA", 8939, 1119),
    ),
)
def test_load_dataset(
    dataset_path: str,
    dataset_name: str,
    expected_num_train: int,
    expected_num_valid: int,
):
    dataset = ds.load_dataset(path=dataset_path, name=dataset_name)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid


def test_load_marc_ja(
    dataset_path: str,
    dataset_name: str = "MARC-ja",
    expected_num_train: int = 187528,
    expected_num_valid: int = 5654,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=dataset_name,
        is_pos_neg=True,
        max_char_length=500,
        filter_review_id_list_valid=True,
        label_conv_review_id_list_valid=True,
    )

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid


def test_load_jcola(
    dataset_path: str,
    dataset_name: str = "JCoLA",
    expected_num_train: int = 6919,
    expected_num_valid: int = 865,
    expected_num_valid_ood: int = 685,
):
    dataset = ds.load_dataset(path=dataset_path, name=dataset_name)
    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid
    assert dataset["validation_out_of_domain"].num_rows == expected_num_valid_ood
