import datasets as ds
import pytest

# In datasets>=3.0.0, HF_DATASETS_TRUST_REMOTE_CODE defaults to False,
# which triggers confirmation dialogs when loading datasets and interrupts testing.
# Therefore, HF_DATASETS_TRUST_REMOTE_CODE is set to True.
ds.config.HF_DATASETS_TRUST_REMOTE_CODE = True


@pytest.fixture
def dataset_path() -> str:
    return "JGLUE.py"


@pytest.mark.parametrize(
    argnames="dataset_name, expected_num_train, expected_num_valid, expected_num_test",
    argvalues=(
        ("JSTS", 12451, 1457, 1589),
        ("JNLI", 20073, 2434, 2508),
        ("JSQuAD", 62697, 4442, 4420),
        ("JCommonsenseQA", 8939, 1119, 1118),
    ),
)
def test_load_dataset(
    dataset_path: str,
    dataset_name: str,
    expected_num_train: int,
    expected_num_valid: int,
    expected_num_test: int,
):
    dataset = ds.load_dataset(path=dataset_path, name=dataset_name)
    assert isinstance(dataset, ds.DatasetDict)

    # For the expected number of train, valid, and test datasets, refer to the JGLUE README
    # ref. https://github.com/yahoojapan/JGLUE#tasksdatasets
    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid
    assert dataset["test"].num_rows == expected_num_test


def test_load_marc_ja(
    dataset_path: str,
    dataset_name: str = "MARC-ja",
    expected_num_train: int = 187528,
    expected_num_valid: int = 5654,
    expected_num_test: int = -1,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=dataset_name,
        is_pos_neg=True,
        max_char_length=500,
        filter_review_id_list_valid=True,
        label_conv_review_id_list_valid=True,
    )
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid
    assert dataset["test"].num_rows == expected_num_test


def test_load_jcola(
    dataset_path: str,
    dataset_name: str = "JCoLA",
    expected_num_train: int = 6919,
    expected_num_valid: int = 865,
    expected_num_valid_ood: int = 685,
):
    dataset = ds.load_dataset(path=dataset_path, name=dataset_name)
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid
    assert dataset["validation_out_of_domain"].num_rows == expected_num_valid_ood
    assert (
        dataset["validation_out_of_domain_annotated"].num_rows == expected_num_valid_ood
    )


def test_jglue_version():
    import tomli

    from JGLUE import JGLUE

    jglue_version = JGLUE.JGLUE_VERSION
    jglue_major, jglue_minor, _ = jglue_version.tuple

    with open("pyproject.toml", "rb") as rf:
        pyproject_toml = tomli.load(rf)

    project_version = ds.Version(pyproject_toml["project"]["version"])
    proj_major, proj_minor, _ = project_version.tuple

    assert jglue_major == proj_major and jglue_minor == proj_minor, (
        f"JGLUE and project version mismatch: {jglue_version=} != {project_version=}"
    )
