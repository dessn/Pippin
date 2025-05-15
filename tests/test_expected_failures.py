import pytest

from tests.utils import get_manager


def test_dataprep_no_opts_fail() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_data1.yml", check=True)


def test_dataprep_no_dir_found() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_data2.yml", check=True)


def test_sim_no_config_found() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_sim1.yml", check=True)


def test_sim_no_sim_comp() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_sim2.yml", check=True)


def test_lcfit_no_base() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_lcfit1.yml", check=True)


def test_lcfit_base_not_found() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_lcfit2.yml", check=True)


def test_lcfit_mask_matches_nothing() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_lcfit3.yml", check=True)


def test_classify_needs_mode() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_classify1.yml", check=True)


def test_classify_mask_matches_nothing() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_classify2.yml", check=True)


def test_classify_predict_doesnt_specify_train() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_classify3.yml", check=True)


def test_classify_trained_model_not_found_nn() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_classify4.yml", check=True)


def test_classify_trained_model_not_found_snirf() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_classify5.yml", check=True)


def test_classify_trained_model_not_found_snn() -> None:
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_classify6.yml", check=True)
