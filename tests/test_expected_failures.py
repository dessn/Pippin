from tests.utils import get_manager
import pytest


def test_dataprep_no_opts_fail():
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_data1.yml", check=True)


def test_dataprep_no_dir_found():
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_data2.yml", check=True)


def test_sim_no_config_found():
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_sim1.yml", check=True)


def test_sim_no_sim_comp():
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_sim2.yml", check=True)


def test_lcfit_no_base():
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_lcfit1.yml", check=True)


def test_lcfit_base_not_found():
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_lcfit2.yml", check=True)


def test_lcfit_mask_matches_nothing():
    with pytest.raises(ValueError):
        get_manager(yaml="tests/config_files/fail_lcfit3.yml", check=True)
