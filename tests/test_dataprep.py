from tests.utils import get_manager


def test_dataprep_config_valid():
    manager = get_manager(yaml="config_files/dataprep_config.yml")
