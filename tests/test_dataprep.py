from pippin.dataprep import DataPrep
from tests.utils import get_manager


def test_dataprep_config_valid():

    # This shouldn't raise an error
    manager = get_manager(yaml="config_files/dataprep.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 1
    task = tasks[0]

    assert isinstance(task, DataPrep)


def test_dataprep_outputs_set():
    manager = get_manager(yaml="config_files/dataprep.yml", check=True)
    tasks = manager.tasks
    task = tasks[0]

    # Check properties and outputs all set correctly
    assert task.name == "LABEL"
    assert not task.output["blind"]
    assert task.output["is_sim"]
    assert task.output["raw_dir"] == "somedir"
    assert len(task.dependencies) == 0
