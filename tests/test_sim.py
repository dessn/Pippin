from pippin.snana_sim import SNANASimulation
from tests.utils import get_manager


def test_dataprep_config_valid():

    # This shouldn't raise an error
    manager = get_manager(yaml="config_files/sim.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 1
    task = tasks[0]

    assert isinstance(task, SNANASimulation)


def test_dataprep_outputs_set():
    manager = get_manager(yaml="config_files/sim.yml", check=True)
    tasks = manager.tasks
    task = tasks[0]

    # Check properties and outputs all set correctly
    assert task.name == "EXAMPLESIM"
    assert task.output["genversion"] == "PIP_SIM_EXAMPLESIM"
    assert task.output["ranseed_change"]
    assert not task.output["blind"]
    assert len(task.dependencies) == 0
