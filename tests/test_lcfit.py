from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from tests.utils import get_manager


def test_lcfit_config_valid():

    # This shouldn't raise an error
    manager = get_manager(yaml="config_files/lcfit.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 2
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], SNANALightCurveFit)


def test_lcfit_outputs_set():
    manager = get_manager(yaml="config_files/lcfit.yml", check=True)
    tasks = manager.tasks
    task = tasks[-1]

    # Check properties and outputs all set correctly
    assert task.name == "DIFFERENT_SN_EXAMPLESIM"
    assert task.output["genversion"] == "PIP_LCFIT_EXAMPLESIM"
    assert task.output["sim_name"] == "EXAMPLESIM"
    assert not task.output["is_data"]
    assert not task.output["blind"]
    assert len(task.dependencies) == 1


def test_lcfit_outputs_mask():
    manager = get_manager(yaml="config_files/lcfit_mask.yml", check=True)
    tasks = manager.tasks
    assert len(tasks) == 5

    expected = {"DIFFERENT_SN_EXAMPLESIM", "DIFFERENT_SN_EXAMPLESIM2", "MASKTEST_EXAMPLESIM2"}
    found = set([t.name for t in tasks if isinstance(t, SNANALightCurveFit)])

    assert expected == found
