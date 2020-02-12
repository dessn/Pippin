from pippin.classifiers.fitprob import FitProbClassifier
from pippin.classifiers.perfect import PerfectClassifier
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from tests.utils import get_manager


def test_classify_sim_only_config_valid():

    # This shouldn't raise an error
    manager = get_manager(yaml="config_files/classify_sim.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 2
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], PerfectClassifier)

    task = tasks[-1]
    assert task.output["name"] == "PERFECT"
    assert task.output["prob_column_name"] == "PROB_PERFECT"
    assert len(task.dependencies) == 1


def test_classifier_lcfit_config_valid():
    manager = get_manager(yaml="config_files/classify_lcfit.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 3
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], SNANALightCurveFit)
    assert isinstance(tasks[2], FitProbClassifier)

    task = tasks[-1]

    # Check properties and outputs all set correctly
    assert task.name == "FITPROBTEST"
    assert task.output["prob_column_name"] == "PROB_FITPROBTEST_D_ASIM"
    assert len(task.dependencies) == 2
