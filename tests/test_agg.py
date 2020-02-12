from pippin.aggregator import Aggregator
from pippin.classifiers.fitprob import FitProbClassifier
from pippin.classifiers.perfect import PerfectClassifier
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from tests.utils import get_manager


def test_agg_config_valid():

    # This shouldn't raise an error
    manager = get_manager(yaml="config_files/agg.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 5
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], SNANALightCurveFit)
    assert isinstance(tasks[2], FitProbClassifier)
    assert isinstance(tasks[3], PerfectClassifier)
    assert isinstance(tasks[4], Aggregator)

    task = tasks[-1]
    assert task.output["name"] == "AGGLABEL_ASIM"
    assert task.output["sim_name"] == "ASIM"
    assert len(task.dependencies) == 2
