from pippin.aggregator import Aggregator
from pippin.biascor import BiasCor
from pippin.classifiers.fitprob import FitProbClassifier
from pippin.classifiers.perfect import PerfectClassifier
from pippin.merge import Merger
from pippin.snana_fit import SNANALightCurveFit
from pippin.snana_sim import SNANASimulation
from tests.utils import get_manager


def test_merge_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="config_files/biascor.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 13  # (2 sims, 2 lcfit, 4 classifiers, 2 agg, 2 merge, 1 bcor)
    assert isinstance(tasks[-1], BiasCor)

    task = tasks[-1]
    assert task.output["name"] == "BCOR"
    assert task.output["blind"]
    assert len(task.dependencies) == 3  # Two merge + classifier
