import os

from pippin.analyse import AnalyseChains
from tests.utils import get_manager
from pippin.dataprep import DataPrep
from pippin.snana_sim import SNANASimulation
from pippin.snana_fit import SNANALightCurveFit
from pippin.classifiers.fitprob import FitProbClassifier
from pippin.classifiers.perfect import PerfectClassifier
from pippin.classifiers.scone import SconeClassifier
from pippin.classifiers.scone_legacy import SconeLegacyClassifier
from pippin.aggregator import Aggregator
from pippin.merge import Merger
from pippin.biascor import BiasCor
from pippin.create_cov import CreateCov
from pippin.cosmofitters.cosmofit import CosmoFit
from pippin.cosmofitters.cosmomc import CosmoMC


def test_dataprep_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="tests/config_files/valid_dataprep.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 1
    task = tasks[0]

    assert isinstance(task, DataPrep)


def test_dataprep_outputs_set():
    manager = get_manager(yaml="tests/config_files/valid_dataprep.yml", check=True)
    tasks = manager.tasks
    task = tasks[0]

    # Check properties and outputs all set correctly
    assert task.name == "LABEL"
    assert not task.output["blind"]
    assert task.output["is_sim"]
    assert os.path.basename(task.output["raw_dir"]) == "surveys"
    assert len(task.dependencies) == 0


def test_sim_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="tests/config_files/valid_sim.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 1
    task = tasks[0]

    assert isinstance(task, SNANASimulation)


def test_sim_outputs_set():
    manager = get_manager(yaml="tests/config_files/valid_sim.yml", check=True)
    tasks = manager.tasks
    task = tasks[0]

    # Check properties and outputs all set correctly
    assert task.name == "EXAMPLESIM"
    assert task.output["genversion"] == "PIP_VALID_SIM_EXAMPLESIM"
    assert task.output["ranseed_change"]
    assert not task.output["blind"]
    assert len(task.dependencies) == 0


def test_lcfit_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="tests/config_files/valid_lcfit.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 2
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], SNANALightCurveFit)


def test_lcfit_outputs_set():
    manager = get_manager(yaml="tests/config_files/valid_lcfit.yml", check=True)
    tasks = manager.tasks
    task = tasks[-1]

    # Check properties and outputs all set correctly
    assert task.name == "DIFFERENT_SN_EXAMPLESIM"
    assert task.output["genversion"] == "PIP_VALID_LCFIT_EXAMPLESIM"
    assert task.output["sim_name"] == "EXAMPLESIM"
    assert not task.output["is_data"]
    assert not task.output["blind"]
    assert len(task.dependencies) == 1


def test_lcfit_outputs_mask():
    manager = get_manager(yaml="tests/config_files/valid_lcfit_mask.yml", check=True)
    tasks = manager.tasks
    assert len(tasks) == 5

    expected = {
        "DIFFERENT_SN_EXAMPLESIM",
        "DIFFERENT_SN_EXAMPLESIM2",
        "MASKTEST_EXAMPLESIM2",
    }
    found = set([t.name for t in tasks if isinstance(t, SNANALightCurveFit)])

    assert expected == found


def test_classify_sim_only_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="tests/config_files/valid_classify_sim.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 2
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], PerfectClassifier)

    task = tasks[-1]
    assert task.output["name"] == "PERFECT"
    assert task.output["prob_column_name"] == "PROB_PERFECT"
    assert len(task.dependencies) == 1


def test_classifier_lcfit_config_valid():
    manager = get_manager(
        yaml="tests/config_files/valid_classify_lcfit.yml", check=True
    )
    tasks = manager.tasks

    assert len(tasks) == 3
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], SNANALightCurveFit)
    assert isinstance(tasks[2], FitProbClassifier)

    task = tasks[-1]

    # Check properties and outputs all set correctly
    assert task.name == "FITPROBTEST"
    assert task.output["prob_column_name"] == "PROB_FITPROBTEST"
    assert len(task.dependencies) == 2


def test_classifier_sim_with_opt_lcfit_config_valid():
    manager = get_manager(
        yaml="tests/config_files/valid_classify_sim_with_lcfit.yml", check=True
    )
    tasks = manager.tasks

    assert len(tasks) == 3
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], SNANALightCurveFit)
    assert isinstance(tasks[2], PerfectClassifier)

    task = tasks[-1]
    assert task.name == "PERFECT"
    assert task.output["prob_column_name"] == "PROB_PERFECT"
    deps = task.dependencies
    assert len(deps) == 2
    assert isinstance(deps[0], SNANASimulation)
    assert isinstance(deps[1], SNANALightCurveFit)


def test_classifier_scone_valid():
    manager = get_manager(
        yaml="tests/config_files/valid_classify_scone.yml", check=True
    )
    tasks = manager.tasks

    # 1 Sim, 1 LCFit, 4 Scone
    assert len(tasks) == 6
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], SNANALightCurveFit)
    for task in tasks[2:]:
        # isinstance => Class or Subclass
        assert isinstance(task, SconeClassifier)

    tests = [
        {
            "task": tasks[2],
            "cls": SconeLegacyClassifier,
            "attr": {"name": "LEGACY_SCONE_TRAIN", "scone_input_file": None},
        },
        {
            "task": tasks[3],
            "cls": SconeLegacyClassifier,
            "attr": {"name": "LEGACY_SCONE_PREDICT", "scone_input_file": None},
        },
        {
            "task": tasks[4],
            "cls": SconeClassifier,
            "attr": {
                "name": "SCONE_TRAIN",
            },
        },
        {
            "task": tasks[5],
            "cls": SconeClassifier,
            "attr": {
                "name": "SCONE_PREDICT",
            },
        },
    ]

    for test in tests:
        task = test["task"]
        assert type(task) is test["cls"]
        for attr, val in test["attr"].items():
            assert hasattr(task, attr)
            assert getattr(task, attr) == val


def test_agg_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="tests/config_files/valid_agg.yml", check=True)
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


def test_merge_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="tests/config_files/valid_merge.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 6
    assert isinstance(tasks[0], SNANASimulation)
    assert isinstance(tasks[1], SNANALightCurveFit)
    assert isinstance(tasks[2], FitProbClassifier)
    assert isinstance(tasks[3], PerfectClassifier)
    assert isinstance(tasks[4], Aggregator)
    assert isinstance(tasks[5], Merger)

    task = tasks[-1]
    assert task.output["name"] == "MERGE_D_ASIM"
    assert task.output["sim_name"] == "ASIM"
    assert task.output["lcfit_name"] == "D_ASIM"
    assert not task.output["blind"]
    assert task.output["classifier_names"] == ["FITPROBTEST", "PERFECT"]
    assert len(task.dependencies) == 2


def test_biascor_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="tests/config_files/valid_biascor.yml", check=True)
    tasks = manager.tasks

    assert len(tasks) == 13  # (2 sims, 2 lcfit, 4 classifiers, 2 agg, 2 merge, 1 bcor)
    assert isinstance(tasks[-1], BiasCor)

    task = tasks[-1]
    assert task.output["name"] == "BCOR"
    assert task.output["blind"]
    assert len(task.dependencies) == 3  # Two merge + classifier


def test_createcov_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="tests/config_files/valid_create_cov.yml", check=True)
    tasks = manager.tasks

    assert (
        len(tasks) == 14
    )  # (2 sims, 2 lcfit, 4 classifiers, 2 agg, 2 merge, 1 bcor, 1 create cov)
    assert isinstance(tasks[-1], CreateCov)

    task = tasks[-1]
    assert task.output["name"] == "COVTEST"
    assert len(task.output["covopts"]) == 2
    assert "ALL" in task.output["covopts"]
    assert "NOSYS" in task.output["covopts"]
    assert task.output["blind"]
    assert len(task.dependencies) == 1  # Biascor only


def test_cosmomc_config_valid():
    # This shouldn't raise an error
    manager = get_manager(yaml="tests/config_files/valid_cosmomc.yml", check=True)
    tasks = manager.tasks

    assert (
        len(tasks) == 15
    )  # (2 sims, 2 lcfit, 4 classifiers, 2 agg, 2 merge, 1 bcor, 1 create cov, 1 cosmomc)
    assert isinstance(tasks[-1], CosmoMC)

    task = tasks[-1]
    assert task.output["name"] == "COSMOMC_SN_OMW_COVTEST"
    assert len(task.output["covopts"]) == 2
    assert "ALL" in task.output["covopts"]
    assert "NOSYS" in task.output["covopts"]
    assert task.output["blind"]
    assert "w" in task.output["cosmology_params"]
    assert "omegam" in task.output["cosmology_params"]
    assert len(task.dependencies) == 1  # Create cov only


def test_analyse_config_valid():
    manager = get_manager(yaml="tests/config_files/valid_analyse.yml", check=True)
    tasks = manager.tasks

    assert (
        len(tasks) == 16
    )  # (2 sims, 2 lcfit, 4 classifiers, 2 agg, 2 merge, 1 bcor, 1 create cov, 1 cosmomc, 1 analyse)
    assert isinstance(tasks[-1], AnalyseChains)

    task = tasks[-1]
    assert task.output["name"] == "ALL_OMW"
    assert (
        len(task.dependencies) == 2
    )  # Create cosmomc for chains, and biascor for hubble diagram
    assert isinstance(task.dependencies[0], (CosmoMC, BiasCor))
    assert isinstance(task.dependencies[1], (CosmoMC, BiasCor))
