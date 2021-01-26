import os
import pickle
import torch.nn as nn
import logging
from pathlib import Path
from typing import Tuple, Dict
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import numpy as np
from bohb_pytorch_worker import PyTorchWorker as worker
logging.basicConfig(level=logging.DEBUG)


def save_result(filename: str, obj: object) -> None:
    """Save object to disk as pickle file.

    Args:
        filename: Name of file in ./results directory to write object to.
        obj: The object to write to file.

    """
    # make sure save directory exists
    save_path = Path("results")
    os.makedirs(save_path, exist_ok=True)

    # save the python objects as bytes
    with (save_path / f"{filename}.pkl").open('wb') as fh:
        pickle.dump(obj, fh)

def load_result(filename: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load object from pickled file.

    Args:
        filename: Name of file in ./results directory to load.

    """
    with (Path("results") / f"{filename}.pkl").open('rb') as fh:
        return pickle.load(fh)

def best_model_bohb(results: hpres.Result) -> Tuple[float, int, nn.Module]:
    """ Compute the model of the best run, evaluated on the largest budget,
        with it's final validation error.

    Args:
        result: Hpbandster result object.

    Returns:
        best error, best configuration id and best configuration

    """
    inc_id = results.get_incumbent_id()  # get config_id of incumbent (lowest loss)
    # START TODO #################
    id2conf = results.get_id2config_mapping()
    inc_runs = results.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    best_error = inc_run.loss
    best_configuration = id2conf[inc_id]['config']
    # END TODO ###################

    return best_error, inc_id, best_configuration

def evaluate(results: hpres.Result) -> None:
    """Evaluate the results from the bohb run.

    Args:
        result: Hpbandster structure results

    Returns:
        None
    """
    # Look for the best model and print it
    best_error, best_config_id, best_config = best_model_bohb(results)
    print("The best model (config_id {}) has the lowest final error with {:.4f}."
          .format(best_config_id, best_error))
    print(f"The best configuration {best_config}")




def setup_bohb():
    # TODO: get it to look like the assignment
    run_id = 'bohb_template'
    # Every process has to lookup the hostname
    host = 'localhost'
    working_dir = os.curdir
    min_budget = 10  # Minimum number of epochs for training
    max_budget = 50  # Maximum number of epochs for training.
    n_iterations = 16  # Number of iterations performed by the optimizer
    port = 0
    # Start a nameserver #####
    ns = hpns.NameServer(run_id=run_id, host=host, port=port,
                         working_directory=working_dir)
    ns_host, ns_port = ns.start()
    result_logger = hpres.json_result_logger(directory=working_dir, overwrite=False)
    bohb = BOHB(configspace=worker.get_configspace(),
                run_id=run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                result_logger=result_logger,
                min_budget=min_budget, max_budget=max_budget,
                )
    try:
        # Start local worker
        w = worker(run_id=run_id, host=host, nameserver=ns_host,
                   nameserver_port=ns_port, timeout=120)
        w.run(background=True)
        # Run an optimize
        res = bohb.run(n_iterations=n_iterations)
        save_result('bohb_result', res)

    finally:
        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()


if __name__ == '__main__':
    setup_bohb()
    results = load_result('bohb_result')
    evaluate(results)

