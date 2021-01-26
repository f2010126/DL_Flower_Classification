import os
import pickle
import time
import logging
logging.basicConfig(level=logging.DEBUG)
from pathlib import Path
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from bohb_pytorch_worker import PyTorchWorker as worker


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
