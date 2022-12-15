# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates file with solutions to the training instances. Needs to be run once #
# before training.                                                              #                                                                     #
# Usage:                                                                        #
# python 02_get_instance_solutions.py <type> -j <njobs> -n <ninstances>         #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import glob
import json
import sys
import argparse
import threading
import queue

import ecole


class OptimalSol:
    def __init__(self):
        pass

    def before_reset(self, model):
        pass

    def extract(self, model, done=False):
        if not done:
            return None

        pyscipopt_model = model.as_pyscipopt()
        return pyscipopt_model.getObjVal(original=True)


def solve_instance(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which instances are received.
    out_queue : queue.Queue
        Output queue in which to solution.
    """
    reward_fun = OptimalSol()
    while not in_queue.empty():
        instance = in_queue.get()
        env = ecole.environment.Configuring(
            scip_params={}, observation_function=None, reward_function=reward_fun
        )
        env.reset(str(instance))
        print(f"Solving {instance}")
        _, _, solution, _, _ = env.step({})
        out_queue.put({instance: solution})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "problem",
        help="MILP instance type to process.",
        choices=["setcover", "cauctions", "ufacilities", "indset", "mknapsack"],
    )
    parser.add_argument(
        "-j",
        "--njobs",
        help="Number of parallel jobs.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-n",
        "--ninst",
        help="Number of instances to solve.",
        type=int,
        default=10000,
    )
    args = parser.parse_args()

    if args.problem == "setcover":
        nrows, ncols, dens = 500, 1000, 0.05
        instance_dir = f"data/instances/setcover/train_{nrows}r_{ncols}c_{dens}d"
        instances = glob.glob(instance_dir + "/*.lp")

    elif args.problem == "cauctions":
        number_of_items, number_of_bids = 100, 500
        instance_dir = (
            f"data/instances/cauctions/train_{number_of_items}_{number_of_bids}"
        )
        instances = glob.glob(instance_dir + "/*.lp")

    elif args.problem == "indset":
        number_of_nodes, affinity = 500, 4
        instance_dir = f"data/instances/indset/train_{number_of_nodes}_{affinity}"
        instances = glob.glob(instance_dir + "/*.lp")

    elif args.problem == "ufacilities":
        number_of_customers, number_of_facilities, ratio = 100, 100, 5
        instance_dir = f"data/instances/ufacilities/train_{number_of_customers}_{number_of_facilities}_{ratio}"
        instances = glob.glob(instance_dir + "/*.lp")

    elif args.problem == "mknapsack":
        number_of_items, number_of_knapsacks = 100, 6
        instance_dir = (
            f"data/instances/mknapsack/train_{number_of_items}_{number_of_knapsacks}"
        )
        instances = glob.glob(instance_dir + "/*.lp")

    else:
        raise NotImplementedError

    num_inst = min(args.ninst, len(instances))
    orders_queue = queue.Queue()
    answers_queue = queue.Queue()
    for instance in instances[:num_inst]:
        orders_queue.put(instance)
    print(f"{num_inst} instances on queue.")

    workers = []
    for i in range(args.njobs):
        p = threading.Thread(
            target=solve_instance, args=(orders_queue, answers_queue), daemon=True
        )
        workers.append(p)
        p.start()

    i = 0
    solutions = {}
    while i < num_inst:
        answer = answers_queue.get()
        solutions.update(answer)
        i += 1

    with open(instance_dir + "/instance_solutions.json", "w") as f:
        json.dump(solutions, f)

    for p in workers:
        assert not p.is_alive()
