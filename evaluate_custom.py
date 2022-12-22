import os
import argparse
import csv
import numpy as np
from time import perf_counter, process_time, strftime

import ecole
from ecole.scip import Model
import glob

from numpy.random import SeedSequence
import torch
from torch import from_numpy
from typing import Iterable, NamedTuple, Callable

from actor.actor import GNNPolicy

from threading import Thread, Event
from queue import Queue, Empty, Full
from functools import partial


class Task(NamedTuple):
    file: str
    j: int
    sk: SeedSequence


def head(
    ss: SeedSequence,
    folder: str,
    filter: str = "*.mps",
    n_replications: int = 5,
) -> Iterable[Task]:
    """Evaluation task source"""
    folder = os.path.abspath(folder)
    files = glob.glob(os.path.join(folder, filter), recursive=False)
    for file in sorted(files, key=lambda f: glob.os.path.split(f)[1]):
        for j, sk in enumerate(ss.spawn(n_replications)):
            yield Task(file, j, sk)


def maybe_raise_sigint(m: Model) -> None:
    """Manually check if SCIP encountered a sigint"""
    if m.as_pyscipopt().getStatus() == "userinterrupt":
        raise KeyboardInterrupt from None


def body(policies: list[dict], task: Task) -> dict:
    scip_params = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": 3600,
        "timing/clocktype": 1,  # 1: CPU user seconds, 2: wall clock time
    }

    for pol in policies:
        if pol["type"] == "gcnn":
            ecm, wt, pt = run_branching(
                task.file, task.sk, scip_params, pol["policy"], pol["device"]
            )

        elif pol["type"] == "internal":
            ecm, wt, pt = run_configuring(
                task.file, task.sk, scip_params, pol["policy"]
            )

        maybe_raise_sigint(ecm)

        m = ecm.as_pyscipopt()
        yield {
            "policy": f"{pol['type']}:{pol['name']}",
            "seed": task.j,
            "type": "custom",
            "instance": task.file,
            "nnodes": m.getNNodes(),
            "nlps": m.getNLPs(),
            "stime": m.getSolvingTime(),
            "gap": m.getGap(),
            "status": m.getStatus(),
            "walltime": wt,
            "proctime": pt,
        }


class CTX(NamedTuple):
    errors: Queue
    fin: Event
    rx: Queue
    tx: Queue


def recv(fin: Event, q: Queue, *, timeout: float = 0.5) -> ...:
    # busy-check the termination flag until we receive a job
    while not fin.is_set():
        try:
            return q.get(True, timeout=timeout)

        except Empty:
            continue


def send(fin: Event, q: Queue, item: ..., *, timeout: float = 0.5) -> None:
    # keep regularly checking the termination flag until we receive a value
    while not fin.is_set():
        try:
            return q.put(item, True, timeout=timeout)

        except Full:
            continue


def t_source(it: Iterable, ctx: CTX, timeout: float = 0.5) -> None:
    try:
        while not ctx.fin.is_set():
            send(ctx.fin, ctx.rx, next(it), timeout=timeout)

    except StopIteration:
        pass

    except Exception as e:
        ctx.errors.put_nowait(e)

    finally:
        ctx.fin.set()


def t_relay(fn: Callable, ctx: CTX, timeout: float = 0.5) -> None:
    try:
        while not ctx.fin.is_set():
            input = recv(ctx.fin, ctx.rx, timeout=timeout)
            for output in fn(input):
                send(ctx.fin, ctx.tx, output, timeout=timeout)

    except StopIteration:
        pass

    except Exception as e:
        ctx.errors.put_nowait(e)

    finally:
        ctx.fin.set()


def run_configuring(
    p: str, ss: SeedSequence, scip_params: dict, branchrule: str
) -> tuple[Model, float, float]:
    # Run SCIP's default brancher
    env = ecole.environment.Configuring(
        scip_params={
            **scip_params,
            f"branching/{branchrule}/priority": 9999999,
            # "branching/vanillafullstrong/maxdepth": -1,
            # "branching/vanillafullstrong/maxbounddist": 1,
            # "branching/vanillafullstrong/integralcands": False,
            "branching/vanillafullstrong/idempotent": True,
            # "branching/vanillafullstrong/scoreall": False,
            # "branching/vanillafullstrong/collectscores": False,
            # "branching/vanillafullstrong/donotbranch": False,
        }
    )

    (seed,) = ss.generate_state(1, dtype=np.uint32)
    env.seed(int(seed))

    wt, pt = perf_counter(), process_time()

    env.reset(p)
    _, _, _, _, _ = env.step({})

    wt, pt = perf_counter() - wt, process_time() - pt

    return env.model, wt, pt


def run_branching(
    p: str, ss: SeedSequence, scip_params: dict, policy: GNNPolicy, device: torch.device
) -> tuple[Model, float, float]:
    (seed_1, seed_2) = ss.generate_state(2, dtype=np.uint32)

    # Run the GNN policy
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params=scip_params,
    )

    env.seed(int(seed_1))
    torch.manual_seed(int(seed_2))

    with torch.inference_mode():
        wt, pt = perf_counter(), process_time()

        obs, act_set, _, fin, _ = env.reset(p)
        while not fin:
            edge = obs.edge_features
            logits = (
                policy(
                    from_numpy(obs.row_features.astype(np.float32)).to(device),
                    from_numpy(edge.indices.astype(np.int64)).to(device),
                    from_numpy(edge.values.astype(np.float32).reshape(-1, 1)).to(
                        device
                    ),
                    from_numpy(obs.variable_features.astype(np.float32)).to(device),
                )
                .cpu()
                .numpy()
            )
            act = act_set[logits[act_set].argmax()]
            obs, act_set, _, fin, _ = env.step(act)

        wt, pt = perf_counter() - wt, process_time() - pt

    return env.model, wt, pt


def main(
    prefix: str,
    folder: str,
    filter: str = "*.mps",
    n_replications: int = 5,
    gpu: int = -1,
    entropy: int = None,
    n_jobs: int = 1,
) -> Iterable[dict]:
    device = torch.device("cpu" if gpu < 0 else f"cuda:{gpu}")

    ss = SeedSequence(entropy)
    print(f"{ss.entropy = }")
    tasks = head(ss, folder, filter, n_replications)

    policies = []
    for name in ["il", "mdp", "tmdp+DFS", "tmdp+ObjLim"]:
        policy = GNNPolicy().to(device)
        policy.load_state_dict(torch.load(f"{prefix}{name}.pkl"))
        policies.append(dict(type="gcnn", name=name, policy=policy, device=device))

    for name in ["internal:relpscost", "internal:vanillafullstrong"]:
        type, name = name.split(":")
        policies.append(dict(type=type, name=name, policy=name, device=None))

    evaluate = partial(body, policies)

    if n_jobs < 2:
        for t in tasks:
            yield from evaluate(t)

    else:
        timeout = 1.0

        # spawn source and worker threads
        ctx = CTX(Queue(), Event(), Queue(2 * n_jobs), Queue())
        threads = [Thread(target=t_source, args=(tasks, ctx, timeout), daemon=True)]
        for _ in range(n_jobs):
            threads.append(
                Thread(target=t_relay, args=(evaluate, ctx, timeout), daemon=True)
            )

        # main loop
        try:
            for t in threads:
                t.start()

            while not ctx.fin.is_set():
                yield recv(ctx.fin, ctx.tx, timeout=timeout)

        finally:
            # shutdown and raise any errors
            ctx.fin.set()
            try:
                if not ctx.errors.empty():
                    raise ctx.errors.get()

            finally:
                for t in threads:
                    t.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="Path prefix to GNN policy snapshots")
    parser.add_argument("folder", type=str, help="The target folder")
    parser.add_argument("--filter", type=str, help="The extensions", default="*.mps")
    parser.add_argument("-j", "--n_jobs", help="n_jobs", type=int, default=1)
    parser.add_argument("--entropy", type=int, help="Seed entropy")
    parser.add_argument("-g", "--gpu", help="GPU (-1 for CPU).", type=int, default=-1)
    parser.set_defaults(filter="*.mps", gpu=-1, entropy=None, n_jobs=1)

    args, _ = parser.parse_known_args()

    result_file = f"custom_{strftime('%Y%m%d-%H%M%S')}.csv"
    fieldnames = [
        "policy",
        "seed",
        "type",
        "instance",
        "nnodes",
        "nlps",
        "stime",
        "gap",
        "status",
        "walltime",
        "proctime",
    ]
    os.makedirs("results", exist_ok=True)

    with open(f"results/{result_file}", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for record in main(**vars(args)):
            writer.writerow(record)
            csvfile.flush()

            print(
                "  {type}:{policy} {seed} - {nnodes} nodes {nlps} lps"
                " {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s."
                " {status}".format(**record)
            )
