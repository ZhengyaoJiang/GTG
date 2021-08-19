import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from agent.neural_logic.nlmagent import NLMAgent
from agent.neural_logic.kbmlp import KnowledgeBaseMLP
from agent.geometric.gnnagent import GNNAgent
from agent.baselines.cnnagent import CNNAgent
import pandas as pd

from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

from environment.minigrid import *
from gym_minigrid.wrappers import FullyObsWrapper, ReseedWrapper
from typing import List


# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, help="Gym environment.", choices=["breakout", "seaquest", "asterix",
                                                                         "freeway", "space_invaders",
                                                                         "boxworld", "rtfm", "rtfm-onehop",
                                                                         "MiniGrid-LavaCrossingS9N1-v0",
                                                                         "MiniGrid-LavaCrossingS9N2-v0",
                                                                         "MiniGrid-LavaCrossingS9N3-v0",
                                                                         "MiniGrid-LavaCrossingClosed-v0",
                                                                         "blockworld"
                                                                         ])
parser.add_argument("--agent", type=str, default="CNN",
                    choices=["CNN", "NLM", "KBMLP", "GCN"],
                    help="agent type.")
parser.add_argument("--action", type=str, default="move_dir" , choices=["raw", "moveto", "move_dir",
                                                                        "relational", "propositional"])
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--actor_id", default=None)

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=10000000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=4, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=30, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")

# Test settings.
parser.add_argument("--episodes", default=200, type=int)
parser.add_argument("--store_stats", action="store_true")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.001,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable

# NLM Settings
parser.add_argument('--depth', type=int, default=4, help='depth of the logic machine')
parser.add_argument("--output_dims", type=int, default=64)
parser.add_argument("--breath", type=int, default=2)
parser.add_argument("--state", type=str, default="absolute")
parser.add_argument("--binary_op", type=str, default="diff", choices=["square", "diff"])
parser.add_argument("--global_units", type=int, default=0)
parser.add_argument("--activation", type=str, default="relu")
parser.add_argument("--bg_code", type=str, default="b3")

# Env Setting
parser.add_argument("--sticky_prob", type=float, default=0.1)
parser.add_argument("--rand_env", action="store_true")
parser.add_argument("--nb_blocks", default=4, type=int)
parser.add_argument("--disable_wiki", action="store_true")
parser.add_argument("--room_size", default=6, type=int)
parser.add_argument("--goal_length", default=2, type=int)
parser.add_argument("--nb_distractor", default=1, type=int)
parser.add_argument("--distractor_length", default=1, type=int)

# CNN GCN Setting
parser.add_argument("--cnn_code", type=str, default="2gm2f")
parser.add_argument("--embedding_size", type=int, default=64)
parser.add_argument("--mp_rounds", type=int, default=1)

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    flags,
    gym_env,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    actor_buffers: Buffers,
    actor_model_queues: List[mp.SimpleQueue],
    actor_env_queues: List[mp.SimpleQueue]
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = gym_env
        #seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        #gym_env.seed(seed)
        if flags.agent in ["CNN"]:
            env = environment.Environment(gym_env, "image")
        elif flags.agent in ["NLM", "KBMLP", "GCN"]:
            if flags.state in ["relative", "integer", "block"]:
                env = environment.Environment(gym_env, "VKB")
            elif flags.state == "absolute":
                env = environment.Environment(gym_env, "absVKB")
        env_output = env.initial()
        for key in env_output:
            actor_buffers[key][actor_index][0] = env_output[key]
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in actor_buffers:
                buffers[key][index][0] = actor_buffers[key][actor_index][0]

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                actor_model_queues[actor_index].put(actor_index)
                env_info = actor_env_queues[actor_index].get()
                if env_info=="exit":
                    return

                timings.time("model")

                env_output = env.step(actor_buffers["action"][actor_index][0])

                timings.time("step")

                for key in actor_buffers:
                    buffers[key][index][t + 1] = actor_buffers[key][actor_index][0]
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in env_output:
                    actor_buffers[key][actor_index][0] = env_output[key]

                timings.time("write")


            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e

def get_inference_batch(flags, actor_buffers: Buffers):
    indices = list(range(flags.num_actors))
    batch = {
        key: torch.stack([actor_buffers[key][m] for m in indices], dim=1) for key in actor_buffers
    }
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    return batch

def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    timings.time("device")
    return batch


def learn(
    flags,
    model,
    batch,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs = model(batch)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()
        return stats


def create_buffers(flags, obs_shape, num_actions, T, num_buffers, img_shape=None) -> Buffers:
    specs = dict(
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    if flags.agent in ["CNN"]:
        specs["frame"] = dict(size=(T + 1, *obs_shape), dtype=torch.uint8)
    if flags.agent in ["NLM", "KBMLP", "GCN"]:
        specs["frame"] = dict(size=(T + 1, *img_shape), dtype=torch.uint8)
        specs["nullary_tensor"] = dict(size=(T+1, obs_shape[0]), dtype=torch.uint8)
        specs["unary_tensor"] = dict(size=(T+1, *obs_shape[1]), dtype=torch.uint8)
        specs["binary_tensor"] = dict(size=(T+1, *obs_shape[2]), dtype=torch.uint8)
    elif flags.agent == "SNLM":
        specs["frame"] = dict(size=(T + 1, *obs_shape), dtype=torch.uint8)
    if flags.agent in ["NLM", "KBMLP", "SNLM"] and "MiniGrid" in flags.env:
        specs["direction"] = dict(size=(T+1, 4), dtype=torch.uint8)
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(num_buffers):
        for key in buffers:
            buffers[key].append(torch.zeros(**specs[key]).share_memory_())
    return buffers


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_actor(checkpointpath):
    import json
    with open(checkpointpath.replace("model.tar", "meta.json"), 'r') as file:
        args = json.load(file)["args"]
        if "breath" not in args:
           args["breath"] = 2
    env = create_gymenv(AttributeDict(args))
    return env, create_model(AttributeDict(args), env), AttributeDict(args)


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    terms = flags.xpid.split("-")
    if len(terms) == 3:
        group = terms[0] + "-" + terms[1]
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if (not flags.disable_cuda) and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda:"+str(torch.cuda.current_device()))
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    step, stats = 0, {}

    env = create_gymenv(flags)
    actor_flags = flags
    try:
        checkpoint = torch.load(checkpointpath, map_location=flags.device)
        step = checkpoint["step"]
    except Exception as e:
        print(e)

    model = create_model(flags, env).to(device=flags.device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        print(e)

    if flags.agent in ["CNN"]:
        buffers = create_buffers(flags, env.observation_space.spaces["image"].shape, env.action_space.n,
                                 flags.unroll_length, flags.num_buffers,
                                 img_shape=env.observation_space.spaces["image"].shape)
        actor_buffers = create_buffers(flags, env.observation_space.spaces["image"].shape, env.action_space.n,
                                       0, flags.num_actors,
                                       img_shape=env.observation_space.spaces["image"].shape)
    elif flags.agent in ["NLM", "KBMLP", "GCN"]:
        buffers = create_buffers(flags, env.obs_shape, model.num_actions,
                                 flags.unroll_length, flags.num_buffers,
                                 img_shape=env.observation_space.spaces["image"].shape)
        actor_buffers = create_buffers(flags, env.obs_shape, model.num_actions,
                                       0, flags.num_actors,
                                       img_shape=env.observation_space.spaces["image"].shape)
    else:
        raise ValueError()

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    actor_model_queues = [ctx.SimpleQueue() for _ in range(flags.num_actors)]
    actor_env_queues = [ctx.SimpleQueue() for _ in range(flags.num_actors)]

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                actor_flags,
                create_gymenv(flags),
                i,
                free_queue,
                full_queue,
                buffers,
                actor_buffers,
                actor_model_queues,
                actor_env_queues
            ),
        )
        actor.start()
        actor_processes.append(actor)


    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    if flags.mode == "imitate":
        stat_keys = [
            "total_loss",
            "accuracy",
            "mean_episode_return",
            ]
    else:
        stat_keys = [
            "total_loss",
            "mean_episode_return",
            "pg_loss",
            "baseline_loss",
            "entropy_loss",
        ]
    logger.info("# Step\t%s", "\t".join(stat_keys))
    finish=False

    def batch_and_inference():
        nonlocal finish
        while not all(finish):
            indexes = []
            for i in range(flags.num_actors):
                indexes.append(actor_model_queues[i].get())
            batch = get_inference_batch(flags, actor_buffers)
            with torch.no_grad():
                agent_output = model(batch)

            for index in indexes:
                for key in agent_output:
                    actor_buffers[key][index][0] = agent_output[key][0, index]
            for i in range(flags.num_actors):
                actor_env_queues[i].put(None)

    finish = [False for _ in range(flags.num_learner_threads)]

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats, finish
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )
            stats = learn(flags, model, batch, optimizer, scheduler)
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())
        finish[i]=True

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    thread = threading.Thread(target=batch_and_inference, name="batch-and-inference")
    thread.start()
    threads.append(thread)
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
                "step": step
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            total_loss = stats.get("total_loss", float("inf"))
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for i in range(flags.num_actors):
            free_queue.put(None)
            actor_env_queues[i].put("exit")

        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


def create_model(flags, env):
    if flags.activation == "relu":
        activation = nn.ReLU()
    elif flags.activation == "sigmoid":
        activation = nn.Sigmoid()
    else:
        raise ValueError()

    if flags.agent == "CNN":
        model = CNNAgent(env.observation_space.spaces["image"].shape, env.action_space.n,
                         flags.cnn_code, flags.use_lstm, embedding_size=flags.embedding_size)
    elif flags.agent == "NLM":
        obs_shape = env.observation_space.spaces["image"].shape
        action = flags.action
        if action == "relational" and flags.state == "integer":
            action = "move_xy"
        model = NLMAgent(env.obj_n, flags.depth, flags.breath,
                         [env.obs_shape[0],env.obs_shape[1][-1], env.obs_shape[2][-1]], env.action_space.n,
                         observation_shape=obs_shape, nn_units=flags.global_units,
                         output_dims=flags.output_dims, residual=False,
                         activation=activation, action_type=action)
    elif flags.agent == "KBMLP":
        model = KnowledgeBaseMLP(env.obj_n, [env.obs_shape[0],env.obs_shape[1][-1], env.obs_shape[2][-1]],
                                 env.action_space.n)
    elif flags.agent in ["GCN"]:
        model = GNNAgent(env.nb_all_entities, env.action_space.n, [env.obs_shape[0],env.obs_shape[1][-1], env.obs_shape[2][-1]],
                         type=flags.agent, net_code=flags.cnn_code, embedding_size=flags.embedding_size, mp_rounds=flags.mp_rounds)
    return model


def test(flags):
    num_episodes = flags.episodes
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        log_path = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, flags.env))
        )
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    gym_env = create_gymenv(flags)
    if flags.agent in ["CNN", "SNLM", "MHA"]:
        env = environment.Environment(gym_env, "image")
    elif flags.agent in ["NLM", "KBMLP", "GCN"]:
        env = environment.Environment(gym_env, "absVKB")
    model = create_model(flags, gym_env)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    observation = env.initial()
    returns = []

    total_steps = 0
    obs_index = 0
    if flags.store_stats:
        stats = dict(episode=[], total_steps=[], reward=[], action=[], obs_index=[])
        evals = [[] for _ in range(3)]
        obs = []
    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()
        agent_outputs = model(observation)
        policy_outputs = agent_outputs
        observation = env.step(policy_outputs["action"])

        if flags.store_stats:
            frame = observation["frame"].numpy()[0,0]
            if not obs or np.any(obs[-1] != frame):
                if "evaluation" in policy_outputs:
                    evaluation = policy_outputs["evaluation"]
                    for i, eval in enumerate(evals):
                        eval.append(evaluation[i].detach().numpy()[0])
                obs.append(frame)
            else:
                obs_index -= 1
            stats["episode"].append(len(returns))
            stats["total_steps"].append(total_steps)
            stats["obs_index"].append(obs_index)
            stats["reward"].append(observation["reward"].numpy()[0,0])
            stats["action"].append(policy_outputs["action"].numpy()[0,0])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.2f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
        #index_img = vector2index_img(observation["frame"])
        #render_index_img(gym_env.get_index_img())
        #print(str(env.gym_env))
        #print("-"*15)
        #time.sleep(0.1)
        total_steps+=1
        obs_index+=1
    env.close()
    if flags.store_stats:
        if "evaluation" in policy_outputs:
            for i, eval in enumerate(evals):
                np.save(log_path+f"/eval-{i}-arity.npy", np.stack(eval))
        np.save(log_path+"/obs.npy", np.stack(obs))
        pd.DataFrame(stats).to_csv(log_path+"/stats.csv")
    mean = sum(returns) / len(returns)
    std = np.std(returns)
    logging.info(
        "Average returns over %i steps: %.2f ± %.2f", num_episodes, mean, std
    )
    env_name = flags.env.replace("MiniGrid-","").replace("-v0","")

    if flags.env in ["rtfm", "rtfm-onehop"]:
        wins = np.array(returns)>-1.0
        win_rate = np.mean(wins)*100
        win_std = np.std(wins)*100
        print(f"{mean:.2f} ± {std:.2f}, {win_rate:.2f} ± {win_std:.2f}")
    else:
        print(f"{mean:.2f} ± {std:.2f}")
    return mean, std


def create_gymenv(flags):
    if flags.env in ["seaquest", "breakout", "asterix", "freeway", "space_invaders"]:
        env_type = "minatar"
    elif flags.env == "random":
        env_type = "random"
    elif "block-" in flags.env:
        env_type = "blockworld"
    elif flags.env in ["rtfm", "rtfm-onehop"]:
        env_type = "rtfm"
    elif flags.env == "boxworld":
        env_type = "boxworld"
    else:
        env_type = "minigrid"

    portal_pairs = []
    if env_type == "minigrid":
        env = gym.make(flags.env)
        #env = ReseedWrapper(env)
        env = FullyObsWrapper(env)
        env = PaddingWrapper(env)
        if flags.action == "moveto":
            env = MoveToActionWrapper(env)
        elif flags.action == "move_dir":
            env = MoveDirActionWrapper(env)
        if flags.env == "MiniGrid-LavaCrossingClosed-v0":
            env = ProtalWrapper(env, portal_pairs)
    elif env_type == "minatar":
        from environment.minatarwarpper import MinAtarEnv
        env = MinAtarEnv(flags.env, flags.sticky_prob)
    elif env_type == "random":
        from environment.random import RandomEnv
        env = RandomEnv()
    elif env_type == "blockworld":
        from environment.blockworld import BlockEnv, GridActionWrapper, BlockActionWrapper
        state_block_spec = False if flags.state != "block" and flags.action == "propositional" else True
        env = BlockEnv(flags.env, nb_blocks=flags.nb_blocks, variation=flags.variation,
                       rand_env=flags.rand_env, state_block_spec=state_block_spec)
        if flags.state != "block" and flags.action == "relational":
            env = GridActionWrapper(env)
        if flags.state == "block" and flags.action == "relational":
            env = BlockActionWrapper(env)
    elif env_type in ["rtfm"]:
        from environment.rtfmkbenv import RTFMEnv, RTFMAbstractEnv, RTFMOneHopEnv
        with_vkb = False if flags.agent in ["CNN", "MHA"] or flags.disable_wiki else True
        if with_vkb:
            if flags.env == "rtfm":
                env = RTFMAbstractEnv(flags.room_size)
            elif flags.env == "rtfm-onehop":
                env = RTFMOneHopEnv(flags.room_size)
            else:
                raise ValueError()
        else:
            env = RTFMEnv()

    if flags.agent in ["NLM", "KBMLP", "GCN"]:
        if env_type == "minigrid":
            env = DirectionWrapper(env)
        if flags.state == "absolute":
            env = AbsoluteVKBWrapper(env, flags.bg_code, portal_pairs)
        elif flags.state == "block":
            from environment.blockworld import BlockVKBWarpper
            env = BlockVKBWarpper(env)
        else:
            raise ValueError(f"state encoding cannot be {flags.state}")
    elif flags.agent in ["SNLM"]:
        if env_type == "minigrid":
            env = DirectionWrapper(env, type="onehot")
            env = OneHotFullyObsWrapper(env)

    return env


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)

if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
