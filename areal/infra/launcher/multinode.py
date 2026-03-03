import getpass
import json
import os
import signal as signal_module
import socket
import subprocess
import sys
import time
import psutil

from areal.api.alloc_mode import AllocationMode, AllocationType
from areal.api.cli_args import (
    ClusterSpecConfig,
    RecoverConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.infra.platforms import current_platform
from areal.utils import logging as areal_logging
from areal.utils import name_resolve, names
from areal.infra.utils.exp_metadata import save_experiment_metadata
from areal.infra.utils.launcher import (
    BASE_ENVIRONS,
    JobException,
    JobState,
    get_scheduling_spec,
    get_thread_env_vars,
    validate_config_for_distributed_launcher,
    wait_llm_server_addrs,
)
from areal.utils.offload import get_tms_env_vars
from areal.utils.recover import check_if_recover

logger = areal_logging.getLogger("MultiNodeLauncher")


def terminate_process_and_children(pid: int, signal: str | int | None = None):
    if signal is None:
        signal = signal_module.SIGKILL
    if isinstance(signal, str):
        signal = getattr(signal_module, signal)
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            terminate_process_and_children(child.pid, signal)
        parent.send_signal(signal)
    except psutil.NoSuchProcess:
        pass


class MultiNodeLauncher:
    def __init__(self, experiment_name: str, trial_name: str, fileroot: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot
        self._jobs: dict[str, subprocess.Popen] = {}
        self._job_states = {}

    @property
    def run_name(self):
        return f"{self.experiment_name}_{self.trial_name}"

    def log_path_of(self, job_name: str) -> str:
        log_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, f"{job_name}.log")

    def submit(
        self,
        job_name: str,
        cmd: str,
        env_vars: dict | None = None,
    ):
        if env_vars is None:
            env_vars = {}
        
        c = (
            " ".join(str(k) + "=" + str(v) for k, v in env_vars.items())
            + " stdbuf -oL "
            + cmd
        )
        c = f"{c} 2>&1 | tee -a {self.log_path_of(job_name)}"
        logger.info("Starting process with command: %s", c)
        process = subprocess.Popen(
            c, shell=True, stdout=sys.stdout, stderr=sys.stdout
        )
        self._jobs[job_name] = process

    def stop_all(self, signal=None):
        for job_name, p in self._jobs.items():
            logger.info(f"Stopping {job_name}")
            terminate_process_and_children(p.pid, signal=signal)
            p.wait()
        self._jobs.clear()

    def wait(self):
        while self._jobs:
            for job_name, p in list(self._jobs.items()):
                ret = p.poll()
                if ret is not None:
                    if ret != 0:
                        raise JobException(
                            run_name=self.run_name, 
                            worker_type=job_name, 
                            host=socket.gethostname(),
                            reason=JobState.FAILED
                        )
                    else:
                        logger.info(f"Job {job_name} finished successfully.")
                        self._jobs.pop(job_name)
            time.sleep(1)


def get_cluster_env():
    if "AFO_ENV_CLUSTER_SPEC" in os.environ and "AFO_SPEC" in os.environ:
        try:
            cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
            afo_spec = json.loads(os.environ["AFO_SPEC"])
            
            node_rank = int(cluster_spec["index"])
            role = cluster_spec["role"]
            
            return node_rank, afo_spec['cluster'][role]
        except Exception as e:
            logger.warning(f"Error parsing AFO env: {e}")

    return 0, []


def multinode_main(config, run_id: int = 0):
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    is_recover_run = check_if_recover(config.recover, run_id)
    validate_config_for_distributed_launcher(config)
    
    my_rank, cluster_hosts = get_cluster_env()
    if not cluster_hosts:
        my_rank = int(os.environ.get("RANK", "0"))
        logger.warning("Could not determine cluster hosts. Assuming running on localhost or single node.")
        cluster_hosts = ["localhost:1234"]

    launcher = MultiNodeLauncher(
        config.experiment_name, config.trial_name, config.cluster.fileroot
    )

    name_resolve.reconfigure(config.cluster.name_resolve)
    if my_rank == 0:
        name_resolve.clear_subtree(
            names.trial_root(
                experiment_name=config.experiment_name, trial_name=config.trial_name
            )
        )
        if not is_recover_run:
            metadata_file = save_experiment_metadata(
                config.cluster.fileroot,
                config.experiment_name,
                config.trial_name,
            )
            logger.info(f"Saved experiment metadata to {metadata_file}")

    logger.info(
        f"MultiNodeLauncher: rank={my_rank}, experiment_name={config.experiment_name}, "
        f"trial_name={config.trial_name}"
    )

    full_world_size = config.cluster.n_nodes
    n_gpus_per_node = config.cluster.n_gpus_per_node
    
    allocation_mode = AllocationMode.from_str(config.allocation_mode)

    n_gen_nodes = 0
    n_train_nodes = 0
    
    if allocation_mode.gen_backend == "sglang":
        n_gen_nodes = max(1, allocation_mode.gen.world_size // n_gpus_per_node)
    elif allocation_mode.gen_backend == "vllm":
        n_gen_nodes = max(1, allocation_mode.gen.world_size // n_gpus_per_node)
        
    if allocation_mode.type_ == AllocationType.DECOUPLED_EVAL:
        n_train_nodes = 1
    elif allocation_mode.type_ == AllocationType.LLM_SERVER_ONLY:
        n_train_nodes = 0
        n_gen_nodes = full_world_size
    else:
        n_train_nodes = full_world_size - n_gen_nodes

    is_gen_node = my_rank < n_gen_nodes
    is_train_node = my_rank >= n_gen_nodes
    
    if is_gen_node and allocation_mode.gen_backend in ("sglang", "vllm"):
        if allocation_mode.gen_backend == "sglang":
            config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
            base_seed = config.sglang.random_seed
            module_name = "areal.infra.launcher.sglang_server"
            seed_arg_name = "sglang.random_seed"
        else:
            config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
            base_seed = config.vllm.seed
            module_name = "areal.infra.launcher.vllm_server"
            seed_arg_name = "vllm.seed"

        n_servers_total = allocation_mode.gen.dp_size
        n_servers_per_node = max(n_servers_total // n_gen_nodes, 1)
        
        rollout_spec = get_scheduling_spec(config.rollout)
        rollout_env_vars = rollout_spec.env_vars
        rollout_cpus_per_task = rollout_spec.cpu
        thread_env = get_thread_env_vars(
            cpus_per_task=rollout_cpus_per_task,
            existing_env_vars=rollout_env_vars,
        )

        for i in range(n_servers_per_node):
            server_index = my_rank * n_servers_per_node + i
            seed = base_seed + server_index
            
            cmd = f"python3 -m {module_name} {' '.join(sys.argv[1:])} {seed_arg_name}={seed}"
            
            current_env = {**BASE_ENVIRONS, **thread_env, **rollout_env_vars}
            
            if n_servers_per_node > 1:
                gpus_per_server = n_gpus_per_node // n_servers_per_node
                start_gpu = i * gpus_per_server
                end_gpu = start_gpu + gpus_per_server
                visible_devices = ",".join(str(g) for g in range(start_gpu, end_gpu))
                current_env[current_platform.device_control_env_var] = visible_devices
            
            launcher.submit(
                job_name=f"llm_server_{i}",
                cmd=cmd,
                env_vars=current_env
            )

    server_addrs = []
    if allocation_mode.type_ != AllocationType.LLM_SERVER_ONLY:
        try:
            n_servers_to_wait = allocation_mode.gen.dp_size
            server_addrs = wait_llm_server_addrs(
                 config.experiment_name,
                 config.trial_name,
                 n_rollout_servers=n_servers_to_wait
            )
            logger.info(f"All LLM Servers Ready: {len(server_addrs)}")
        except (TimeoutError, KeyboardInterrupt) as e:
            launcher.stop_all(signal="SIGINT")
            raise e

    if is_train_node and allocation_mode.type_ != AllocationType.LLM_SERVER_ONLY:
        trainer_rank = my_rank - n_gen_nodes
        trainer_master_node_index = n_gen_nodes
        
        if trainer_master_node_index < len(cluster_hosts):
            master_entry = cluster_hosts[trainer_master_node_index]
            master_addr = master_entry.split(":")[0]
            master_port = 29505 
        else:
            logger.error("Cluster host list too short for trainer master index")
            master_addr = "localhost"
            master_port = 29505

        if config.get("enable_offload", False):
            tms_env_vars = get_tms_env_vars()
        else:
            tms_env_vars = {}
            
        actor_spec = get_scheduling_spec(config.actor)
        actor_env_vars = actor_spec.env_vars
        actor_cpus_per_task = actor_spec.cpu
        thread_env = get_thread_env_vars(
            cpus_per_task=actor_cpus_per_task,
            existing_env_vars=actor_env_vars,
        )

        _env_vars = dict(
            AREAL_LLM_SERVER_ADDRS=",".join(server_addrs),
        )
        
        if allocation_mode.gen_backend == "sglang":
             _env_vars["NCCL_CUMEM_ENABLE"] = "0"
             _env_vars["NCCL_NVLS_ENABLE"] = "0"

        # training script is the first argument after launcher module
        # But sys.argv contains `python -m areal.launcher.multinode train.py ...`
        # So we pass sys.argv[1:] to torchrun
        
        cmd = (
            f"torchrun "
            f"--nnodes {n_train_nodes} "
            f"--nproc-per-node {n_gpus_per_node} "
            f"--node_rank {trainer_rank} "
            f"--master_addr {master_addr} "
            f"--master_port {master_port} "
            f"{' '.join(sys.argv[1:])}"
        )
        
        launcher.submit(
            job_name="trainer",
            cmd=cmd,
            env_vars={
                **BASE_ENVIRONS,
                **thread_env,
                **actor_env_vars,
                **_env_vars,
                **tms_env_vars,
                "AREAL_SPMD_MODE": "1",
            }
        )

    try:
        launcher.wait()
    except (KeyboardInterrupt, JobException) as e:
        launcher.stop_all(signal="SIGTERM")
        raise e


def main():
    config, _ = parse_cli_args(sys.argv[1:])
    multinode_main(config)


if __name__ == "__main__":
    main()
