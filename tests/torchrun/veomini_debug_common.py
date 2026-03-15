"""Common configuration for VeOMini debug/parity tests.

Provides a shared TrainEngineConfig factory that all VeOMini parity test scripts
import.  Uses Qwen3-0.6B as the reference model.
"""

from areal.api.cli_args import OptimizerConfig, TrainEngineConfig
# from tests.utils import get_model_path

MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/FMG/pengxinyu05/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"


def make_debug_train_config(
    experiment_name: str,
    lr: float = 1e-5,
    gradient_clipping: float = 1.0,
    init_from_scratch: bool = False,
) -> TrainEngineConfig:
    """Create a TrainEngineConfig for debug/parity testing with Qwen3-0.6B.

    Parameters
    ----------
    experiment_name : str
        Unique name for the experiment run.
    lr : float
        Learning rate for the Adam optimizer.
    gradient_clipping : float
        Max gradient norm for gradient clipping.
    init_from_scratch : bool
        If True, initialize model with random weights instead of loading checkpoint.
    """
    return TrainEngineConfig(
        experiment_name=experiment_name,
        trial_name="debug",
        path=MODEL_PATH,
        optimizer=OptimizerConfig(
            type="adam",
            lr=lr,
            weight_decay=0.0,
            gradient_clipping=gradient_clipping,
        ),
        init_from_scratch=init_from_scratch,
        dtype="bfloat16",
        attn_impl="flash_attention_2",
        gradient_checkpointing=False,
        disable_dropout=True,
    )
