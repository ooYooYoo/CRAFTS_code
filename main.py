import importlib
import os
import time
import warnings

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from conf.agent_parser import agent_parse_args
from conf.recommend_parser import recommend_parse_args


def _configure_torch():
    """Import and configure torch after CUDA_VISIBLE_DEVICES has been set."""
    import torch

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="torch._weights_only_unpickler",
    )

    original_load = torch.load

    def safe_load(*args, **kwargs):
        # Compatibility with existing RecBole/PyTorch checkpoints.
        # Only load checkpoints from trusted sources.
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        if not torch.cuda.is_available() and "map_location" not in kwargs:
            kwargs["map_location"] = torch.device("cpu")
        return original_load(*args, **kwargs)

    torch.load = safe_load
    return torch


def main():
    # ------------------------------------------------------------------
    # 1. Parse arguments before importing CUDA-dependent modules.
    # ------------------------------------------------------------------
    recommend_args = recommend_parse_args()
    agent_args = agent_parse_args()

    # Merge agent arguments into the shared argument namespace used by ARS.
    for key, value in vars(agent_args).items():
        setattr(recommend_args, key, value)

    # ------------------------------------------------------------------
    # 2. Configure device and random seed.
    # ------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(recommend_args.gpu_id)
    _configure_torch()

    from util.tool import seedSet
    seedSet(recommend_args.seed)

    # Import the remaining project modules only after the CUDA device
    # has been fixed.
    from ARLib import ARLib
    from util.DataLoader import DataLoader

    # ------------------------------------------------------------------
    # 3. Load the requested recommender class.
    # ------------------------------------------------------------------
    module_name = (
        f"recommender.{recommend_args.model_type}.{recommend_args.model_name}"
    )
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, recommend_args.model_name)

    # ------------------------------------------------------------------
    # 4. Prepare data.
    # ------------------------------------------------------------------
    if recommend_args.agent and not recommend_args.test_only:
        print(
            f"[Main] Preparing dataset '{recommend_args.dataset}' "
            "for the CRAFTS build stage..."
        )

        from util.DataLoader import DataLoader_recbole_Raw
        from util.split_data import BenchmarkSplitter

        raw_loader = DataLoader_recbole_Raw(recommend_args)
        splitter = BenchmarkSplitter(
            raw_loader=raw_loader,
            agent_args=agent_args,
            rec_args=recommend_args,
            dataset=recommend_args.dataset,
            agentname=recommend_args.model_name,
            save_dir=recommend_args.save_dir,
        )
        splitter.split_and_save()

        from util.DataLoader import DataLoader_recbole
        data = DataLoader_recbole(recommend_args)

    elif recommend_args.agent:
        print(
            f"[Main] Test-only mode: reusing frozen benchmark files "
            f"for '{recommend_args.dataset}'."
        )

        from util.DataLoader import DataLoader_recbole
        data = DataLoader_recbole(recommend_args)

    else:
        data = DataLoader(recommend_args)

    # ------------------------------------------------------------------
    # 5. Build model and coordinator.
    # ------------------------------------------------------------------
    recommend_model = ModelClass(recommend_args, data)

    arlib = ARLib(
        recommendModel=recommend_model,
        recommendArg=recommend_args,
        agentArg=agent_args,
    )

    # ------------------------------------------------------------------
    # 6. Run one pipeline stage.
    #
    # build:
    #   training/tool preparation
    #   Skill Bank initialization
    #   failure-driven Skill Bank evolution
    #   skill-retriever training
    #   teacher-trajectory generation
    #
    # infer (--test_only):
    #   load frozen artifacts and evaluate the trained student
    # ------------------------------------------------------------------
    start = time.time()

    if recommend_args.test_only:
        arlib.PrepareForTest()
        arlib.RecommendTest()
    else:
        arlib.RecommendTrain()

    elapsed = time.time() - start
    print(f"Running time: {elapsed:.6f} s")


if __name__ == "__main__":
    main()
