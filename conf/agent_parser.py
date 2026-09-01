import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        raise argparse.ArgumentTypeError("Boolean value expected.")

    value = v.lower()
    if value == "true":
        return True
    if value == "false":
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def agent_parse_args():
    """Parser for the CRAFTS recommendation pipeline.

    Model roles:
    1) Qwen3-8B is used as the teacher model for Skill Bank initialization,
       Skill Bank evolution, and teacher-trajectory generation.
    2) Qwen3-4B is used as the recommendation-agent/student backbone.
    3) The student is trained with trajectory-supervised LoRA SFT and is
       used for final recommendation inference.
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Tool / recommender wrapper
    parser.add_argument(
        "--tool_name",
        type=str,
        help="name of the tool model module",
    )
    parser.add_argument(
        "--tool_type",
        type=str,
        default="sequential",
        help="package/type of the tool model",
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        default=None,
        help="path to external config file for the tool model",
    )
    parser.add_argument(
        "--load_recommender",
        type=str2bool,
        default=True,
        help="load saved recommender artifacts",
    )
    parser.add_argument(
        "--MAX_ITEM_LIST_LENGTH",
        type=int,
        default=50,
        help="maximum interaction sequence length",
    )
    parser.add_argument(
        "--MIN_ITEM_LIST_LENGTH",
        type=int,
        default=5,
        help="minimum interaction sequence length",
    )

    # Recommendation agent / student
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-4B",
        help="recommendation-agent/student model name",
    )
    parser.add_argument(
        "--agent",
        type=str2bool,
        default=True,
        help="whether to use agent recommender mode",
    )
    parser.add_argument(
        "--use_candidate_prior",
        type=str2bool,
        default=True,
        help="whether to generate prior recommendations from the candidate set",
    )
    parser.add_argument(
        "--candidate_generate_mode",
        type=str,
        default="random",
        help="candidate generation mode: random, tool, or tool_shuffle",
    )
    parser.add_argument(
        "--agent_workers",
        type=int,
        default=4,
        help="number of agent worker threads for the test stage",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="student/recommendation-agent backbone path or model identifier",
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default="Qwen/Qwen3-8B",
        help=(
            "teacher model path or identifier used for Skill Bank initialization, "
            "Skill Bank evolution, and teacher-trajectory generation"
        ),
    )
    parser.add_argument(
        "--student_lora_path",
        type=str,
        default="./artifacts/skill_student/qwen3_4b_Instruments_lora",
        help="LoRA adapter path for the trained recommendation agent",
    )
    parser.add_argument(
        "--train_user_num",
        type=str,
        default="all",
        help="users used for train-stage skill initialization and update",
    )
    parser.add_argument(
        "--test_user_num",
        type=str,
        default="all",
        help="users used in the test stage; use 'all' for the full set",
    )
    parser.add_argument(
        "--report_k",
        type=int,
        default=1,
        help="online HR reporting cutoff during agent evaluation",
    )
    parser.add_argument(
        "--report_interval",
        type=int,
        default=50,
        help="online HR reporting interval",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=4,
        help="maximum recommendation-user interaction rounds",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM generation temperature",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="load frozen artifacts and run test only",
    )
    parser.add_argument(
        "--test_user_file",
        type=str,
        default="",
        help="optional ordered raw-user-id manifest",
    )
    parser.add_argument(
        "--retriever_checkpoint",
        type=str,
        default="",
        help="explicit frozen skill-retriever checkpoint",
    )
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="",
        help="optional experiment-specific directory for evaluation metrics",
    )
    parser.add_argument(
        "--prompt_log_path",
        type=str,
        default="",
        help="optional prompt debug log; empty disables prompt logging",
    )

    # Skill framework
    parser.add_argument(
        "--enable_memskill",
        action="store_true",
        default=False,
        help="enable the CRAFTS skill framework",
    )
    parser.add_argument(
        "--skill_bank_path",
        type=str,
        default="./artifacts/skills",
        help="path to the Skill Bank directory",
    )
    parser.add_argument(
        "--user_skill_top_k",
        type=int,
        default=2,
        help="number of retrieved user-profile skills per recommendation",
    )
    parser.add_argument(
        "--functional_skill_top_k",
        type=int,
        default=2,
        help="maximum number of applicable functional skills per recommendation",
    )
    parser.add_argument(
        "--skill_update_interval",
        type=int,
        default=10,
        help="trigger Skill Bank evolution every N accumulated failed train cases",
    )
    parser.add_argument(
        "--failure_buffer_size",
        type=int,
        default=200,
        help="maximum number of failed cases stored in the buffer",
    )
    parser.add_argument(
        "--max_skill_cases_per_round",
        type=int,
        default=20,
        help="maximum number of failed cases analyzed per evolution round",
    )
    parser.add_argument(
        "--initial_user_skill_budget",
        type=int,
        default=200,
        help="maximum number of train users used to initialize user-profile skills",
    )
    parser.add_argument(
        "--user_skill_cluster_size",
        type=int,
        default=6,
        help="target cluster/prototype size when aggregating user-profile skills",
    )
    parser.add_argument(
        "--skill_similarity_threshold",
        type=float,
        default=0.35,
        help="similarity threshold used during user-profile skill aggregation/retrieval",
    )
    parser.add_argument(
        "--skill_use_llm_summary",
        type=str2bool,
        default=False,
        help="use the teacher LLM to summarize histories when constructing skills",
    )
    parser.add_argument(
        "--skill_init_use_llm",
        type=str2bool,
        default=True,
        help="use the teacher LLM to initialize user-profile skills",
    )
    parser.add_argument(
        "--skill_evolve_use_llm",
        type=str2bool,
        default=True,
        help="use the teacher LLM to evolve skills from failed interactions",
    )
    parser.add_argument(
        "--skill_update_rounds",
        type=int,
        default=1,
        help="number of train-stage Skill Bank update rounds",
    )

    # Trajectory distillation
    parser.add_argument(
        "--distill_user_num",
        type=str,
        default="all",
        help="train users used for teacher-trajectory generation",
    )
    parser.add_argument(
        "--distill_keep_success_only",
        type=str2bool,
        default=True,
        help="retain only teacher trajectories whose final prediction matches ground truth",
    )

    # Trainable user-profile skill retriever
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="frozen text encoder used for skill representations",
    )
    parser.add_argument(
        "--retriever_negatives_per_positive",
        type=int,
        default=2,
        help="number of negative user-profile skills sampled per positive skill",
    )
    parser.add_argument(
        "--retriever_negative_strategy",
        type=str,
        default="bottom",
        help="negative-sampling strategy for retriever training",
    )
    parser.add_argument(
        "--retriever_bottom_negative_ratio",
        type=float,
        default=0.3,
        help="bottom-similarity pool ratio used for negative sampling",
    )
    parser.add_argument(
        "--retriever_middle_negative_ratio",
        type=float,
        default=0.3,
        help="middle-similarity pool ratio used for negative sampling",
    )
    parser.add_argument(
        "--retriever_positive_similarity_threshold",
        type=float,
        default=0.35,
        help="minimum semantic similarity for positive-skill construction",
    )
    parser.add_argument(
        "--retriever_use_success_only",
        type=str2bool,
        default=True,
        help="construct retriever supervision only from successful train trajectories",
    )
    parser.add_argument(
        "--retriever_min_train_records",
        type=int,
        default=8,
        help="minimum pairwise records required to train the skill retriever",
    )
    parser.add_argument(
        "--retriever_hidden_dim",
        type=int,
        default=256,
        help="hidden dimension of the trainable retriever projector",
    )
    parser.add_argument(
        "--retriever_output_dim",
        type=int,
        default=128,
        help="output dimension of the trainable retriever projector",
    )
    parser.add_argument(
        "--retriever_dropout",
        type=float,
        default=0.1,
        help="dropout used in the trainable retriever projector",
    )
    parser.add_argument(
        "--retriever_batch_size",
        type=int,
        default=32,
        help="retriever training batch size",
    )
    parser.add_argument(
        "--retriever_epochs",
        type=int,
        default=20,
        help="number of retriever training epochs",
    )
    parser.add_argument(
        "--retriever_learning_rate",
        type=float,
        default=1e-3,
        help="retriever learning rate",
    )
    parser.add_argument(
        "--retriever_weight_decay",
        type=float,
        default=1e-5,
        help="retriever weight decay",
    )
    parser.add_argument(
        "--retriever_margin",
        type=float,
        default=None,
        help="optional margin used by the retriever ranking loss",
    )
    parser.add_argument(
        "--retriever_temperature",
        type=float,
        default=1.0,
        help="temperature used by the retriever objective",
    )
    parser.add_argument(
        "--retriever_grad_clip_norm",
        type=float,
        default=1.0,
        help="gradient clipping norm for retriever training",
    )
    parser.add_argument(
        "--retriever_num_workers",
        type=int,
        default=0,
        help="number of workers used by the retriever data loader",
    )

    return parser.parse_known_args()[0]
