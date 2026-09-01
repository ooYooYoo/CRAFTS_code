import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


STUDENT_BOT = None
TEACHER_BOT = None
STUDENT_BOT_LOCK = threading.Lock()
TEACHER_BOT_LOCK = threading.Lock()
BOT_INIT_LOCK = threading.Lock()


current_dir = os.path.dirname(os.path.abspath(__file__))
ars_sub_package_path = os.path.join(current_dir, "ARS")
if os.path.exists(ars_sub_package_path) and ars_sub_package_path not in sys.path:
    sys.path.insert(0, ars_sub_package_path)

try:
    from utils.agent import RecAgent, UserModelAgent
    from utils.local_request import QwenLocal
    from utils.regular_function import split_rec_reponse, split_user_response
    from dataset.crafts_dataset import CRAFTSDataset

    print("[ARS] Dependencies loaded successfully from ARS sub-package.")
except ImportError as exc:
    print(f"[ARS] CRITICAL ERROR: Could not import dependencies: {exc}")
    RecAgent = None
    UserModelAgent = None
    QwenLocal = None
    CRAFTSDataset = None

from .skill import FailureBuffer, FailureCase, SkillBank, SkillDesigner

try:
    from .skill.skill_retriever import (
        TrainableSkillRetriever,
        build_pairwise_records_from_logs,
        save_records_jsonl,
        train_skill_retriever,
    )

    print("[ARS] skill_retriever loaded successfully.")
except Exception as exc:
    print("[ARS] skill_retriever import failed:", repr(exc))
    TrainableSkillRetriever = None
    build_pairwise_records_from_logs = None
    save_records_jsonl = None
    train_skill_retriever = None


class ARS:
    """CRAFTS recommendation agent.

    Model roles in the released Qwen3-4B experiment:
      - Teacher: Qwen3-8B for Skill Bank initialization, Skill Bank evolution,
        and teacher-trajectory generation.
      - Student / Recommendation Agent: Qwen3-4B for skill-guided interaction,
        trajectory-supervised LoRA SFT, and final inference.
      - Retriever encoder: frozen all-MiniLM-L6-v2 with a trainable projector.
    """

    def __init__(self, args, data):
        print("Recommender: CRAFTS (SASRec + Skill-Guided Agent Reranking)")
        self.args = args
        self.data = data
        self.use_candidate_prior = args.use_candidate_prior
        self.tool_model = None
        self.model = None

        self.item_num = int(self.data.item_num)
        self.state_size = 50
        self.final_item_embs = None
        self.agent_results = {}

        if hasattr(self.args, "topK"):
            self.max_N = max(int(n) for n in str(self.args.topK).split(","))
        else:
            self.max_N = 10

        self.worker_init(self.args)

        self.enable_memskill = getattr(args, "enable_memskill", False)
        dataset_name = str(
            getattr(self.data, "dataName", getattr(args, "dataset", ""))
        )
        self.skill_bank = SkillBank(
            getattr(args, "skill_bank_path", "./artifacts/skills"),
            dataset_scope=dataset_name,
        )
        self.failure_buffer = FailureBuffer(
            max_size=getattr(args, "failure_buffer_size", 200)
        )
        self.designer = SkillDesigner(
            self.skill_bank,
            embedding_model_name=getattr(
                self.args,
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            local_bot=TEACHER_BOT,
        )

        self.id2name_cache = {}
        self.retriever_dir = os.path.join(self.skill_bank.skill_dir, "retriever")
        self.retriever_projector_path = os.path.join(
            self.retriever_dir,
            "projector.pt",
        )
        self.active_retriever_path = (
            str(getattr(self.args, "retriever_checkpoint", "") or "").strip()
            or self.retriever_projector_path
        )
        self.retriever_records_path = os.path.join(
            self.retriever_dir,
            "train_pairwise_records.jsonl",
        )
        self.retriever_history_path = os.path.join(
            self.retriever_dir,
            "train_history.json",
        )
        self.retrieval_logs_path = os.path.join(
            self.retriever_dir,
            "retrieval_interaction_logs.jsonl",
        )
        self.retrieval_interaction_logs = []
        self.skill_evolution_call_count = 0

        self.distill_dir = os.path.join(self.skill_bank.skill_dir, "distill")
        self.distill_data_path = os.path.join(
            self.distill_dir,
            "query_user_skill_distill_train.jsonl",
        )

    # ------------------------------------------------------------------
    # model initialization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_adapter_path(path):
        if path is None:
            return None
        value = str(path).strip()
        if value.lower() in {"", "none", "null"}:
            return None
        return value

    @staticmethod
    def worker_init(args):
        """Load Student and Teacher as separate model roles."""
        global STUDENT_BOT, TEACHER_BOT

        if QwenLocal is None:
            raise RuntimeError("QwenLocal is unavailable; ARS dependencies failed to load.")

        with BOT_INIT_LOCK:
            student_model_path = getattr(args, "local_model_path", None)
            teacher_model_path = getattr(args, "teacher_model_path", None)
            student_adapter_path = ARS._normalize_adapter_path(
                getattr(args, "student_lora_path", None)
            )

            if STUDENT_BOT is None and student_model_path:
                print(
                    f"[worker {os.getpid()}] loading Student / Rec Agent from "
                    f"{student_model_path} ..."
                )
                STUDENT_BOT = QwenLocal(
                    model_path=student_model_path,
                    adapter_path=student_adapter_path,
                )
                STUDENT_BOT.set_generation_config(
                    max_new_tokens=2048,
                    do_sample=False,
                )

            # Teacher is unnecessary for frozen test-only inference.
            if (
                not getattr(args, "test_only", False)
                and TEACHER_BOT is None
                and teacher_model_path
            ):
                print(
                    f"[worker {os.getpid()}] loading Teacher from "
                    f"{teacher_model_path} ..."
                )
                TEACHER_BOT = QwenLocal(
                    model_path=teacher_model_path,
                    adapter_path=None,
                )
                TEACHER_BOT.set_generation_config(
                    max_new_tokens=2048,
                    do_sample=False,
                )

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------
    def _to_internal_id(self, raw_id, is_user=True):
        mapping = self.data.user if is_user else self.data.item
        if raw_id is None:
            return None
        value = mapping.get(str(raw_id))
        return int(value) if value is not None else None

    def _to_raw_id(self, internal_id, is_user=True):
        mapping = self.data.id2user if is_user else self.data.id2item
        if internal_id is None:
            return None
        value = mapping.get(int(internal_id))
        return str(value) if value is not None else None

    def _get_item_display_name(self, internal_item_id):
        if internal_item_id is None:
            return "Unknown"

        name = None
        if hasattr(self.data, "get_item_meta"):
            try:
                meta = self.data.get_item_meta(internal_item_id, fields=None)
                if isinstance(meta, dict):
                    if meta.get("title"):
                        name = str(meta["title"])
                    elif meta.get("movie_title"):
                        name = str(meta["movie_title"])
            except Exception:
                name = None

        if not name:
            name = self._to_raw_id(internal_item_id, is_user=False) or "Unknown"

        return str(name).strip()

    # ------------------------------------------------------------------
    # base recommender / SASRec
    # ------------------------------------------------------------------
    def train(self, save_dir=None, root_dir=None, **kwargs):
        if self.tool_model is None:
            raise ValueError("tool_model (SASRec wrapper) was not injected from ARLib.")
        if (
            not hasattr(self.tool_model, "recbole_model")
            or self.tool_model.recbole_model is None
        ):
            raise ValueError("tool_model was not loaded successfully.")

        self.model = self.tool_model.recbole_model
        self.tool_model.arlib_dataset = self.data

        try:
            self.state_size = int(
                self.tool_model.recbole_config["MAX_ITEM_LIST_LENGTH"]
            )
        except Exception:
            self.state_size = getattr(self, "state_size", 50)

        self.precompute_embeddings_recbole()

        prior_csv_path = (
            os.path.join(save_dir, "prior_csv", "SASRec.csv")
            if save_dir
            else None
        )
        if prior_csv_path and os.path.exists(prior_csv_path):
            print(f"Found existing prior CSV: {prior_csv_path}; skip regeneration.")
        else:
            candidates_map = self._load_candidate_json(save_dir=root_dir)
            self._generate_prior_score_csv(
                candidates_map,
                output_dir=save_dir,
                use_candidates=self.use_candidate_prior,
            )

    def prepare_for_test(self, save_dir=None, root_dir=None):
        """Load frozen artifacts without entering any build/evolution path."""
        if self.tool_model is None:
            raise RuntimeError("Frozen SASRec tool was not injected.")
        if (
            not hasattr(self.tool_model, "recbole_model")
            or self.tool_model.recbole_model is None
        ):
            raise RuntimeError("Frozen SASRec checkpoint did not load correctly.")

        prior_path = os.path.join(save_dir, "prior_csv", "SASRec.csv")
        candidate_path = os.path.join(root_dir, "candidates.json")
        active_retriever_path = (
            str(getattr(self.args, "retriever_checkpoint", "") or "").strip()
            or self.retriever_projector_path
        )
        self.active_retriever_path = active_retriever_path

        required = [
            prior_path,
            candidate_path,
            self.skill_bank.manifest_path,
            active_retriever_path,
        ]
        missing = [path for path in required if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "Missing frozen inference artifacts: " + ", ".join(missing)
            )

        self.model = self.tool_model.recbole_model
        self.tool_model.arlib_dataset = self.data
        try:
            self.state_size = int(
                self.tool_model.recbole_config["MAX_ITEM_LIST_LENGTH"]
            )
        except Exception:
            self.state_size = 50

        self.precompute_embeddings_recbole()
        print(f"[ARS] TEST-ONLY Skill Bank: {self.skill_bank.manifest_path}")
        print(f"[ARS] TEST-ONLY retriever checkpoint: {active_retriever_path}")

    # ------------------------------------------------------------------
    # Skill Bank lifecycle
    # ------------------------------------------------------------------
    def bootstrap_skill_bank_from_train(self, save_dir=None, root_dir=None):
        if not self.enable_memskill:
            print("[ARS] Skill framework disabled; skip Skill Bank initialization.")
            return

        if self.skill_bank.has_user_skills() and self.skill_bank.has_functional_skills():
            print("[ARS] Existing complete Skill Bank detected; skip initialization.")
            return

        agent_args = self._prepare_agent_args()
        if getattr(agent_args, "teacher_bot", None) is None:
            raise RuntimeError(
                "Teacher model is required for Skill Bank initialization. "
                "Set --teacher_model_path."
            )

        agent_dataset = self._load_agent_dataset(root_dir, stage="train")
        prior_recs = self._load_prior_recs(save_dir=save_dir)
        self.id2name_cache = {
            iid: self._get_item_display_name(iid)
            for iid in self.data.id2item.keys()
        }

        max_users = self._parse_user_limit(
            getattr(self.args, "train_user_num", "all")
        )
        train_users = self._get_train_users(limit=max_users)

        train_histories = []
        for u_raw in tqdm(train_users, desc="Init Skill Bank"):
            sample = agent_dataset.get(u_raw)
            if not sample:
                continue
            data_sample = self._prepare_agent_sample(
                u_raw,
                sample,
                agent_args,
                prior_recs,
            )
            if data_sample is None:
                continue
            history_items = [
                item.strip()
                for item in data_sample.get("seq_names", "").split(",")
                if item.strip()
            ]
            if len(history_items) >= 2:
                train_histories.append(history_items)

        self.designer.initialize_from_train_histories(
            user_histories=train_histories,
            budget=getattr(self.args, "initial_user_skill_budget", None),
            cluster_size=getattr(self.args, "user_skill_cluster_size", 8),
            similarity_threshold=float(
                getattr(self.args, "skill_similarity_threshold", 0.72)
            ),
            llm_args=agent_args,
            local_bot=getattr(agent_args, "teacher_bot", None),
        )

        print(
            f"[ARS] Skill Bank initialized with "
            f"{len(self.skill_bank.user_skills)} user-profile skills and "
            f"{len(self.skill_bank.functional_skills)} functional skills."
        )

    def update_skill_bank_on_train(self, save_dir=None, root_dir=None):
        if not self.enable_memskill:
            print("[ARS] Skill framework disabled; skip train-stage evolution.")
            return

        rounds = max(1, int(getattr(self.args, "skill_update_rounds", 1)))
        agent_args = self._prepare_agent_args()
        self.designer.set_local_bot(getattr(agent_args, "teacher_bot", None))

        if getattr(agent_args, "local_bot", None) is None:
            raise RuntimeError("Student model is required for train-stage interaction.")
        if getattr(agent_args, "teacher_bot", None) is None:
            raise RuntimeError("Teacher model is required for Skill Bank evolution.")

        agent_dataset = self._load_agent_dataset(root_dir, stage="train")
        prior_recs = self._load_prior_recs(save_dir=save_dir)
        self.id2name_cache = {
            iid: self._get_item_display_name(iid)
            for iid in self.data.id2item.keys()
        }

        max_users = self._parse_user_limit(
            getattr(self.args, "train_user_num", "all")
        )
        train_users = self._get_train_users(limit=max_users)
        if not train_users:
            print("[ARS] No eligible train users found for skill update pass.")
            return

        os.makedirs(self.retriever_dir, exist_ok=True)
        with open(self.retrieval_logs_path, "w", encoding="utf-8"):
            pass

        print(
            f"[ARS] Running train update pass on {len(train_users)} users "
            f"for {rounds} round(s)."
        )
        retrieval_logs = []

        for round_idx in range(rounds):
            print(f"[ARS] Train update round {round_idx + 1}/{rounds}")

            for idx, u_raw in enumerate(
                tqdm(train_users, desc=f"Train skill pass {round_idx + 1}"),
                start=1,
            ):
                sample = agent_dataset.get(u_raw)
                if not sample:
                    continue

                data_sample = self._prepare_agent_sample(
                    u_raw,
                    sample,
                    agent_args,
                    prior_recs,
                )
                if data_sample is None:
                    continue

                gt_item_int = self._extract_train_gt_item(sample)
                result = self._run_agent_loop(
                    agent_args=agent_args,
                    data_sample=data_sample,
                    u_raw=u_raw,
                    gt_item_int=gt_item_int,
                    stage="train_update",
                    persist_memory=False,
                )

                success = bool(result.get("success", False))
                routed = result.get("routed")
                selected_all_skills = list(result.get("selected_skills", []))

                if selected_all_skills:
                    self.skill_bank.register_outcome(
                        selected_skill_names=selected_all_skills,
                        success=success,
                    )

                preference_query = (
                    getattr(routed, "preference_query", "")
                    if routed is not None
                    else ""
                )

                # Retriever supervision uses only user-profile skills.
                selected_user_skills = []
                if routed is not None and getattr(routed, "user_skills", None):
                    selected_user_skills = [
                        skill.name for skill in routed.user_skills
                    ]

                candidate_user_skill_names = [
                    skill.name for skill in self.skill_bank.user_skills
                ]
                row = {
                    "user_id": str(u_raw),
                    "round_idx": int(round_idx + 1),
                    "preference_query": preference_query,
                    "selected_skills": selected_user_skills,
                    "candidate_skill_names": candidate_user_skill_names,
                    "success": success,
                    "rec_item": result.get("rec_item", "") or "",
                    "ground_truth": (
                        self._get_item_display_name(gt_item_int)
                        if gt_item_int is not None
                        else ""
                    ),
                }
                retrieval_logs.append(row)
                self._append_retrieval_log(row)

                if not success:
                    failure_case = FailureCase(
                        user_id=str(u_raw),
                        stage="train_update",
                        user_history=data_sample.get("seq_names", ""),
                        candidates=data_sample.get("cans_str", ""),
                        interaction_history=result.get("interaction_history", []),
                        final_recommendation=result.get("rec_item", "") or "",
                        ground_truth=(
                            self._get_item_display_name(gt_item_int)
                            if gt_item_int is not None
                            else ""
                        ),
                        user_feedback=result.get("user_reason", "") or "",
                        # Failure e_u records the full selected skill set S_u.
                        selected_skills=selected_all_skills,
                        preference_query=preference_query,
                        state_summary=(
                            getattr(routed, "state_summary", {})
                            if routed is not None
                            else {}
                        ),
                        difficulty=float(result.get("epochs_used", 1)),
                    )
                    self.failure_buffer.add_case(failure_case)

                if self.failure_buffer.should_trigger(
                    getattr(self.args, "skill_update_interval", 10)
                ):
                    self.skill_evolution_call_count += 1
                    updated = self.designer.evolve_from_failures(
                        self.failure_buffer,
                        limit=getattr(
                            self.args,
                            "max_skill_cases_per_round",
                            20,
                        ),
                    )
                    print(
                        f"[ARS] Skill evolution triggered at train step {idx}: "
                        f"updated={updated}"
                    )

        self.retrieval_interaction_logs = retrieval_logs
        self._train_retriever_from_interaction_logs(retrieval_logs)

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------
    def test(self, save_dir=None, root_dir=None, **kwargs):
        print("Evaluating CRAFTS with a frozen Skill Bank and retriever...")
        self.agent_results = {}

        agent_args = self._prepare_agent_args()
        agent_dataset = self._load_agent_dataset(root_dir, stage="test")
        prior_recs = self._load_prior_recs(save_dir=save_dir)
        self.id2name_cache = {
            iid: self._get_item_display_name(iid)
            for iid in self.data.id2item.keys()
        }

        test_users = list(agent_dataset.keys())
        user_manifest = str(
            getattr(self.args, "test_user_file", "") or ""
        ).strip()
        if user_manifest:
            if not os.path.isfile(user_manifest):
                raise FileNotFoundError(
                    f"Test user manifest not found: {user_manifest}"
                )
            with open(user_manifest, "r", encoding="utf-8") as handle:
                requested_users = [
                    line.strip() for line in handle if line.strip()
                ]
            missing_users = [
                user for user in requested_users if user not in agent_dataset
            ]
            if missing_users:
                raise RuntimeError(
                    f"{len(missing_users)} manifest users are absent from the frozen test set"
                )
            test_users = requested_users

        limit = self._parse_user_limit(
            getattr(self.args, "test_user_num", "all")
        )
        if limit is not None:
            test_users = test_users[:limit]

        skill_top_k = int(
            getattr(self.args, "user_skill_top_k", 2)
            + getattr(self.args, "functional_skill_top_k", 2)
        )

        with open(self.skill_bank.manifest_path, "rb") as handle:
            skill_bank_sha256 = hashlib.sha256(handle.read()).hexdigest()

        user_skill_count = len(self.skill_bank.user_skills)
        functional_skill_count = len(self.skill_bank.functional_skills)
        variant = "CRAFTS"

        print(f"Testing on {len(test_users)} users using ThreadPool...")
        print(
            json.dumps(
                {
                    "dataset": getattr(self.args, "dataset", ""),
                    "variant": variant,
                    "student_backbone": getattr(
                        self.args,
                        "local_model_path",
                        "",
                    ),
                    "student_lora": getattr(
                        self.args,
                        "student_lora_path",
                        "",
                    ),
                    "teacher_backbone": getattr(
                        self.args,
                        "teacher_model_path",
                        "",
                    ),
                    "semantic_encoder": getattr(
                        self.args,
                        "embedding_model",
                        "sentence-transformers/all-MiniLM-L6-v2",
                    ),
                    "retriever_checkpoint": self.active_retriever_path,
                    "skill_bank": self.skill_bank.manifest_path,
                    "skill_bank_sha256": skill_bank_sha256,
                    "top_k": skill_top_k,
                    "seed": int(getattr(self.args, "seed", 2018)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        gt_map = self._build_test_gt()
        report_k = int(getattr(self.args, "report_k", self.max_N))
        report_interval = int(getattr(self.args, "report_interval", 100))
        processed = 0
        hits = 0
        error_count = 0
        parse_success_count = 0

        output_dir = str(
            getattr(self.args, "eval_output_dir", "") or save_dir
        )
        os.makedirs(output_dir, exist_ok=True)

        inference_path = os.path.join(output_dir, "inference_results.jsonl")
        retrieval_path = os.path.join(output_dir, "skill_retrieval.jsonl")
        errors_path = os.path.join(output_dir, "errors.jsonl")
        incorrect_path = os.path.join(
            output_dir,
            "incorrect_recommendations.jsonl",
        )
        for path in (
            inference_path,
            retrieval_path,
            errors_path,
            incorrect_path,
        ):
            with open(path, "w", encoding="utf-8"):
                pass

        with open(
            os.path.join(output_dir, "run_command.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(" ".join(sys.argv) + "\n")

        config_payload = dict(vars(self.args))
        config_payload.update(
            {
                "variant": variant,
                "semantic_encoder": getattr(
                    self.args,
                    "embedding_model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ),
                "skill_bank_path": self.skill_bank.manifest_path,
                "skill_bank_sha256": skill_bank_sha256,
                "retriever_checkpoint": self.active_retriever_path,
                "test_user_count": len(test_users),
                "skill_top_k": skill_top_k,
                "user_profile_skill_count": user_skill_count,
                "functional_skill_count": functional_skill_count,
                "candidate_mapping_policy": "exact_normalized_candidate_title",
            }
        )
        with open(
            os.path.join(output_dir, "experiment_config.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(config_payload, handle, ensure_ascii=False, indent=2)

        max_workers = int(getattr(self.args, "agent_workers", 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_user_agent,
                    u_raw,
                    agent_args,
                    agent_dataset,
                    prior_recs,
                ): u_raw
                for u_raw in test_users
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Agent Reasoning",
            ):
                try:
                    result = future.result()
                except Exception as exc:
                    error_count += 1
                    with open(errors_path, "a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "user_id": str(futures[future]),
                                    "error": repr(exc),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    result = None

                processed += 1
                if result:
                    u_key, _, internal_recs_list, details = result
                    candidate_match_success = bool(
                        details.get("candidate_match_success", False)
                    )
                    if internal_recs_list:
                        u_int = self._to_internal_id(u_key, is_user=True)
                        if u_int is not None:
                            self.agent_results[u_int] = internal_recs_list

                    gt_i = gt_map.get(str(u_key))
                    gt_name = (
                        self.id2name_cache.get(gt_i, f"Unknown_{gt_i}")
                        if gt_i is not None
                        else "None"
                    )
                    hit = bool(
                        candidate_match_success
                        and gt_i is not None
                        and int(gt_i) in set(internal_recs_list[:report_k])
                    )
                    hits += int(hit)

                    routed = details.get("routed")
                    retrieval_row = {
                        "user_id": str(u_key),
                        "query": (
                            getattr(routed, "preference_query", "")
                            if routed
                            else ""
                        ),
                        "candidate_item_count": int(
                            details.get("candidate_count", 0)
                        ),
                        "user_profile_skill_count": user_skill_count,
                        "functional_skill_count": functional_skill_count,
                        "top_k": (
                            len(getattr(routed, "all_skills", []))
                            if routed
                            else 0
                        ),
                        "selected_user_skills": (
                            [skill.name for skill in routed.user_skills]
                            if routed
                            else []
                        ),
                        "selected_functional_skills": (
                            [skill.name for skill in routed.functional_skills]
                            if routed
                            else []
                        ),
                        "semantic_encoder": getattr(
                            self.args,
                            "embedding_model",
                            "sentence-transformers/all-MiniLM-L6-v2",
                        ),
                        "score_function": "trainable_projector_dot_product",
                        "variant": variant,
                        "seed": int(getattr(self.args, "seed", 2018)),
                        "skill_bank_sha256": skill_bank_sha256,
                    }
                    inference_row = {
                        "user_id": str(u_key),
                        "ground_truth_item_id": (
                            int(gt_i) if gt_i is not None else None
                        ),
                        "ground_truth": gt_name,
                        "recommendation": details.get("rec_item"),
                        "recommended_internal_ids": internal_recs_list,
                        "candidate_match_success": candidate_match_success,
                        "hit_at_1": hit,
                        "raw_model_output": details.get("raw_model_output"),
                        "parse_success": candidate_match_success,
                        "selected_skills": details.get("selected_skills", []),
                        "selected_skill_prompt": self._extract_selected_skill_prompt(
                            details.get("final_user_prompt", "")
                        ),
                        "final_system_prompt": details.get(
                            "final_system_prompt",
                            "",
                        ),
                        "final_user_prompt": details.get(
                            "final_user_prompt",
                            "",
                        ),
                        "interaction_history": details.get(
                            "interaction_history",
                            [],
                        ),
                    }

                    with open(retrieval_path, "a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(retrieval_row, ensure_ascii=False) + "\n"
                        )
                    with open(inference_path, "a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(inference_row, ensure_ascii=False) + "\n"
                        )

                    if candidate_match_success:
                        parse_success_count += 1
                        if not hit:
                            with open(
                                incorrect_path,
                                "a",
                                encoding="utf-8",
                            ) as handle:
                                handle.write(
                                    json.dumps(
                                        inference_row,
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                    else:
                        error_count += 1
                        with open(errors_path, "a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        "user_id": str(u_key),
                                        "error": (
                                            "recommendation_not_in_candidate_list"
                                            if details.get("rec_item")
                                            else "recommendation_parse_failed"
                                        ),
                                        "parsed_recommendation": details.get(
                                            "rec_item"
                                        ),
                                        "raw_model_output": details.get(
                                            "raw_model_output"
                                        ),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                if report_interval > 0 and (
                    processed % report_interval == 0
                    or processed == len(test_users)
                ):
                    tqdm.write(
                        f"[ARS][HR@{report_k}] processed={processed}/{len(test_users)} "
                        f"hits={hits} hr={hits / max(1, processed):.4f}"
                    )

        overall_metrics = self._compute_overall_topk_metrics()
        test_metrics = {
            "stage": "test",
            "ts": int(time.time()),
            "processed_users": int(processed),
            "hr_at_k": float(hits / max(1, processed)),
            "h1": float(hits / max(1, processed)),
            "hits_at_1": int(hits),
            "report_k": int(report_k),
            "overall_topk_metrics": overall_metrics,
            "parse_success_count": int(parse_success_count),
            "parse_failure_count": int(processed - parse_success_count),
            "error_count": int(error_count),
            "variant": variant,
            "frozen_skill_bank": True,
            "student_backbone_path": getattr(
                self.args,
                "local_model_path",
                "",
            ),
            "student_lora_path": getattr(
                self.args,
                "student_lora_path",
                "",
            ),
            "teacher_model_path": getattr(
                self.args,
                "teacher_model_path",
                "",
            ),
            "semantic_encoder": getattr(
                self.args,
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            "retriever_checkpoint": self.active_retriever_path,
            "skill_bank_path": self.skill_bank.manifest_path,
            "skill_bank_sha256": skill_bank_sha256,
            "user_profile_skill_count": user_skill_count,
            "functional_skill_count": functional_skill_count,
            "skill_top_k": skill_top_k,
            "random_seed": int(getattr(self.args, "seed", 2018)),
            "candidate_mapping_policy": "exact_normalized_candidate_title",
            "dataset": str(getattr(self.args, "dataset", "")),
            "model": str(getattr(self.args, "model", "")),
            "num_test_samples": int(processed),
        }

        metrics_dir = getattr(self.args, "eval_output_dir", "") or save_dir
        self._write_eval_metrics(metrics_dir, test_metrics, stage="test")
        with open(
            os.path.join(metrics_dir, "final_metrics.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(test_metrics, handle, ensure_ascii=False, indent=2)

        return self.agent_results

    # ------------------------------------------------------------------
    # train/test sample helpers
    # ------------------------------------------------------------------
    def _get_train_users(self, limit=None):
        users = set()

        if hasattr(self.data, "train_user_items"):
            users = set(self.data.train_user_items.keys())
        elif hasattr(self.data, "train_data") and self.data.train_data:
            for row in self.data.train_data:
                u_int = row[0] if isinstance(row, (list, tuple)) else row
                u_raw = self._to_raw_id(u_int, is_user=True)
                if u_raw:
                    users.add(str(u_raw))
        elif hasattr(self.data, "user_list"):
            users = set(self.data.user_list)
        elif hasattr(self.data, "user"):
            users = set(self.data.user.keys())

        users = sorted(str(user) for user in users)
        if limit and len(users) > limit:
            users = users[:limit]

        print(f"[ARS] Training on {len(users)} users.")
        return users

    def _prepare_agent_sample(self, u_raw, sample, agent_args, prior_recs):
        data_sample = sample.copy()
        data_sample["prior_answer"] = prior_recs.get(u_raw, "Unknown")

        u_int = self._to_internal_id(u_raw, is_user=True)
        if u_int is None:
            return None

        if "seq" in sample and sample["seq"] is not None:
            seq_int = list(sample["seq"])
        else:
            seq_int = self.data.get_user_history_by_internal_id(u_int) or []
        seq_int = seq_int[-self.state_size :]

        seq_names = [
            self.id2name_cache.get(item_id, "Unknown")
            for item_id in seq_int
        ]
        data_sample["seq_names"] = ", ".join(seq_names)
        if not data_sample.get("seq_str"):
            data_sample["seq_str"] = ", ".join(seq_names)

        cands_source = sample.get("cans", [])
        if not cands_source:
            return None

        if isinstance(cands_source[0], (int, np.integer)):
            cands_int = [int(item) for item in cands_source]
        else:
            cands_int = [
                self._to_internal_id(item, is_user=False)
                for item in cands_source
            ]
            cands_int = [item for item in cands_int if item is not None]

        if not cands_int:
            return None

        cand_names = [
            self.id2name_cache.get(item_id, "Unknown")
            for item_id in cands_int
        ]
        data_sample["cans_str"] = ", ".join(cand_names)
        data_sample["len_cans"] = len(cands_int)
        data_sample["cans"] = cands_int
        data_sample["seq"] = np.array(seq_int, dtype=np.int64)
        data_sample["len_seq"] = len(seq_int)
        return data_sample

    def _extract_train_gt_item(self, sample):
        for key in [
            "next",
            "ground_truth_id",
            "target_item",
            "pos_item",
            "item",
        ]:
            if key not in sample or sample[key] is None:
                continue
            value = sample[key]
            try:
                return int(value)
            except Exception:
                converted = self._to_internal_id(value, is_user=False)
                if converted is not None:
                    return int(converted)
        return None

    # ------------------------------------------------------------------
    # agent execution
    # ------------------------------------------------------------------
    def _process_single_user_agent(
        self,
        u_raw,
        agent_args,
        agent_dataset,
        prior_recs,
    ):
        sample = agent_dataset.get(u_raw)
        if not sample:
            return None

        data_sample = sample.copy()
        data_sample["prior_answer"] = prior_recs.get(u_raw, "Unknown")

        u_int = self._to_internal_id(u_raw, is_user=True)
        if u_int is None:
            return None

        seq_int = self.data.get_user_history_by_internal_id(u_int) or []
        seq_int = seq_int[-self.state_size :]
        seq_names = [
            self.id2name_cache.get(item_id, "Unknown")
            for item_id in seq_int
        ]
        data_sample["seq_names"] = ", ".join(seq_names)
        if not data_sample.get("seq_str"):
            data_sample["seq_str"] = ", ".join(seq_names[-10:])

        cands_source = sample.get("cans", [])
        if not cands_source:
            return None

        if isinstance(cands_source[0], (int, np.integer)):
            cands_int = [int(item) for item in cands_source]
        else:
            cands_int = [
                self._to_internal_id(item, is_user=False)
                for item in cands_source
            ]
            cands_int = [item for item in cands_int if item is not None]

        if not cands_int:
            return None

        cand_names = [
            self.id2name_cache.get(item_id, "Unknown")
            for item_id in cands_int
        ]
        data_sample["cans_str"] = ", ".join(cand_names)
        data_sample["len_cans"] = len(cands_int)
        data_sample["cans"] = cands_int
        data_sample["seq"] = np.array(seq_int, dtype=np.int64)
        data_sample["len_seq"] = len(seq_int)

        gt_i = self._build_test_gt().get(str(u_raw))
        result = self._run_agent_loop(
            agent_args=agent_args,
            data_sample=data_sample,
            u_raw=u_raw,
            gt_item_int=gt_i,
            stage="test",
            persist_memory=False,
        )

        rec_item_name = result.get("rec_item")
        final_order_ints = []
        candidate_match_success = False

        if rec_item_name:
            normalized_rec = self._normalize_item_name(rec_item_name)
            for idx, name in enumerate(cand_names):
                if normalized_rec == self._normalize_item_name(name):
                    candidate_match_success = True
                    final_order_ints.append(cands_int[idx])
                    break

        if candidate_match_success:
            for iid in cands_int:
                if iid not in final_order_ints:
                    final_order_ints.append(iid)

        result["candidate_match_success"] = candidate_match_success
        rec_list_entry = [
            (self._to_raw_id(iid, is_user=False), 100.0 - rank)
            for rank, iid in enumerate(final_order_ints[: self.max_N])
        ]
        return (
            u_raw,
            rec_list_entry,
            final_order_ints[: self.max_N],
            result,
        )

    def _run_agent_loop(
        self,
        agent_args,
        data_sample,
        u_raw,
        gt_item_int=None,
        stage="test",
        persist_memory=False,
    ):
        if not agent_args:
            return {
                "rec_item": data_sample.get("prior_answer", ""),
                "success": False,
                "selected_skills": [],
            }

        rec_agent = RecAgent(agent_args, mode="prior_rec", dataset=self.data)
        user_agent = UserModelAgent(
            agent_args,
            mode="prior_rec",
            dataset=self.data,
        )
        user_agent.id2name = getattr(self, "id2name_cache", {})

        rec_item_final = None
        user_reason = None
        rec_reason = None
        interaction_history = []
        epochs_used = 0

        for epoch in range(1, getattr(agent_args, "max_epoch", 4) + 1):
            epochs_used = epoch
            rec_reason, rec_item = self._retry_agent_action(
                rec_agent,
                data_sample,
                mode="rec",
            )
            if not rec_item:
                break

            rec_item_final = rec_item
            if getattr(agent_args, "max_epoch", 1) == 1:
                break

            user_reason, is_accepted = self._retry_agent_action(
                user_agent,
                data_sample,
                mode="user",
                context=(rec_reason, rec_item),
            )

            memory_record = {
                "epoch": epoch,
                "rec_reason": rec_reason,
                "rec_item": rec_item,
                "user_reason": user_reason,
                "Does the user accept?": is_accepted,
            }
            rec_agent.update_memory(memory_record)
            user_agent.update_memory(memory_record)
            interaction_history.append(memory_record)

            if is_accepted:
                break

        success = self._matches_ground_truth(rec_item_final, gt_item_int)
        return {
            "rec_item": rec_item_final,
            "rec_reason": rec_reason,
            "user_reason": user_reason,
            "epochs_used": epochs_used,
            "interaction_history": interaction_history,
            "selected_skills": rec_agent.get_last_selected_skills(),
            "routed": rec_agent.get_last_routed(),
            "success": success,
            "stage": stage,
            "raw_model_output": getattr(rec_agent, "last_response", None),
            "final_system_prompt": getattr(
                rec_agent,
                "last_system_prompt",
                "",
            ),
            "final_user_prompt": getattr(
                rec_agent,
                "last_user_prompt",
                "",
            ),
            "candidate_count": len(data_sample.get("cans", [])),
        }

    @staticmethod
    def _normalize_item_name(text):
        if text is None:
            return ""
        text = str(text).strip()
        text = text.replace("**", "").replace("*", "").replace("`", "")
        text = text.replace('"', "").replace("'", "")
        text = text.strip(" .,:;")
        return " ".join(text.lower().split())

    @staticmethod
    def _extract_selected_skill_prompt(final_user_prompt: str) -> str:
        if not final_user_prompt:
            return ""
        text = str(final_user_prompt)
        markers = [
            "## Selected Skill Guidance",
            "## Retrieved Skill Guidance",
        ]
        start = -1
        for marker in markers:
            start = text.find(marker)
            if start >= 0:
                break
        if start < 0:
            return ""

        next_marker = text.find("\nGiven the following", start)
        if next_marker < 0:
            next_marker = text.find("\nPlease", start)
        if next_marker < 0:
            return text[start:].strip()
        return text[start:next_marker].strip()

    # Backward-compatible alias for older analysis scripts.
    _extract_retrieved_skill_prompt = _extract_selected_skill_prompt

    def _matches_ground_truth(
        self,
        rec_item_name: Optional[str],
        gt_item_int: Optional[int],
    ) -> bool:
        """Exact normalized match for trajectory filtering and train success."""
        if not rec_item_name or gt_item_int is None:
            return False
        gt_name = self._get_item_display_name(int(gt_item_int))
        return self._normalize_item_name(rec_item_name) == self._normalize_item_name(
            gt_name
        )

    def _retry_agent_action(self, agent, sample, mode="rec", context=None):
        for retry_idx in range(3):
            try:
                if mode == "rec":
                    response = agent.act(sample)
                    reason, item = split_rec_reponse(response)
                    if item:
                        return reason, item
                else:
                    reason_rec, item_rec = context
                    response = agent.act(sample, reason_rec, item_rec)
                    reason, accepted = split_user_response(response)
                    if accepted is not None:
                        return reason, accepted
            except Exception as exc:
                print(
                    f"[Agent Error Retry {retry_idx + 1}/3] "
                    f"User: {sample.get('user_id', 'unknown')} | Error: {exc}"
                )
                time.sleep(1)
        return None, None

    # ------------------------------------------------------------------
    # retriever supervision
    # ------------------------------------------------------------------
    def _append_retrieval_log(self, row: dict):
        os.makedirs(self.retriever_dir, exist_ok=True)
        with open(self.retrieval_logs_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_retrieval_logs(self):
        logs = []
        if not os.path.exists(self.retrieval_logs_path):
            print(
                f"[ARS] No retrieval log file found: {self.retrieval_logs_path}"
            )
            return logs

        with open(self.retrieval_logs_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    logs.append(json.loads(line))
                except Exception as exc:
                    print(f"[ARS] Skip bad retrieval log line: {exc}")
        return logs

    def train_retriever_only(self):
        logs = self._load_retrieval_logs()
        if not logs:
            print("[ARS] No logs loaded; skip retriever-only training.")
            return
        self._train_retriever_from_interaction_logs(logs)

    def _train_retriever_from_interaction_logs(self, interaction_logs):
        if not interaction_logs:
            print("[ARS] No retrieval interaction logs; skip projector training.")
            return
        if (
            TrainableSkillRetriever is None
            or build_pairwise_records_from_logs is None
            or train_skill_retriever is None
        ):
            print("[ARS] skill_retriever module unavailable; skip projector training.")
            return
        if len(self.skill_bank.user_skills) < 2:
            print("[ARS] Fewer than 2 user-profile skills; skip projector training.")
            return

        records = build_pairwise_records_from_logs(
            interaction_logs=interaction_logs,
            skill_bank=self.skill_bank,
            negatives_per_positive=int(
                getattr(self.args, "retriever_negatives_per_positive", 2)
            ),
            random_seed=int(getattr(self.args, "seed", 2018)),
            encoder_name=getattr(
                self.args,
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            negative_strategy=getattr(
                self.args,
                "retriever_negative_strategy",
                "bottom",
            ),
            bottom_ratio=float(
                getattr(self.args, "retriever_bottom_negative_ratio", 0.3)
            ),
            middle_ratio=float(
                getattr(self.args, "retriever_middle_negative_ratio", 0.3)
            ),
            positive_similarity_threshold=getattr(
                self.args,
                "retriever_positive_similarity_threshold",
                None,
            ),
            use_success_only=getattr(
                self.args,
                "retriever_use_success_only",
                True,
            ),
        )

        min_records = int(
            getattr(self.args, "retriever_min_train_records", 8)
        )
        if len(records) < min_records:
            print(
                f"[ARS] Only {len(records)} retriever records available "
                f"(< {min_records}); skip projector training."
            )
            return

        os.makedirs(self.retriever_dir, exist_ok=True)
        if save_records_jsonl is not None:
            save_records_jsonl(records, self.retriever_records_path)

        retriever = TrainableSkillRetriever(
            encoder_name=getattr(
                self.args,
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            hidden_dim=int(getattr(self.args, "retriever_hidden_dim", 256)),
            output_dim=int(getattr(self.args, "retriever_output_dim", 128)),
            dropout=float(getattr(self.args, "retriever_dropout", 0.1)),
        )

        history = train_skill_retriever(
            retriever=retriever,
            records=records,
            save_path=self.retriever_projector_path,
            batch_size=int(getattr(self.args, "retriever_batch_size", 32)),
            epochs=int(getattr(self.args, "retriever_epochs", 20)),
            learning_rate=float(
                getattr(self.args, "retriever_learning_rate", 1e-3)
            ),
            weight_decay=float(
                getattr(self.args, "retriever_weight_decay", 1e-5)
            ),
            margin=getattr(self.args, "retriever_margin", None),
            temperature=float(
                getattr(self.args, "retriever_temperature", 1.0)
            ),
            grad_clip_norm=float(
                getattr(self.args, "retriever_grad_clip_norm", 1.0)
            ),
            num_workers=int(
                getattr(self.args, "retriever_num_workers", 0)
            ),
            verbose=True,
        )

        with open(
            self.retriever_history_path,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "num_logs": len(interaction_logs),
                    "num_records": len(records),
                    "history": history,
                    "projector_path": self.retriever_projector_path,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"[ARS] Trained skill retriever with {len(records)} pairwise records. "
            f"Saved to {self.retriever_projector_path}"
        )

    # ------------------------------------------------------------------
    # teacher trajectories
    # ------------------------------------------------------------------
    @staticmethod
    def _format_skill_for_distill(skill):
        parts = [
            f"Skill Name: {getattr(skill, 'name', '')}",
            f"Description: {getattr(skill, 'description', '')}",
        ]

        if hasattr(skill, "core_preference"):
            parts.extend(
                [
                    "Core Preferences: "
                    + ", ".join(getattr(skill, "core_preference", [])),
                    "Secondary Preferences: "
                    + ", ".join(getattr(skill, "secondary_preference", [])),
                    "Exploration Boundary: "
                    + ", ".join(getattr(skill, "exploration_boundary", [])),
                    "Ranking Guidance: "
                    + ", ".join(getattr(skill, "ranking_guidance", [])),
                    "Constraints: "
                    + ", ".join(getattr(skill, "constraints", [])),
                ]
            )
        else:
            parts.extend(
                [
                    f"Purpose: {getattr(skill, 'purpose', '')}",
                    "When to Use: "
                    + ", ".join(getattr(skill, "when_to_use", [])),
                    "Input Signals: "
                    + ", ".join(getattr(skill, "input_signals", [])),
                    "How to Apply: "
                    + ", ".join(getattr(skill, "how_to_apply", [])),
                    "Constraints: "
                    + ", ".join(getattr(skill, "constraints", [])),
                ]
            )

        return "\n".join(part for part in parts if part and part.strip())

    def _build_distill_teacher_prompt(
        self,
        history_items,
        candidate_items,
        prior_answer,
        routed,
    ):
        user_skills_text = "\n\n".join(
            self._format_skill_for_distill(skill)
            for skill in getattr(routed, "user_skills", [])
        ) or "None"

        functional_skills_text = "\n\n".join(
            self._format_skill_for_distill(skill)
            for skill in getattr(routed, "functional_skills", [])
        ) or "None"

        system_prompt = """
You are a skill-aware recommendation teacher.

Given the current recommendation context and selected transferable skills,
demonstrate how to selectively apply relevant skill signals to preference analysis,
candidate comparison, and final ranking.

Rules:
1. Recommend exactly one item from Candidate Items.
2. Final Recommendation must copy the candidate name exactly.
3. Candidate Comparison may discuss only Candidate Items.
4. Use user-profile and functional skills only when they are applicable.
5. Ignore weak, irrelevant, or conflicting skill signals rather than forcing all skills.
6. Ground skill use in the current user history and candidate set.
7. Do not hallucinate unsupported item facts.
8. Follow the requested output format exactly and do not output JSON.
"""

        user_prompt = f"""
User History:
{", ".join(history_items)}

Candidate Items:
{", ".join(candidate_items)}

Prior Recommendation:
{prior_answer or "None"}

Preference Query:
{getattr(routed, "preference_query", "")}

Selected User-Profile Skills:
{user_skills_text}

Selected Functional Skills:
{functional_skills_text}

Output format:
Preference Analysis: ...
Useful Skill Signals: ...
Candidate Comparison: ...
Final Recommendation: <one exact candidate item>
Reason: ...
"""
        return system_prompt.strip(), user_prompt.strip()

    def _call_teacher_llm(self, system_prompt, user_prompt, agent_args):
        bot = getattr(agent_args, "teacher_bot", None)
        lock = getattr(agent_args, "teacher_bot_lock", None)

        if bot is None:
            print("[ARS] No Teacher LLM found; skip distillation sample.")
            return None

        try:
            if lock is None:
                return bot.chat(system_prompt, user_prompt)
            with lock:
                return bot.chat(system_prompt, user_prompt)
        except Exception as exc:
            print("[ARS] Teacher LLM failed:", repr(exc))
            return None

    @staticmethod
    def _parse_teacher_final_item(response):
        if not response:
            return ""
        for line in str(response).splitlines():
            line = line.strip()
            if line.lower().startswith("final recommendation:"):
                return line.split(":", 1)[1].strip()
        return ""

    def _append_distill_record(self, record):
        os.makedirs(self.distill_dir, exist_ok=True)
        with open(self.distill_data_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate_skill_distill_data(self, save_dir=None, root_dir=None):
        if not self.enable_memskill:
            print("[ARS] Skill framework disabled; skip distillation data generation.")
            return

        agent_args = self._prepare_agent_args()
        if getattr(agent_args, "teacher_bot", None) is None:
            raise RuntimeError(
                "Teacher model is required for teacher-trajectory generation."
            )

        agent_dataset = self._load_agent_dataset(root_dir, stage="train")
        prior_recs = self._load_prior_recs(save_dir=save_dir)
        self.id2name_cache = {
            iid: self._get_item_display_name(iid)
            for iid in self.data.id2item.keys()
        }

        max_users = self._parse_user_limit(
            getattr(self.args, "distill_user_num", "all")
        )
        train_users = self._get_train_users(limit=max_users)
        if not train_users:
            print("[ARS] No train users for distillation.")
            return

        router_agent = RecAgent(
            agent_args,
            mode="prior_rec",
            dataset=self.data,
        )
        if not getattr(router_agent, "router", None):
            print("[ARS] No skill router available; skip distillation.")
            return

        keep_success_only = bool(
            getattr(self.args, "distill_keep_success_only", True)
        )

        os.makedirs(self.distill_dir, exist_ok=True)
        with open(self.distill_data_path, "w", encoding="utf-8"):
            pass

        saved = 0
        total = 0
        print(
            f"[ARS] Generating teacher trajectories for {len(train_users)} users..."
        )

        for u_raw in tqdm(train_users, desc="Generate teacher trajectories"):
            sample = agent_dataset.get(u_raw)
            if not sample:
                continue

            data_sample = self._prepare_agent_sample(
                u_raw,
                sample,
                agent_args,
                prior_recs,
            )
            if data_sample is None:
                continue

            gt_item_int = self._extract_train_gt_item(sample)
            gt_name = (
                self._get_item_display_name(gt_item_int)
                if gt_item_int is not None
                else ""
            )

            history_items = [
                item.strip()
                for item in data_sample.get("seq_names", "").split(",")
                if item.strip()
            ]
            candidate_items = [
                item.strip()
                for item in data_sample.get("cans_str", "").split(",")
                if item.strip()
            ]
            prior_answer = data_sample.get("prior_answer", "")

            if not history_items or not candidate_items:
                continue

            routed = router_agent.router.select_skills(
                history_items=history_items,
                interaction_memory="",
                candidate_items=candidate_items,
                prior_answer=prior_answer,
            )

            system_prompt, user_prompt = self._build_distill_teacher_prompt(
                history_items=history_items,
                candidate_items=candidate_items,
                prior_answer=prior_answer,
                routed=routed,
            )
            teacher_output = self._call_teacher_llm(
                system_prompt,
                user_prompt,
                agent_args,
            )
            teacher_rec = self._parse_teacher_final_item(teacher_output)
            total += 1

            if not teacher_output or not teacher_rec:
                continue

            success = self._matches_ground_truth(
                teacher_rec,
                gt_item_int,
            )
            if keep_success_only and not success:
                continue

            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": str(teacher_output).strip(),
                    },
                ],
                "metadata": {
                    "user_id": str(u_raw),
                    "ground_truth": gt_name,
                    "teacher_rec": teacher_rec,
                    "success": success,
                    "preference_query": getattr(
                        routed,
                        "preference_query",
                        "",
                    ),
                    "selected_user_skills": [
                        skill.name
                        for skill in getattr(routed, "user_skills", [])
                    ],
                    "selected_functional_skills": [
                        skill.name
                        for skill in getattr(routed, "functional_skills", [])
                    ],
                },
            }
            self._append_distill_record(record)
            saved += 1

        print(
            f"[ARS] Teacher trajectory generation finished: "
            f"total={total}, saved={saved}, path={self.distill_data_path}"
        )

    # ------------------------------------------------------------------
    # metrics / IO
    # ------------------------------------------------------------------
    def _build_test_gt(self):
        gt = {}
        for row in getattr(self.data, "test_data", []):
            u_int = int(row[0])
            i_int = int(row[1])
            u_raw = self._to_raw_id(u_int, is_user=True)
            if u_raw is not None:
                gt[str(u_raw)] = i_int
        return gt

    def _get_topk_list(self):
        top_raw = str(getattr(self.args, "topK", "1"))
        ks = []
        for value in top_raw.split(","):
            try:
                k = int(value.strip())
            except Exception:
                continue
            if k > 0:
                ks.append(k)
        return sorted(set(ks or [1]))

    def _compute_overall_topk_metrics(self):
        topk = self._get_topk_list()
        gt_map = {}

        for row in getattr(self.data, "test_data", []):
            u_int, i_int = int(row[0]), int(row[1])
            gt_map.setdefault(u_int, set()).add(i_int)

        users = [user for user in self.agent_results if user in gt_map]
        if not users:
            return {
                "topk": topk,
                "num_eval_users": 0,
                "precision": [0.0 for _ in topk],
                "hr": [0.0 for _ in topk],
                "recall": [0.0 for _ in topk],
                "ndcg": [0.0 for _ in topk],
            }

        precision = []
        hr = []
        recall = []
        ndcg = []

        for k in topk:
            hit_cnt = 0
            hr_hit_users = 0
            total_prec_den = 0
            total_recall_den = 0
            dcg_sum = 0.0
            idcg_sum = 0.0

            for user in users:
                pred = list(self.agent_results.get(user, []))[:k]
                true_items = gt_map[user]
                intersection = set(pred) & true_items

                hit_cnt += len(intersection)
                total_prec_den += k
                total_recall_den += len(true_items)
                if intersection:
                    hr_hit_users += 1

                dcg_u = 0.0
                for rank, iid in enumerate(pred):
                    if iid in true_items:
                        dcg_u += 1.0 / math.log2(rank + 2)

                idcg_u = 0.0
                for rank in range(min(k, len(true_items))):
                    idcg_u += 1.0 / math.log2(rank + 2)

                dcg_sum += dcg_u
                idcg_sum += idcg_u

            precision.append(hit_cnt / max(1, total_prec_den))
            hr.append(hr_hit_users / max(1, len(users)))
            recall.append(hit_cnt / max(1, total_recall_den))
            ndcg.append(dcg_sum / max(1e-12, idcg_sum))

        return {
            "topk": topk,
            "num_eval_users": int(len(users)),
            "precision": precision,
            "hr": hr,
            "recall": recall,
            "ndcg": ndcg,
        }

    @staticmethod
    def _write_eval_metrics(save_dir, metrics, stage="test"):
        if not save_dir:
            return
        out_dir = os.path.join(save_dir, "eval_metrics")
        os.makedirs(out_dir, exist_ok=True)

        latest_path = os.path.join(out_dir, f"{stage}_metrics_latest.json")
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)

        history_path = os.path.join(out_dir, f"{stage}_metrics_history.jsonl")
        with open(history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # recommender / dataset helpers
    # ------------------------------------------------------------------
    def precompute_embeddings_recbole(self):
        if self.model is None or self.tool_model is None:
            return
        if (
            not hasattr(self.tool_model, "recbole_dataset")
            or self.tool_model.recbole_dataset is None
        ):
            return

        print("Caching RecBole item embeddings...")
        weight = self.model.item_embedding.weight.detach().cpu().numpy()
        dim = weight.shape[1]
        self.final_item_embs = np.zeros(
            (self.item_num, dim),
            dtype=np.float32,
        )
        rb_dataset = self.tool_model.recbole_dataset
        count = 0

        for rb_iid in range(
            getattr(rb_dataset, "item_num", weight.shape[0])
        ):
            try:
                raw_iid = rb_dataset.id2token(
                    rb_dataset.iid_field,
                    rb_iid,
                )
            except Exception:
                continue

            arlib_iid = self._to_internal_id(raw_iid, is_user=False)
            if arlib_iid is not None and 0 <= arlib_iid < self.item_num:
                self.final_item_embs[arlib_iid] = weight[rb_iid]
                count += 1

        print(f"Mapped {count} items into ARLib item space.")

    def _prepare_agent_args(self):
        global STUDENT_BOT, TEACHER_BOT
        global STUDENT_BOT_LOCK, TEACHER_BOT_LOCK

        ns = SimpleNamespace(**self.args.__dict__)
        ns.model = getattr(ns, "model", "Qwen3-4B")
        ns.maxlen = self.state_size
        ns.item_size = self.item_num
        ns.max_retry_num = 3
        ns.max_epoch = getattr(ns, "max_epoch", 4)
        ns.temperature = getattr(ns, "temperature", 0.2)
        ns.sasrec_tool = self.tool_model

        # Runtime Recommendation Agent / Student.
        ns.local_bot = STUDENT_BOT
        ns.local_bot_lock = STUDENT_BOT_LOCK

        # Teacher role for Skill Bank construction/evolution and trajectories.
        ns.teacher_bot = TEACHER_BOT
        ns.teacher_bot_lock = TEACHER_BOT_LOCK
        return ns

    def _load_agent_dataset(self, save_dir, stage="test"):
        if CRAFTSDataset is None:
            return {}

        candidates_map = self._load_candidate_json(save_dir)
        if not candidates_map:
            print(
                f"[ARS] Warning: no candidate map loaded for stage={stage}."
            )

        ds = CRAFTSDataset(
            data_dir=None,
            stage=stage,
            cans_num=self.max_N,
            arlib_dataset=self.data,
            candidates_map=candidates_map,
        )

        agent_data_dict = {}
        for sample in ds:
            u_int = sample.get("user_id")
            if u_int is None:
                continue
            u_raw = self._to_raw_id(u_int, is_user=True)
            if u_raw is not None:
                agent_data_dict[str(u_raw)] = sample

        print(
            f"[ARS] Loaded {len(agent_data_dict)} users from Agent Dataset "
            f"(stage={stage})."
        )
        return agent_data_dict

    @staticmethod
    def _load_prior_recs(save_dir):
        csv_path = (
            os.path.join(save_dir, "prior_csv", "SASRec.csv")
            if save_dir
            else None
        )
        recs = {}
        if not csv_path or not os.path.exists(csv_path):
            print("[ARS] Warning: prior CSV not found.")
            return recs

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            uid = (
                str(row["user_id"])
                if "user_id" in row
                else str(row.get("id"))
            )
            recs[uid] = str(row["generate"])
        return recs

    @staticmethod
    def _load_candidate_json(save_dir):
        candidate_path = (
            os.path.join(save_dir, "candidates.json") if save_dir else None
        )
        if candidate_path and os.path.exists(candidate_path):
            with open(candidate_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        print(f"[ARS] Warning: candidates.json not found in {save_dir}")
        return {}

    def _generate_prior_score_csv(
        self,
        candidates_map,
        output_dir=None,
        use_candidates: bool = False,
        topk: int = 20,
        filter_seen: bool = True,
    ):
        out_dir = os.path.join(output_dir, "prior_csv")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "SASRec.csv")
        print(f"Generating prior CSV -> {csv_path}")

        test_targets = {}
        for row in getattr(self.data, "test_data", []):
            u_raw = self._to_raw_id(row[0], is_user=True)
            i_raw = self._to_raw_id(row[1], is_user=False)
            if u_raw is not None and i_raw is not None:
                test_targets[str(u_raw)] = str(i_raw)

        target_users = list(test_targets.keys()) or (
            list(candidates_map.keys())
            if use_candidates and candidates_map
            else []
        )

        csv_data = []
        for u_raw in tqdm(target_users, desc="Scoring Prior Candidates"):
            u_int = self._to_internal_id(u_raw, is_user=True)
            if u_int is None:
                continue

            seq_int = self.data.get_user_history_by_internal_id(u_int) or []
            seq_int = seq_int[-self.state_size :]

            best_i_int = None
            best_raw = None
            try:
                if use_candidates:
                    cands_raw = candidates_map.get(u_raw, []) if candidates_map else []
                    cands_int = [
                        self._to_internal_id(cand, is_user=False)
                        for cand in cands_raw
                    ]
                    cands_int = [
                        item for item in cands_int if item is not None
                    ]
                    if not cands_int:
                        continue

                    scores_dict = self.tool_model.predict_with_sequence(
                        seq_int,
                        cands_int,
                    )
                    if not scores_dict:
                        continue
                    best_i_int = max(scores_dict, key=scores_dict.get)
                    best_raw = self._to_raw_id(best_i_int, is_user=False)
                else:
                    top_list = self.tool_model.recommend_topk_all_items(
                        item_sequence=seq_int,
                        k=topk,
                        filter_seen=filter_seen,
                        exclude_items=None,
                        with_scores=True,
                        with_names=False,
                    )
                    if not top_list:
                        continue
                    top1 = top_list[0]
                    best_i_int = top1.get("item_int")
                    best_raw = top1.get("item_raw")
                    if best_i_int is None and best_raw is not None:
                        best_i_int = self._to_internal_id(
                            str(best_raw),
                            is_user=False,
                        )
            except Exception:
                continue

            if best_i_int is None and best_raw is None:
                continue

            best_name = (
                self._get_item_display_name(best_i_int)
                if best_i_int is not None
                else str(best_raw)
            )
            if best_raw is None and best_i_int is not None:
                best_raw = self._to_raw_id(best_i_int, is_user=False)

            real_raw = test_targets.get(u_raw)
            real_i_int = (
                self._to_internal_id(real_raw, is_user=False)
                if real_raw
                else None
            )
            real_name = (
                self._get_item_display_name(real_i_int)
                if real_i_int is not None
                else None
            )

            csv_data.append(
                {
                    "id": u_raw,
                    "user_id": u_raw,
                    "generate": best_name,
                    "real": real_name,
                    "generate_raw_item": best_raw,
                    "real_raw_item": real_raw,
                }
            )

        if csv_data:
            pd.DataFrame(csv_data).to_csv(csv_path, index=False)
            print(f"Saved {len(csv_data)} rows to {csv_path}")
        else:
            print("[ARS] Warning: no prior CSV rows generated.")

    @staticmethod
    def _parse_user_limit(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str) and value.lower() == "all":
            return None
        try:
            limit = int(value)
            return None if limit <= 0 else limit
        except Exception:
            return None

    def load(self, save_path, root_path):
        return None

    def save(self, save_dir=None):
        return None
