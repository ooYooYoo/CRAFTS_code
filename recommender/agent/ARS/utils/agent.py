import json
import os
from typing import Optional

import torch

from recommender.agent.skill import SkillBank, SkillPromptExecutor, SkillRouter


def llm_request(system_prompt: str, user_prompt: str, args):
    """Unified inference entry.

    Uses the configured local language model.
    """
    bot = getattr(args, "local_bot", None)
    lock = getattr(args, "local_bot_lock", None)

    if bot is not None:
        try:
            if lock is None:
                return bot.chat(system_prompt, user_prompt)
            with lock:
                return bot.chat(system_prompt, user_prompt)
        except Exception as exc:
            print("[llm_request] local bot exception:", repr(exc))
            return None
    raise RuntimeError("CRAFTS requires --local_model_path for language-model inference")


class RecAgent:
    def __init__(self, args, mode: str = "prior_rec", dataset=None):
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.dataset = dataset
        self._last_routed = None
        self._last_selected_skills = []

        if self.dataset is None:
            raise ValueError("[RecAgent] dataset (ARLib DataLoader) is required for initialization!")

        self.enable_memskill = getattr(args, "enable_memskill", False)
        self.skill_bank: Optional[SkillBank] = None
        self.router: Optional[SkillRouter] = None
        if self.enable_memskill:
            dataset_name = str(getattr(self.dataset, "dataName", getattr(args, "dataset", "")))
            skill_bank_path = getattr(args, "skill_bank_path", "./artifacts/skills")
            self.skill_bank = SkillBank(skill_bank_path, dataset_scope=dataset_name)
            self.router = SkillRouter(
                skill_bank=self.skill_bank,
                user_skill_top_k=getattr(args, "user_skill_top_k", 2),
                functional_skill_top_k=getattr(args, "functional_skill_top_k", 2),
                similarity_threshold=getattr(args, "skill_similarity_threshold", 0.4),
                local_bot=getattr(args, "local_bot", None),
                retriever_path=(getattr(args, "retriever_checkpoint", "") or None),
            )
            print(
                f"[RecAgent] Skill mode enabled with {len(self.skill_bank.user_skills)} user skills "
                f"and {len(self.skill_bank.functional_skills)} functional skills"
            )

        self.load_prompt()

    def load_prompt(self):
        if self.mode != "prior_rec":
            raise ValueError(f"Invalid mode: {self.mode}")

        from constant.instruments_prior_model_prompt import (
            rec_build_memory,
            rec_memory_system_prompt,
            rec_memory_user_prompt,
            rec_system_prompt,
            rec_user_prompt,
        )

        self.rec_system_prompt = rec_system_prompt
        self.rec_user_prompt = rec_user_prompt
        self.rec_memory_system_prompt = rec_memory_system_prompt
        self.rec_memory_user_prompt = rec_memory_user_prompt
        self.rec_build_memory = rec_build_memory

    def act(self, data, reason=None, item=None):
        seq_names = data.get("seq_names", data.get("seq_str", ""))
        len_cans = data.get("len_cans", "0")
        cans_str = data.get("cans_str", "")
        prior_answer = data.get("prior_answer", "")
        user_id = data.get("user_id", "unknown")
        memory_str = "\n".join(str(m) for m in self.memory) if self.memory else ""

        if not self.memory:
            system_prompt = self.rec_system_prompt
            user_prompt = self.rec_user_prompt.format(seq_names, len_cans, cans_str, prior_answer)
        else:
            system_prompt = self.rec_memory_system_prompt
            user_prompt = self.rec_memory_user_prompt.format(seq_names, len_cans, cans_str, memory_str)

        self._last_routed = None
        self._last_selected_skills = []
        if self.enable_memskill and self.router and self.skill_bank:
            history_items = [item.strip() for item in seq_names.split(",") if item.strip()]
            candidate_items = [item.strip() for item in cans_str.split(",") if item.strip()]
            routed = self.router.select_skills(
                history_items=history_items,
                interaction_memory=memory_str,
                candidate_items=candidate_items,
                prior_answer=prior_answer,
            )
            self._last_routed = routed
            self._last_selected_skills = routed.all_skill_names
            if routed.all_skills:
                user_prompt = SkillPromptExecutor.build_rec_prompt(
                    base_prompt_template=user_prompt,
                    routed_skills=routed,
                )
            log_path = getattr(self.args, "prompt_log_path", "")
            if log_path:
                log_dir = os.path.dirname(log_path)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write("=" * 120 + "\n")
                    f.write(f"USER: {user_id}\n")

                    if routed is not None:
                        f.write("\n[Preference Query]\n")
                        f.write(str(routed.preference_query) + "\n")

                        f.write("\n[State Summary]\n")
                        f.write(str(routed.state_summary) + "\n")

                    if routed is not None and routed.user_skills:
                        f.write("\n[USER SKILLS]\n")
                        for s in routed.user_skills:
                            f.write(f"- {s.name}\n")

                    if routed is not None and routed.functional_skills:
                        f.write("\n[FUNCTIONAL SKILLS]\n")
                        for s in routed.functional_skills:
                            f.write(f"- {s.name}\n")

                    f.write("\n[FINAL PROMPT]\n")
                    f.write(user_prompt + "\n")

                    f.write("\n\n")
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        response = llm_request(system_prompt, user_prompt, self.args)
        self.last_response = response
        return response

    def get_last_selected_skills(self):
        return list(self._last_selected_skills)

    def get_last_routed(self):
        return self._last_routed

    def build_memory(self, info):
        return self.rec_build_memory.format(info["epoch"], info["rec_item"], info["rec_reason"], info["user_reason"])

    def update_memory(self, info):
        info["selected_skills"] = self.get_last_selected_skills()
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            for item in self.info_list:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_memory(self, path):
        self.info_list = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    self.info_list.append(json.loads(line))
        self.memory = [self.build_memory(info) for info in self.info_list]


class UserModelAgent:
    def __init__(self, args, mode="prior_rec", dataset=None):
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

        if self.dataset is None:
            raise ValueError("[UserModelAgent] dataset (ARLib DataLoader) is required!")

        self.load_prompt()
        self.id2name = {}
        self.name2id = {}
        self.build_id2name()
        self.sasrec_tool = args.sasrec_tool
        self.use_recbole_tool = True
        self.item_num = getattr(args, "item_size", 3494)
        self.seq_size = getattr(args, "maxlen", 50)

    def build_id2name(self):
        if not hasattr(self.dataset, "id2item"):
            print("[UserModelAgent] Warning: dataset has no id2item attribute.")
            return
        for internal_id, raw_id in self.dataset.id2item.items():
            name = str(raw_id)
            if hasattr(self.dataset, "get_item_meta"):
                try:
                    meta = self.dataset.get_item_meta(internal_id)
                    if meta and "movie_title" in meta:
                        name = str(meta["movie_title"])
                    elif meta and "title" in meta:
                        name = str(meta["title"])
                except Exception:
                    pass
            clean_name = name.split("|")[0].split(", The")[0].strip()
            self.id2name[internal_id] = clean_name
            self.name2id[clean_name] = internal_id

    def load_prompt(self):
        if self.mode != "prior_rec":
            raise ValueError(f"Invalid mode: {self.mode}")

        from constant.instruments_prior_model_prompt import (
            user_build_memory,
            user_build_memory_2,
            user_memory_system_prompt,
            user_memory_user_prompt,
            user_system_promt,
            user_user_prompt,
        )

        self.user_system_promt = user_system_promt
        self.user_user_prompt = user_user_prompt
        self.user_memory_system_prompt = user_memory_system_prompt
        self.user_memory_user_prompt = user_memory_user_prompt
        self.user_build_memory = user_build_memory
        self.user_build_memory_2 = user_build_memory_2

    def act(self, data, reason=None, item=None):
        if self.mode != "prior_rec":
            raise ValueError(f"Invalid mode: {self.mode}")

        model_output = self.model_generate(data["seq"], data["len_seq"], data["cans"])
        if not self.memory:
            system_prompt = self.user_system_promt.format(data["seq_names"], data["prior_answer"])
            user_prompt = self.user_user_prompt.format(data["cans_str"], model_output, item, reason)
        else:
            system_prompt = self.user_memory_system_prompt.format(data["seq_names"], data["prior_answer"])
            history_str = "\n".join(self.memory)
            user_prompt = self.user_memory_user_prompt.format(data["cans_str"], model_output, history_str, item, reason)
        return llm_request(system_prompt, user_prompt, self.args)

    def build_memory(self, info):
        if info["user_reason"] is not None:
            return self.user_build_memory.format(info["epoch"], info["rec_item"], info["rec_reason"], info["user_reason"])
        return self.user_build_memory_2.format(info["epoch"], info["rec_item"], info["rec_reason"])

    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            for item in self.info_list:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_memory(self, path):
        self.info_list = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    self.info_list.append(json.loads(line))
        self.memory = [self.build_memory(info) for info in self.info_list]

    def model_generate(self, seq, len_seq, candidates):
        return self._model_generate_recbole(seq, len_seq, candidates)

    def _model_generate_recbole(self, seq, len_seq, candidates):
        score_dict = self._score_recbole(seq, len_seq, candidates)
        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        len_ret = max(1, int(len(sorted_items) / 2))
        name_list = [name for name, _ in sorted_items[:len_ret]]
        return ", ".join(name_list)

    def _score_recbole(self, seq, len_seq, candidates):
        seq = [int(x) for x in list(seq)]
        candidates = [int(x) for x in candidates]
        arlib_dataset = self.sasrec_tool.arlib_dataset
        raw_seq = [arlib_dataset.id2item.get(i) for i in seq if arlib_dataset.id2item.get(i) is not None]
        raw_candidates = [arlib_dataset.id2item.get(i) for i in candidates if arlib_dataset.id2item.get(i) is not None]
        if not raw_seq or not raw_candidates:
            return {}
        score_dict_by_raw_id = self.sasrec_tool.predict_with_sequence(item_sequence=raw_seq, candidate_ids=raw_candidates)
        score_dict = {}
        for raw_id, score in score_dict_by_raw_id.items():
            iid_internal = arlib_dataset.item.get(str(raw_id))
            if iid_internal is not None:
                name = self.id2name.get(iid_internal, str(raw_id))
                score_dict[name] = score
        return score_dict
