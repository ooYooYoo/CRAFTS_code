import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .skill_bank import FunctionalSkill, SkillBank, UserProfileSkill

try:
    from .skill_retriever import TrainableSkillRetriever
except Exception:
    TrainableSkillRetriever = None


_FROZEN_ENCODER_CACHE = {}
_FROZEN_RETRIEVER_CACHE = {}
_FROZEN_MODEL_CACHE_LOCK = threading.Lock()


@dataclass
class RoutedSkills:
    """Skills and context selected for one recommendation instance."""

    preference_query: str
    state_summary: Dict[str, object]
    user_skills: List[UserProfileSkill] = field(default_factory=list)
    functional_skills: List[FunctionalSkill] = field(default_factory=list)
    skill_scores: List[Dict[str, object]] = field(default_factory=list)

    @property
    def all_skills(self) -> List[object]:
        return list(self.user_skills) + list(self.functional_skills)

    @property
    def all_skill_names(self) -> List[str]:
        return [skill.name for skill in self.all_skills]


class SkillRouter:
    """Select user-profile and functional skills for the current context.

    User-profile skills are retrieved with the trainable task-specific retriever
    when its checkpoint is available.

    Functional skills follow a different routing mechanism: the current
    recommendation context is summarized into open-ended observable signals, and
    each functional skill is selected by matching that context against the
    skill's own predefined ``when_to_use`` conditions and ``input_signals``.
    The controller therefore does not rely on a fixed global state vocabulary.
    Newly initialized or failure-evolved functional skills can become routable as
    soon as their own applicability conditions are stored in the Skill Bank.
    """

    def __init__(
        self,
        skill_bank: SkillBank,
        user_skill_top_k: int = 2,
        functional_skill_top_k: int = 2,
        similarity_threshold: float = 0.4,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_bot=None,
        retriever=None,
        retriever_path: Optional[str] = None,
        retriever_hidden_dim: int = 256,
        retriever_output_dim: int = 128,
        retriever_dropout: float = 0.1,
        retriever_use_success_bias: bool = False,
    ):
        self.skill_bank = skill_bank
        self.user_skill_top_k = int(user_skill_top_k)
        self.functional_skill_top_k = int(functional_skill_top_k)
        self.similarity_threshold = float(similarity_threshold)
        self.local_bot = local_bot
        self.embedding_model_name = embedding_model_name
        self.retriever = retriever
        self.retriever_use_success_bias = bool(retriever_use_success_bias)
        self.skill_pool_audit = self._audit_skill_pool()
        self._last_user_skill_scores: List[Dict[str, object]] = []
        self._last_functional_skill_scores: List[Dict[str, object]] = []

        default_retriever_path = None
        if getattr(self.skill_bank, "skill_dir", None):
            default_retriever_path = os.path.join(
                self.skill_bank.skill_dir,
                "retriever",
                "projector.pt",
            )
        self.retriever_path = retriever_path or default_retriever_path

        share_frozen = os.environ.get("CRAFTS_SHARE_FROZEN_RETRIEVER", "0") == "1"
        encoder_key = str(embedding_model_name)
        with _FROZEN_MODEL_CACHE_LOCK:
            self.embedding_model = (
                _FROZEN_ENCODER_CACHE.get(encoder_key) if share_frozen else None
            )
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer(embedding_model_name)
                if hasattr(self.embedding_model, "eval"):
                    self.embedding_model.eval()
                if share_frozen:
                    _FROZEN_ENCODER_CACHE[encoder_key] = self.embedding_model

        if (
            self.retriever is None
            and TrainableSkillRetriever is not None
            and self.retriever_path
            and os.path.exists(self.retriever_path)
        ):
            try:
                retriever_key = (
                    str(embedding_model_name),
                    os.path.abspath(self.retriever_path),
                    int(retriever_hidden_dim),
                    int(retriever_output_dim),
                    float(retriever_dropout),
                )
                with _FROZEN_MODEL_CACHE_LOCK:
                    cached_retriever = (
                        _FROZEN_RETRIEVER_CACHE.get(retriever_key)
                        if share_frozen
                        else None
                    )
                    if cached_retriever is None:
                        cached_retriever = TrainableSkillRetriever(
                            encoder_name=embedding_model_name,
                            hidden_dim=retriever_hidden_dim,
                            output_dim=retriever_output_dim,
                            dropout=retriever_dropout,
                        )
                        cached_retriever.load_projector(self.retriever_path)
                        cached_retriever.requires_grad_(False)
                        cached_retriever.eval()
                        if share_frozen:
                            _FROZEN_RETRIEVER_CACHE[retriever_key] = cached_retriever
                    self.retriever = cached_retriever
                print(f"[SkillRouter] Loaded trainable retriever from {self.retriever_path}")
            except Exception as exc:
                print(f"[SkillRouter] Failed to load trainable retriever: {exc}")
                self.retriever = None

        self._log_skill_pool()

    @staticmethod
    def _normalized_skill_type(skill) -> str:
        value = str(getattr(skill, "skill_type", "") or "").lower()
        return re.sub(r"[^a-z0-9]+", "_", value).strip("_")

    @classmethod
    def _is_user_profile_skill(cls, skill) -> bool:
        normalized = cls._normalized_skill_type(skill)
        aliases = {"user_profile", "userprofile", "profile_skill", "user_profile_skill"}
        class_name = skill.__class__.__name__.lower()
        return (
            normalized in aliases
            or "userprofile" in class_name
            or hasattr(skill, "core_preference")
        )

    @classmethod
    def _is_functional_skill(cls, skill) -> bool:
        normalized = cls._normalized_skill_type(skill)
        return (
            normalized in {"functional", "functional_skill"}
            or "functional" in skill.__class__.__name__.lower()
        )

    def _audit_skill_pool(self) -> Dict[str, object]:
        all_skills = list(self.skill_bank.user_skills) + list(
            self.skill_bank.functional_skills
        )
        user_skills = []
        functional_skills = []
        unknown_skills = []
        ambiguous_skills = []

        for skill in all_skills:
            is_user = self._is_user_profile_skill(skill)
            is_functional = self._is_functional_skill(skill)
            row = {
                "skill_id": str(getattr(skill, "skill_id", "")),
                "name": str(getattr(skill, "name", "")),
                "skill_type": str(getattr(skill, "skill_type", "")),
            }
            if is_user and is_functional:
                ambiguous_skills.append(row)
            elif is_user:
                user_skills.append(row)
            elif is_functional:
                functional_skills.append(row)
            else:
                unknown_skills.append(row)

        if unknown_skills or ambiguous_skills:
            print(
                "[SkillRouter][WARNING] Unclassified/ambiguous skills require manual inspection:",
                unknown_skills + ambiguous_skills,
            )

        return {
            "user_profile_count": len(user_skills),
            "functional_count": len(functional_skills),
            "unknown_count": len(unknown_skills),
            "ambiguous_count": len(ambiguous_skills),
            "user_profile_skills": user_skills,
            "functional_skills": functional_skills,
            "unknown_skills": unknown_skills,
            "ambiguous_skills": ambiguous_skills,
        }

    def _log_skill_pool(self):
        all_skills = list(self.skill_bank.user_skills) + list(
            self.skill_bank.functional_skills
        )
        before_user = sum(self._is_user_profile_skill(s) for s in all_skills)
        before_functional = sum(self._is_functional_skill(s) for s in all_skills)
        print(f"[SkillRouter] retriever checkpoint: {self.retriever_path}")
        print(f"[SkillRouter] User-profile Skills before filter: {before_user}")
        print(f"[SkillRouter] Functional Skills before filter: {before_functional}")

    @staticmethod
    def _extract_json(text: str) -> Dict:
        if not text:
            return {}

        text = str(text).strip()
        text = (
            text.replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )

        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
                return payload if isinstance(payload, dict) else {}
            except Exception:
                pass

        return {}

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            dim = 384
            try:
                dim = int(self.embedding_model.get_sentence_embedding_dimension())
            except Exception:
                pass
            return np.zeros((0, dim), dtype=np.float32)

        return self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    @staticmethod
    def _cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        if len(doc_vecs) == 0:
            return np.array([])
        return np.dot(doc_vecs, query_vec)

    @staticmethod
    def _tokenize_for_rule_match(text: str) -> set:
        """Tokenize text for the deterministic routing fallback only."""
        tokens = re.findall(r"[a-zA-Z0-9']+", str(text or "").lower())
        stop_words = {
            "the", "a", "an", "and", "or", "to", "of", "for", "in", "on",
            "with", "is", "are", "be", "when", "use", "using", "user", "users",
            "candidate", "candidates", "item", "items", "current", "recommendation",
        }
        return {
            token
            for token in tokens
            if len(token) > 2 and token not in stop_words
        }

    @staticmethod
    def _user_skill_to_text(skill: UserProfileSkill) -> str:
        parts = [
            f"Skill Name: {skill.name}",
            f"Description: {skill.description}",
            "Core Preference: " + ", ".join(skill.core_preference),
            "Secondary Preference: " + ", ".join(skill.secondary_preference),
            "Exploration Boundary: " + ", ".join(skill.exploration_boundary),
            "Ranking Guidance: " + ", ".join(skill.ranking_guidance),
            "Constraints: " + ", ".join(getattr(skill, "constraints", [])),
        ]
        return "\n".join(p for p in parts if p and p.strip())

    def summarize_preference(
        self,
        history_items: Sequence[str],
        candidate_items: Sequence[str],
        interaction_memory: str = "",
    ) -> str:
        """Build q_u from the current history and candidate set."""

        history = [str(x).strip() for x in history_items if str(x).strip()]
        candidates = [str(x).strip() for x in candidate_items if str(x).strip()]
        history_text = ", ".join(history[:30])
        candidate_text = ", ".join(candidates[:30])
        memory_text = interaction_memory.strip() if interaction_memory else "None"

        fallback = (
            "Preference evidence should be inferred from the observed user history "
            "and used to discriminate among the current candidate items."
        )

        if self.local_bot is None:
            return fallback

        system_prompt = """
You are an expert recommender-system preference analyzer.

Construct a compact preference query for user-profile skill retrieval.
The query must summarize preference evidence that is useful for
distinguishing among the current candidate items.

Return one short paragraph in plain text only.
"""

        user_prompt = f"""
User history:
{history_text}

Candidate items:
{candidate_text}

Interaction memory:
{memory_text}

Produce one compact preference query.

Requirements:
1. Infer dominant and secondary preference signals from the history.
2. Consider the candidate set so that the query emphasizes evidence
   useful for the current ranking decision.
3. Do not copy item names unnecessarily.
4. Do not invent unsupported preferences.
5. Output plain text only.
"""

        try:
            response = self.local_bot.chat(system_prompt, user_prompt)
            if response:
                return str(response).strip()
        except Exception as exc:
            print(f"[SkillRouter] summarize_preference failed: {exc}")

        return fallback

    def build_state_summary(
        self,
        history_items: Sequence[str],
        interaction_memory: str,
        candidate_items: Sequence[str],
        prior_answer: str = "",
    ) -> Dict[str, object]:
        """Build an open-ended context representation for functional routing.

        The representation intentionally avoids a closed vocabulary such as
        ``short_history`` or ``candidate_homogeneous``. Functional skills can be
        created or updated during Skill Bank evolution, so their applicability
        conditions must be matchable without modifying the controller code.
        """

        history = [str(x).strip() for x in history_items if str(x).strip()]
        candidates = [str(x).strip() for x in candidate_items if str(x).strip()]
        history_text = ", ".join(history)
        candidate_text = ", ".join(candidates)
        memory_text = interaction_memory.strip() if interaction_memory else "None"

        if self.local_bot is not None:
            system_prompt = """
You are a recommender-system context analyzer.

Summarize the current recommendation context into open-ended
decision signals. These signals will be matched against explicit
applicability conditions of functional recommendation skills.

Do not use a fixed state vocabulary.
Do not decide which skill to use.
Return valid JSON only.
"""

            user_prompt = f"""
User history:
{history_text}

Candidate items:
{candidate_text}

Prior recommendation:
{prior_answer or "None"}

Interaction memory:
{memory_text}

Return JSON in this format:
{{
  "context_summary": "one concise description of the current recommendation situation",
  "context_signals": [
    "observable signal 1",
    "observable signal 2"
  ],
  "history_len": {len(history)},
  "candidate_count": {len(candidates)}
}}

Requirements:
1. Describe only evidence observable from the current context.
2. Signals may concern preference stability, sparsity, candidate
   composition, compatibility, uncertainty, repeated patterns,
   diversity needs, or other task-relevant properties.
3. Do not restrict the signals to a predefined list.
4. Do not name or select functional skills.
5. Do not invent item attributes that are not supported by the input.
6. Output JSON only.
"""

            try:
                response = self.local_bot.chat(system_prompt, user_prompt)
                payload = self._extract_json(response)
                if payload:
                    summary = str(payload.get("context_summary", "")).strip()
                    raw_signals = payload.get("context_signals", [])
                    if not isinstance(raw_signals, list):
                        raw_signals = [raw_signals]
                    signals = [
                        str(signal).strip()
                        for signal in raw_signals
                        if str(signal).strip()
                    ]
                    return {
                        "context_summary": summary or "Current recommendation context.",
                        "context_signals": signals,
                        "history_len": len(history),
                        "candidate_count": len(candidates),
                        "prior_answer": prior_answer or "",
                    }
            except Exception as exc:
                print(f"[SkillRouter] context summary generation failed: {exc}")

        # Deterministic fallback. It remains open-ended and includes only
        # observable information from the current instance.
        fallback_signals = [
            f"observed history length is {len(history)}",
            f"candidate set contains {len(candidates)} items",
        ]
        if prior_answer:
            fallback_signals.append("a prior recommendation is available")
        if interaction_memory:
            fallback_signals.append("recent interaction context is available")

        return {
            "context_summary": (
                f"Observed user history: {history_text or 'None'}. "
                f"Observed candidate set: {candidate_text or 'None'}. "
                f"Prior recommendation: {prior_answer or 'None'}."
            ),
            "context_signals": fallback_signals,
            "history_len": len(history),
            "candidate_count": len(candidates),
            "prior_answer": prior_answer or "",
        }

    def retrieve_user_skills(self, preference_query: str) -> List[UserProfileSkill]:
        self._last_user_skill_scores = []

        if not self.skill_bank.user_skills:
            return []

        if self.retriever is not None:
            try:
                ranked = self.retriever.retrieve_topk(
                    query_text=preference_query,
                    skills=self.skill_bank.user_skills,
                    top_k=self.user_skill_top_k,
                    similarity_threshold=None,
                    use_success_bias=False,
                )
                if ranked:
                    selected = ranked[: self.user_skill_top_k]
                    self._last_user_skill_scores = [
                        {
                            "skill_type": "user_profile",
                            "name": skill.name,
                            "score": float(score),
                            "selection_method": "trainable_retriever",
                        }
                        for score, skill in selected
                    ]
                    return [skill for _, skill in selected]
            except Exception as exc:
                print(
                    "[SkillRouter] trainable retriever failed, "
                    f"fallback to semantic retrieval: {exc}"
                )

        # Initial semantic matching / fallback retrieval.
        # No success-rate bias is used so the score remains pure semantic relevance.
        skill_texts = [
            self._user_skill_to_text(skill) for skill in self.skill_bank.user_skills
        ]
        query_vec = self._encode_texts([preference_query])[0]
        skill_vecs = self._encode_texts(skill_texts)
        scores = self._cosine_scores(query_vec, skill_vecs)

        ranked = []
        for score, skill in zip(scores, self.skill_bank.user_skills):
            score = float(score)
            if score >= self.similarity_threshold:
                ranked.append((score, skill))

        ranked.sort(key=lambda row: row[0], reverse=True)
        selected = ranked[: self.user_skill_top_k]
        self._last_user_skill_scores = [
            {
                "skill_type": "user_profile",
                "name": skill.name,
                "score": float(score),
                "selection_method": "semantic_fallback",
            }
            for score, skill in selected
        ]
        return [skill for _, skill in selected]

    @staticmethod
    def _functional_skill_payload(skill: FunctionalSkill) -> Dict[str, object]:
        """Expose only applicability information to the routing controller."""
        return {
            "name": skill.name,
            "purpose": skill.purpose or skill.description,
            "when_to_use": list(getattr(skill, "when_to_use", [])),
            "input_signals": list(getattr(skill, "input_signals", [])),
        }

    def _route_functional_skills_with_llm(
        self,
        state_summary: Dict[str, object],
    ) -> Tuple[List[FunctionalSkill], List[Dict[str, object]], bool]:
        """Evaluate each skill's own applicability rules against the context.

        Returns:
            selected skills,
            audit rows,
            whether the LLM produced a valid routing decision.

        A valid empty selection is preserved. The deterministic fallback is used
        only when the LLM route is unavailable or malformed.
        """

        if self.local_bot is None or not self.skill_bank.functional_skills:
            return [], [], False

        functional_payload = [
            self._functional_skill_payload(skill)
            for skill in self.skill_bank.functional_skills
        ]

        system_prompt = """
You are a functional-skill applicability controller for a
recommender system.

Each functional skill already contains explicit applicability
conditions (when_to_use) and expected input signals.

Your task is RULE/CONDITION MATCHING:
- evaluate the current recommendation context against those
  existing conditions and signals;
- do not invent new routing rules;
- do not rewrite the skills;
- select only skills whose applicability conditions are supported
  by the current context.

Return valid JSON only.
"""

        user_prompt = f"""
Current recommendation context:
{json.dumps(state_summary, ensure_ascii=False, indent=2)}

Available functional skills:
{json.dumps(functional_payload, ensure_ascii=False, indent=2)}

Return JSON in this format:
{{
  "selected": [
    {{
      "name": "exact existing skill name",
      "score": 0.0,
      "matched_conditions": ["condition from when_to_use"],
      "matched_input_signals": ["supported input signal"]
    }}
  ]
}}

Requirements:
1. Match against each skill's EXISTING when_to_use and input_signals.
2. A newly evolved skill is treated exactly like an initialized skill:
   its own stored conditions become its routing rules.
3. Do not require conditions to come from any fixed global state vocabulary.
4. Do not select a skill merely because its topic sounds similar.
5. Select a skill only when the current context provides evidence
   supporting its applicability condition(s).
6. score must be between 0 and 1 and represent applicability confidence.
7. Use exact existing skill names.
8. Return at most {self.functional_skill_top_k} selected skills.
9. If no skill is applicable, return {{"selected": []}}.
10. Output JSON only.
"""

        try:
            response = self.local_bot.chat(system_prompt, user_prompt)
            payload = self._extract_json(response)
            if not payload or "selected" not in payload:
                return [], [], False

            rows = payload.get("selected", [])
            if not isinstance(rows, list):
                return [], [], False

            skill_by_name = {
                skill.name: skill for skill in self.skill_bank.functional_skills
            }
            validated = []

            for row in rows:
                if not isinstance(row, dict):
                    continue

                name = str(row.get("name", "")).strip()
                skill = skill_by_name.get(name)
                if skill is None:
                    continue

                try:
                    score = float(row.get("score", 0.0))
                except Exception:
                    score = 0.0
                score = max(0.0, min(1.0, score))
                if score <= 0.0:
                    continue

                matched_conditions = row.get("matched_conditions", [])
                matched_input_signals = row.get("matched_input_signals", [])
                if not isinstance(matched_conditions, list):
                    matched_conditions = [matched_conditions]
                if not isinstance(matched_input_signals, list):
                    matched_input_signals = [matched_input_signals]

                validated.append(
                    (
                        score,
                        skill,
                        {
                            "skill_type": "functional",
                            "name": skill.name,
                            "score": score,
                            "selection_method": "applicability_condition_matching",
                            "matched_conditions": [
                                str(x) for x in matched_conditions if str(x).strip()
                            ],
                            "matched_input_signals": [
                                str(x)
                                for x in matched_input_signals
                                if str(x).strip()
                            ],
                        },
                    )
                )

            validated.sort(key=lambda row: row[0], reverse=True)
            validated = validated[: self.functional_skill_top_k]

            return (
                [skill for _, skill, _ in validated],
                [audit for _, _, audit in validated],
                True,
            )

        except Exception as exc:
            print(
                "[SkillRouter] functional applicability matching failed: "
                f"{exc}"
            )
            return [], [], False

    def _route_functional_skills_fallback(
        self,
        state_summary: Dict[str, object],
    ) -> Tuple[List[FunctionalSkill], List[Dict[str, object]]]:
        """Deterministic fallback for open-ended applicability matching.

        This fallback compares observable context tokens against each skill's own
        ``when_to_use`` and ``input_signals``. It does not use any fixed global
        controller-state vocabulary.
        """

        context_parts = [str(state_summary.get("context_summary", ""))]
        signals = state_summary.get("context_signals", [])
        if isinstance(signals, list):
            context_parts.extend(str(x) for x in signals)

        context_tokens = self._tokenize_for_rule_match(" ".join(context_parts))
        ranked = []

        for skill in self.skill_bank.functional_skills:
            conditions = list(getattr(skill, "when_to_use", []))
            input_signals = list(getattr(skill, "input_signals", []))

            condition_tokens = self._tokenize_for_rule_match(
                " ".join(str(x) for x in conditions)
            )
            input_tokens = self._tokenize_for_rule_match(
                " ".join(str(x) for x in input_signals)
            )

            condition_overlap = (
                len(context_tokens & condition_tokens) / max(1, len(condition_tokens))
            )
            signal_overlap = (
                len(context_tokens & input_tokens) / max(1, len(input_tokens))
            )

            default_bonus = 0.0
            if any(
                str(condition).strip().lower() in {"default", "generally applicable"}
                for condition in conditions
            ):
                default_bonus = 0.15

            score = (
                0.6 * condition_overlap
                + 0.4 * signal_overlap
                + default_bonus
            )
            if score <= 0.0:
                continue

            ranked.append(
                (
                    float(score),
                    skill,
                    {
                        "skill_type": "functional",
                        "name": skill.name,
                        "score": float(score),
                        "selection_method": "rule_token_fallback",
                        "matched_conditions": [],
                        "matched_input_signals": [],
                    },
                )
            )

        ranked.sort(key=lambda row: row[0], reverse=True)
        ranked = ranked[: self.functional_skill_top_k]

        return (
            [skill for _, skill, _ in ranked],
            [audit for _, _, audit in ranked],
        )

    def route_functional_skills(
        self,
        state_summary: Dict[str, object],
    ) -> List[FunctionalSkill]:
        """Select functional skills by their stored applicability rules."""

        self._last_functional_skill_scores = []

        if not self.skill_bank.functional_skills:
            return []

        selected, audit, valid_decision = self._route_functional_skills_with_llm(
            state_summary
        )

        # A valid empty LLM decision means no functional skill is applicable.
        # Fallback is used only for unavailable/malformed LLM routing.
        if not valid_decision:
            selected, audit = self._route_functional_skills_fallback(state_summary)

        self._last_functional_skill_scores = audit
        return selected

    def select_skills(
        self,
        history_items: Sequence[str],
        interaction_memory: str,
        candidate_items: Sequence[str],
        prior_answer: str = "",
    ) -> RoutedSkills:
        preference_query = self.summarize_preference(
            history_items=history_items,
            candidate_items=candidate_items,
            interaction_memory=interaction_memory,
        )

        state_summary = self.build_state_summary(
            history_items=history_items,
            interaction_memory=interaction_memory,
            candidate_items=candidate_items,
            prior_answer=prior_answer,
        )

        user_skills = self.retrieve_user_skills(preference_query)
        functional_skills = self.route_functional_skills(state_summary)

        skill_scores = (
            list(self._last_user_skill_scores)
            + list(self._last_functional_skill_scores)
        )

        return RoutedSkills(
            preference_query=preference_query,
            state_summary=state_summary,
            user_skills=user_skills,
            functional_skills=functional_skills,
            skill_scores=skill_scores,
        )
