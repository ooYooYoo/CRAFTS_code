import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .failure_buffer import FailureBuffer, FailureCase
from .skill_bank import (
    FunctionalSkill,
    PreferenceTemplateBuilder,
    SkillBank,
    UserProfileSkill,
)



class SkillDesigner:
    """Construct and evolve the domain-specific CRAFTS Skill Bank."""

    def __init__(
        self,
        skill_bank: SkillBank,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_bot=None,
    ):
        self.skill_bank = skill_bank
        self.evolution_round = 0
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        if hasattr(self.embedding_model, "eval"):
            self.embedding_model.eval()
        self.local_bot = local_bot

    def set_local_bot(self, local_bot):
        self.local_bot = local_bot

    def _call_llm(self, system_prompt: str, user_prompt: str, local_bot=None) -> str:
        bot = local_bot if local_bot is not None else self.local_bot
        if bot is None:
            return ""
        try:
            response = bot.chat(system_prompt, user_prompt)
            return "" if response is None else str(response).strip()
        except Exception as exc:
            print(f"[SkillDesigner] LLM call failed: {exc}")
            return ""

    @staticmethod
    def _ensure_list(value, max_items: Optional[int] = None) -> List[str]:
        if value is None:
            items: List[str] = []
        elif isinstance(value, list):
            items = [str(x).strip() for x in value if str(x).strip()]
        else:
            text = str(value).strip()
            if not text:
                items = []
            elif "|" in text:
                items = [seg.strip() for seg in text.split("|") if seg.strip()]
            elif ";" in text:
                items = [seg.strip() for seg in text.split(";") if seg.strip()]
            else:
                items = [text]
        if max_items is not None:
            items = items[:max_items]
        return items

    def initialize_from_train_histories(
        self,
        user_histories: Iterable[Sequence[str]],
        budget: Optional[int] = None,
        cluster_size: int = 8,
        similarity_threshold: float = 0.72,
        llm_args=None,
        local_bot=None,
    ) -> SkillBank:
        """Initialize user-profile and functional skills.

        User-profile skills are first generated from individual training histories
        and then aggregated across semantically similar profiles. Functional skills
        are generated independently from the recommendation task and general
        recommendation principles.
        """

        use_llm_init = bool(getattr(llm_args, "skill_init_use_llm", True))
        teacher_bot = local_bot if use_llm_init else None
        print(
            f"[SkillDesigner] init use_llm={use_llm_init}, "
            f"teacher_bot is None? {teacher_bot is None}"
        )

        # User-profile skill initialization.
        if not self.skill_bank.has_user_skills():
            raw_skills: List[UserProfileSkill] = []
            max_profiles = None
            if budget is not None:
                try:
                    parsed_budget = int(budget)
                    if parsed_budget > 0:
                        max_profiles = parsed_budget
                except Exception:
                    max_profiles = None

            for index, history in enumerate(user_histories, start=1):
                clean_history = [str(item).strip() for item in history if str(item).strip()]
                if len(clean_history) < 2:
                    continue

                skill = self._build_user_skill_with_llm(
                    skill_index=index,
                    history_items=clean_history,
                    llm_args=llm_args,
                    local_bot=teacher_bot,
                )
                if skill is not None:
                    raw_skills.append(skill)

                if max_profiles is not None and len(raw_skills) >= max_profiles:
                    break

            merged_skills = self._merge_similar_user_skills(
                raw_skills,
                cluster_size=cluster_size,
                similarity_threshold=similarity_threshold,
            )
            self.skill_bank.set_user_skills(merged_skills, reason="train_init")

        # Functional skill initialization is independent of user-profile generation.
        if not self.skill_bank.has_functional_skills():
            functional_skills = self._build_functional_skills_with_llm(
                local_bot=teacher_bot
            )
            if functional_skills:
                self.skill_bank.set_functional_skills(
                    functional_skills,
                    reason="train_init_llm",
                )
            else:
                print(
                    "[SkillDesigner] Functional-skill LLM initialization failed; "
                    "using compact fallback skills."
                )
                self.skill_bank.build_default_functional_skills()

        return self.skill_bank

    @staticmethod
    def _skill_to_text(skill: UserProfileSkill) -> str:
        parts = [
            skill.description,
            "Core Preference: " + ", ".join(skill.core_preference),
            "Secondary Preference: " + ", ".join(skill.secondary_preference),
            "Exploration Boundary: " + ", ".join(skill.exploration_boundary),
            "Ranking Guidance: " + ", ".join(skill.ranking_guidance),
            "Constraints: " + ", ".join(getattr(skill, "constraints", [])),
        ]
        return "\n".join(p for p in parts if p and p.strip())

    def _encode_skill_texts(self, texts: List[str]):
        return self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    @staticmethod
    def _cosine_sim(a, b) -> float:
        return float(np.dot(a, b))

    def _build_user_skill_with_llm(
        self,
        skill_index: int,
        history_items: Sequence[str],
        llm_args=None,
        local_bot=None,
    ) -> Optional[UserProfileSkill]:
        history_items = [
            str(item).strip() for item in history_items if str(item).strip()
        ]
        if len(history_items) < 2:
            return None

        history_text = ", ".join(history_items[:50])

        system_prompt = """
You are an expert recommender-system profile designer for Amazon Musical Instruments recommendation.

Infer a reusable and discriminative user-profile skill from a user's interaction history.
Abstract latent preference patterns rather than copying product titles, ASINs,
brands, or exact item names. The resulting profile should provide transferable
preference evidence for later candidate ranking.

Return valid JSON only.
"""

        user_prompt = f"""
User interaction history:
{history_text}

Return JSON in the following format:
{{
  "name": "short_skill_name",
  "description": "2-3 sentence reusable profile summary",
  "core_preference": ["...", "...", "..."],
  "secondary_preference": ["...", "..."],
  "exploration_boundary": ["...", "..."],
  "ranking_guidance": ["...", "...", "..."],
  "constraints": ["...", "..."],
  "applicable_patterns": ["...", "...", "..."]
}}

Requirements:
1. Infer high-level preference patterns instead of copying exact interacted items.
2. Distinguish stable core preferences from weaker secondary tendencies.
3. Capture product function, use scenario, compatibility, quality/price tendency,
   or other preference dimensions only when supported by the history.
4. Specify an acceptable exploration boundary and compatibility constraints.
5. Provide explicit candidate-ranking guidance.
6. The skill must be reusable across preference-similar users.
7. Output valid JSON only.
"""

        response = self._call_llm(system_prompt, user_prompt, local_bot=local_bot)
        if response:
            payload = self._extract_json(response)
            if payload:
                description = (
                    str(payload.get("description", "")).strip()
                    or "LLM-generated user-profile skill."
                )
                name = (
                    str(payload.get("name", f"user_profile_{skill_index}"))
                    .strip()[:80]
                    or f"user_profile_{skill_index}"
                )
                return UserProfileSkill(
                    skill_id=f"user_profile_{skill_index}",
                    name=name,
                    skill_type="user_profile",
                    dataset_scope=self.skill_bank.dataset_scope,
                    description=description,
                    core_preference=self._ensure_list(
                        payload.get("core_preference"), 4
                    ),
                    secondary_preference=self._ensure_list(
                        payload.get("secondary_preference"), 4
                    ),
                    exploration_boundary=self._ensure_list(
                        payload.get("exploration_boundary"), 4
                    ),
                    ranking_guidance=self._ensure_list(
                        payload.get("ranking_guidance"), 4
                    ),
                    constraints=self._ensure_list(payload.get("constraints"), 4),
                    applicable_patterns=self._ensure_list(
                        payload.get("applicable_patterns"), 6
                    ),
                    prototype_histories=[" | ".join(history_items[:10])],
                    source="train_init_llm",
                )

            print(
                "[SkillDesigner] Invalid LLM user-profile JSON; "
                "using fallback profile construction."
            )

        fallback = PreferenceTemplateBuilder.build_skill(
            skill_index=skill_index,
            dataset_scope=self.skill_bank.dataset_scope,
            history_items=history_items,
            source="train_init_template_fallback",
        )
        fallback.constraints = fallback.constraints or [
            "do not infer highly specific preferences without sufficient evidence"
        ]
        return fallback

    def _build_functional_skills_with_llm(self, local_bot=None) -> List[FunctionalSkill]:
        """Generate functional skills with open-ended applicability conditions."""

        if local_bot is None:
            return []

        system_prompt = """
You are an expert recommender-system skill designer.

Generate a compact set of reusable functional skills for candidate comparison and
ranking in Amazon Musical Instruments recommendation. Functional skills must encode
general decision strategies rather than user-specific preferences or exact products.

Each skill must define its own explicit applicability conditions and observable
input signals. Do not force applicability conditions into a fixed global state
vocabulary: later failure-driven evolution may introduce new reusable conditions.

Return valid JSON only.
"""

        user_prompt = """
Create a compact functional-skill set for recommendation.

Each functional skill must contain:
- name
- description
- purpose
- when_to_use
- input_signals
- how_to_apply
- constraints

`when_to_use` must contain explicit natural-language applicability conditions.
`input_signals` must describe observable evidence needed to judge whether those
conditions hold in the current recommendation context.

Return JSON only in this format:
{
  "functional_skills": [
    {
      "name": "preference_alignment",
      "description": "...",
      "purpose": "...",
      "when_to_use": [
        "the user history provides stable recurring preference evidence"
      ],
      "input_signals": [
        "recurring product functions or categories",
        "consistent preference cues in the interaction history"
      ],
      "how_to_apply": ["..."],
      "constraints": ["..."]
    }
  ]
}

Requirements:
1. Produce reusable candidate-comparison strategies, not user profiles.
2. Keep the skill set compact; 3-6 complementary skills are sufficient.
3. Cover common recommendation situations without creating item-specific rules.
4. Make each skill's applicability conditions explicit and independently testable
   from the current recommendation context.
5. Do not restrict `when_to_use` to a predefined list of state names.
6. Make `input_signals` observable from history, candidates, prior recommendation,
   or available interaction context; do not require hidden labels.
7. Make application rules actionable at candidate-ranking time.
8. Output valid JSON only.
"""

        response = self._call_llm(system_prompt, user_prompt, local_bot=local_bot)
        payload = self._extract_json(response) if response else None
        if not payload:
            return []

        items = payload.get("functional_skills", [])
        if not isinstance(items, list):
            return []

        skills: List[FunctionalSkill] = []
        seen_names = set()

        for index, item in enumerate(items[:6], start=1):
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()[:80]
            if not name or name in seen_names:
                continue

            conditions = self._ensure_list(item.get("when_to_use"), 6)
            input_signals = self._ensure_list(item.get("input_signals"), 6)
            if not conditions or not input_signals:
                print(
                    "[SkillDesigner] Skip functional skill without explicit "
                    f"applicability conditions/input signals: {name}"
                )
                continue

            seen_names.add(name)
            description = str(item.get("description", "")).strip()
            purpose = str(item.get("purpose", "")).strip() or description

            skills.append(
                FunctionalSkill(
                    skill_id=f"functional_init_{index}",
                    name=name,
                    skill_type="functional",
                    dataset_scope=self.skill_bank.dataset_scope,
                    description=description or "LLM-generated functional skill.",
                    purpose=purpose or "Reusable candidate-ranking strategy.",
                    when_to_use=conditions,
                    input_signals=input_signals,
                    how_to_apply=self._ensure_list(item.get("how_to_apply"), 8),
                    constraints=self._ensure_list(item.get("constraints"), 6),
                    source="train_init_llm",
                )
            )

        return skills

    def _merge_similar_user_skills(
        self,
        skills: List[UserProfileSkill],
        cluster_size: int,
        similarity_threshold: float,
    ) -> List[UserProfileSkill]:
        if not skills:
            return []

        skill_texts = [self._skill_to_text(skill) for skill in skills]
        skill_embs = self._encode_skill_texts(skill_texts)

        groups: List[List[int]] = []
        group_centroids = []
        threshold = float(similarity_threshold)

        for idx, emb in enumerate(skill_embs):
            placed = False
            for group_idx, centroid in enumerate(group_centroids):
                sim = self._cosine_sim(emb, centroid)
                if (
                    sim >= threshold
                    and len(groups[group_idx]) < max(1, int(cluster_size))
                ):
                    groups[group_idx].append(idx)
                    member_embs = skill_embs[groups[group_idx]]
                    new_centroid = member_embs.mean(axis=0)
                    new_centroid = new_centroid / (
                        np.linalg.norm(new_centroid) + 1e-12
                    )
                    group_centroids[group_idx] = new_centroid
                    placed = True
                    break

            if not placed:
                groups.append([idx])
                centroid = emb / (np.linalg.norm(emb) + 1e-12)
                group_centroids.append(centroid)

        merged: List[UserProfileSkill] = []
        for group_index, group_indices in enumerate(groups, start=1):
            group_skills = [skills[i] for i in group_indices]
            merged.append(self._aggregate_user_skill_group(group_index, group_skills))
        return merged

    @staticmethod
    def _normalized_preference_key(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    def _aggregate_user_skill_group(
        self,
        group_index: int,
        group: List[UserProfileSkill],
    ) -> UserProfileSkill:
        counts = Counter()
        representative_text = {}
        prototypes: List[str] = []

        for skill in group:
            for signal in list(skill.core_preference) + list(skill.secondary_preference):
                key = self._normalized_preference_key(signal)
                if not key:
                    continue
                counts[key] += 1
                representative_text.setdefault(key, signal)
            prototypes.extend(skill.prototype_histories[:1])

        ranked_keys = [key for key, _ in counts.most_common(8)]
        ranked = [representative_text[key] for key in ranked_keys]
        core = ranked[:4] or ["general stable preference"]
        secondary = ranked[4:8]

        constraints = []
        for skill in group:
            constraints.extend(skill.constraints)
        constraints = list(dict.fromkeys(x for x in constraints if x))[:4]

        return UserProfileSkill(
            skill_id=f"user_cluster_{group_index}",
            name=f"user_cluster_{group_index}_{'_'.join(core[:2])}"[:80],
            skill_type="user_profile",
            dataset_scope=self.skill_bank.dataset_scope,
            description=(
                "Group-level user-profile skill aggregated from semantically "
                "similar training-user preference profiles."
            ),
            core_preference=core,
            secondary_preference=secondary,
            exploration_boundary=[
                f"stay close to the dominant preference signals: {', '.join(core[:2])}"
            ],
            ranking_guidance=[
                "first satisfy multiple core preference signals",
                "use secondary preferences for tie-breaking when core alignment is similar",
            ],
            constraints=constraints,
            applicable_patterns=ranked,
            prototype_histories=prototypes[:5],
            source="train_init_cluster",
        )

    def evolve_from_failures(
        self,
        failure_buffer: FailureBuffer,
        limit: int = 20,
    ) -> bool:
        hard_cases = failure_buffer.top_cases(limit=limit)
        if not hard_cases:
            return False

        self.evolution_round += 1
        llm_payload = self._propose_skill_updates_with_llm(hard_cases)
        changes_applied = 0

        if llm_payload:
            changes_applied += self._apply_llm_skill_updates(llm_payload)
        else:
            print(
                "[SkillDesigner] No valid LLM evolution payload; "
                "skip failure-driven skill update."
            )

        if changes_applied > 0:
            self.skill_bank.save()

        failure_buffer.mark_evolved()
        return changes_applied > 0

    def _propose_skill_updates_with_llm(
        self,
        cases: List[FailureCase],
    ) -> Dict[str, Any]:
        if self.local_bot is None:
            return {}

        serialized_cases = []
        for idx, case in enumerate(cases, start=1):
            serialized_cases.append(
                {
                    "case_id": idx,
                    "user_history": case.user_history,
                    "candidates": case.candidates,
                    "final_recommendation": case.final_recommendation,
                    "ground_truth": case.ground_truth,
                    "user_feedback": case.user_feedback,
                    "selected_skills": case.selected_skills,
                    "preference_query": case.preference_query,
                    "state_summary": case.state_summary,
                    "interaction_history": case.interaction_history[-2:],
                }
            )

        system_prompt = """
You are an expert skill-bank designer for a recommender system.

Analyze a batch of failed recommendation cases together with the current Skill Bank.
Identify recurring failure patterns and determine whether they provide sufficient
evidence of missing or insufficient transferable skill knowledge.

You may update an existing skill or create a new reusable skill.
For functional skills, applicability conditions are open-ended natural-language
rules stored inside each skill. A newly created condition becomes a routing rule
for future recommendation instances.

Return valid JSON only.
"""

        user_prompt = f"""
Current dataset scope: {self.skill_bank.dataset_scope or 'unknown'}

Existing functional skills:
{json.dumps([{'name': s.name, 'description': s.description, 'purpose': s.purpose, 'when_to_use': s.when_to_use, 'input_signals': s.input_signals, 'how_to_apply': s.how_to_apply, 'constraints': s.constraints} for s in self.skill_bank.functional_skills], ensure_ascii=False, indent=2)}

Existing user-profile skills:
{json.dumps([{'name': s.name, 'description': s.description, 'core_preference': s.core_preference, 'secondary_preference': s.secondary_preference, 'exploration_boundary': s.exploration_boundary, 'ranking_guidance': s.ranking_guidance, 'constraints': s.constraints} for s in self.skill_bank.user_skills], ensure_ascii=False, indent=2)}

Failed recommendation cases:
{json.dumps(serialized_cases, ensure_ascii=False, indent=2)}

Return JSON only in this format:
{{
  "analysis": {{
    "failure_pattern_summary": ["..."],
    "root_causes": ["..."]
  }},
  "functional_skill_updates": [
    {{
      "mode": "update",
      "name": "existing_skill_name",
      "description": "...",
      "purpose": "...",
      "when_to_use": [
        "explicit reusable applicability condition"
      ],
      "input_signals": [
        "observable context signal needed to test applicability"
      ],
      "how_to_apply": ["..."],
      "constraints": ["..."]
    }}
  ],
  "user_skill_updates": [
    {{
      "mode": "create",
      "name": "new_skill_name",
      "description": "...",
      "core_preference": ["..."],
      "secondary_preference": ["..."],
      "exploration_boundary": ["..."],
      "ranking_guidance": ["..."],
      "constraints": ["..."],
      "applicable_patterns": ["..."]
    }}
  ]
}}

Requirements:
1. Propose a change only when it is supported by recurring failure evidence.
2. Prefer refining an existing reusable skill over creating a one-off skill.
3. Never create skills or applicability conditions tied to exact item titles.
4. `mode` must be exactly `update` or `create`.
5. For `update`, the skill name must exactly match an existing skill name.
6. For `create`, the skill name must be new.
7. Functional `when_to_use` entries must be explicit, reusable natural-language
   applicability conditions supported by the recurring failures. They may introduce
   new conditions; do not force them into a fixed global state vocabulary.
8. Functional `input_signals` must describe observable evidence from the
   recommendation context that can be used to judge those applicability conditions.
9. When updating a functional skill, preserve useful existing conditions unless
   the failure evidence justifies refining or replacing them.
10. At most one functional-skill update and one user-profile-skill update may be proposed.
11. If evidence is insufficient, return an empty list for that skill type.
12. Output valid JSON only.
"""

        response = self._call_llm(system_prompt, user_prompt)
        if not response:
            return {}

        payload = self._extract_json(response)
        if not payload:
            print(
                f"[SkillDesigner] Failed to parse LLM evolution payload: {response}"
            )
        return payload or {}

    def _apply_llm_skill_updates(self, payload: Dict[str, Any]) -> int:
        changes = 0

        functional_updates = payload.get("functional_skill_updates", [])
        if isinstance(functional_updates, list):
            functional_updates = functional_updates[:1]
        else:
            functional_updates = []

        for idx, item in enumerate(functional_updates, start=1):
            if not isinstance(item, dict):
                continue

            mode = str(item.get("mode", "")).strip().lower()
            name = str(item.get("name", "")).strip()[:80]
            if mode not in {"update", "create"} or not name:
                continue

            existing = self._get_functional_skill(name)
            if mode == "update" and existing is None:
                print(
                    f"[SkillDesigner] Skip functional update for unknown skill: {name}"
                )
                continue
            if mode == "create" and existing is not None:
                print(
                    f"[SkillDesigner] Skip functional create for existing skill: {name}"
                )
                continue

            proposed_conditions = self._ensure_list(item.get("when_to_use"), 6)
            proposed_signals = self._ensure_list(item.get("input_signals"), 6)

            if existing is not None:
                conditions = proposed_conditions or list(existing.when_to_use)
                input_signals = proposed_signals or list(existing.input_signals)
            else:
                conditions = proposed_conditions
                input_signals = proposed_signals

            # A created functional skill must be independently routable.
            if not conditions or not input_signals:
                print(
                    "[SkillDesigner] Skip functional "
                    f"{mode} without explicit applicability conditions/input signals: {name}"
                )
                continue

            description = str(item.get("description", "")).strip()
            purpose = str(item.get("purpose", "")).strip()
            if existing is not None:
                description = description or existing.description
                purpose = purpose or existing.purpose

            skill = FunctionalSkill(
                skill_id=(
                    existing.skill_id
                    if existing is not None
                    else f"functional_llm_{self.evolution_round}_{idx}"
                ),
                name=name,
                skill_type="functional",
                dataset_scope=self.skill_bank.dataset_scope,
                description=description or "LLM-evolved functional skill.",
                purpose=purpose or description or "Reusable candidate-ranking strategy.",
                when_to_use=conditions,
                input_signals=input_signals,
                how_to_apply=(
                    self._ensure_list(item.get("how_to_apply"), 8)
                    or (list(existing.how_to_apply) if existing is not None else [])
                ),
                constraints=(
                    self._ensure_list(item.get("constraints"), 6)
                    or (list(existing.constraints) if existing is not None else [])
                ),
                source="failure_evolution_llm",
            )

            self.skill_bank.upsert_functional_skill(
                skill,
                reason=f"failure_round_{self.evolution_round}_llm",
            )
            changes += 1

        user_updates = payload.get("user_skill_updates", [])
        if isinstance(user_updates, list):
            user_updates = user_updates[:1]
        else:
            user_updates = []

        for idx, item in enumerate(user_updates, start=1):
            if not isinstance(item, dict):
                continue
            mode = str(item.get("mode", "")).strip().lower()
            name = str(item.get("name", "")).strip()[:80]
            if mode not in {"update", "create"} or not name:
                continue

            existing = next(
                (skill for skill in self.skill_bank.user_skills if skill.name == name),
                None,
            )
            if mode == "update" and existing is None:
                print(f"[SkillDesigner] Skip user-skill update for unknown skill: {name}")
                continue
            if mode == "create" and existing is not None:
                print(f"[SkillDesigner] Skip user-skill create for existing skill: {name}")
                continue

            description = str(item.get("description", "")).strip()
            if existing is not None:
                description = description or existing.description

            skill = UserProfileSkill(
                skill_id=(
                    existing.skill_id
                    if existing is not None
                    else f"user_llm_{self.evolution_round}_{idx}"
                ),
                name=name,
                skill_type="user_profile",
                dataset_scope=self.skill_bank.dataset_scope,
                description=description or "LLM-evolved user-profile skill.",
                core_preference=(
                    self._ensure_list(item.get("core_preference"), 4)
                    or (list(existing.core_preference) if existing is not None else [])
                ),
                secondary_preference=(
                    self._ensure_list(item.get("secondary_preference"), 4)
                    or (list(existing.secondary_preference) if existing is not None else [])
                ),
                exploration_boundary=(
                    self._ensure_list(item.get("exploration_boundary"), 4)
                    or (list(existing.exploration_boundary) if existing is not None else [])
                ),
                ranking_guidance=(
                    self._ensure_list(item.get("ranking_guidance"), 6)
                    or (list(existing.ranking_guidance) if existing is not None else [])
                ),
                constraints=(
                    self._ensure_list(item.get("constraints"), 6)
                    or (list(existing.constraints) if existing is not None else [])
                ),
                applicable_patterns=(
                    self._ensure_list(item.get("applicable_patterns"), 8)
                    or (list(existing.applicable_patterns) if existing is not None else [])
                ),
                prototype_histories=(
                    existing.prototype_histories[:5] if existing is not None else []
                ),
                source="failure_evolution_llm",
            )
            self.skill_bank.upsert_user_skill(
                skill,
                reason=f"failure_round_{self.evolution_round}_llm",
            )
            changes += 1

        return changes

    def _analyze_failures(self, cases: List[FailureCase]) -> Dict[str, int]:
        """Lightweight audit over open-ended context signals.

        The main evolution decision is LLM-based. This helper intentionally avoids
        assuming a fixed controller-state vocabulary.
        """

        summary = defaultdict(int)

        for case in cases:
            state = case.state_summary or {}
            signals = state.get("context_signals", [])
            if not isinstance(signals, list):
                signals = [signals]

            for signal in signals:
                normalized = " ".join(str(signal or "").strip().lower().split())
                if normalized:
                    summary[f"context_signal::{normalized}"] += 1

            if case.selected_skills:
                summary["skill_guided_failures"] += 1

        return dict(summary)

    def _get_functional_skill(self, skill_name: str) -> Optional[FunctionalSkill]:
        for skill in self.skill_bank.functional_skills:
            if skill.name == skill_name:
                return skill
        return None

    @staticmethod
    def _extract_json(text):
        if not text:
            return None

        text = str(text).strip()
        text = text.replace("```json", "").replace("```JSON", "").replace("```", "")
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            print("[SkillDesigner] No valid JSON boundaries.")
            return None

        json_text = text[start : end + 1]
        try:
            return json.loads(json_text)
        except Exception as exc:
            print("[SkillDesigner] JSON parse failed:", repr(exc))
            print("[SkillDesigner] JSON text:")
            print(json_text[:3000])
            return None
