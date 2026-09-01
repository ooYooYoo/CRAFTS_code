import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Sequence


_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return [tok for tok in _TOKEN_RE.findall(text) if len(tok) > 1]


def jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    lset = set(left)
    rset = set(right)
    if not lset and not rset:
        return 0.0
    return len(lset & rset) / max(1, len(lset | rset))


@dataclass
class SkillRecord:
    skill_id: str
    name: str
    skill_type: str
    dataset_scope: str
    description: str
    version: int = 1
    source: str = "generated"
    usage_count: int = 0
    success_count: int = 0
    last_update_reason: str = ""

    def register_outcome(self, success: bool):
        self.usage_count += 1
        if success:
            self.success_count += 1

    @property
    def success_rate(self) -> float:
        if self.usage_count <= 0:
            return 0.0
        return self.success_count / self.usage_count


@dataclass
class UserProfileSkill(SkillRecord):
    core_preference: List[str] = field(default_factory=list)
    secondary_preference: List[str] = field(default_factory=list)
    exploration_boundary: List[str] = field(default_factory=list)
    ranking_guidance: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    applicable_patterns: List[str] = field(default_factory=list)
    prototype_histories: List[str] = field(default_factory=list)

    def profile_text(self) -> str:
        chunks = [self.name, self.description]
        chunks.extend(self.core_preference)
        chunks.extend(self.secondary_preference)
        chunks.extend(self.exploration_boundary)
        chunks.extend(self.ranking_guidance)
        chunks.extend(self.constraints)
        chunks.extend(self.applicable_patterns)
        return " | ".join(chunk for chunk in chunks if chunk)

    def retrieval_tokens(self) -> List[str]:
        return tokenize(self.profile_text())

    def to_prompt(self) -> str:
        """Serialize the complete structured user-profile skill for the Rec Agent."""
        return (
            f"[User-Profile Skill] {self.name}\n"
            f"Description: {self.description or 'N/A'}\n"
            f"Core Preferences: {', '.join(self.core_preference) or 'N/A'}\n"
            f"Secondary Preferences: {', '.join(self.secondary_preference) or 'N/A'}\n"
            f"Exploration Boundary: {', '.join(self.exploration_boundary) or 'N/A'}\n"
            f"Ranking Guidance: {', '.join(self.ranking_guidance) or 'N/A'}\n"
            f"Constraints: {', '.join(self.constraints) or 'N/A'}"
        )


@dataclass
class FunctionalSkill(SkillRecord):
    purpose: str = ""
    when_to_use: List[str] = field(default_factory=list)
    input_signals: List[str] = field(default_factory=list)
    how_to_apply: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    def applicability_text(self) -> str:
        """Serialize only the fields used to decide whether this skill applies."""
        chunks = [
            f"Skill Name: {self.name}",
            f"Purpose: {self.purpose or self.description}",
            "When to Use: " + "; ".join(self.when_to_use),
            "Input Signals: " + "; ".join(self.input_signals),
        ]
        return "\n".join(chunk for chunk in chunks if chunk and chunk.strip())

    def routing_tokens(self) -> List[str]:
        chunks = [self.name, self.description, self.purpose]
        chunks.extend(self.when_to_use)
        chunks.extend(self.input_signals)
        chunks.extend(self.how_to_apply)
        chunks.extend(self.constraints)
        return tokenize(" | ".join(chunk for chunk in chunks if chunk))

    def to_prompt(self) -> str:
        """Serialize the complete structured functional skill for the Rec Agent."""
        return (
            f"[Functional Skill] {self.name}\n"
            f"Description: {self.description or 'N/A'}\n"
            f"Purpose: {self.purpose or self.description or 'N/A'}\n"
            f"When to Use: {', '.join(self.when_to_use) or 'N/A'}\n"
            f"Input Signals: {', '.join(self.input_signals) or 'N/A'}\n"
            f"How to Apply: {', '.join(self.how_to_apply) or 'N/A'}\n"
            f"Constraints: {', '.join(self.constraints) or 'N/A'}"
        )


class SkillBank:
    """Domain-specific CRAFTS Skill Bank.

    Storage layout under ``skill_dir``:
      - skill_bank.json
      - user_skills/*.md
      - functional_skills/*.md
    """

    def __init__(self, skill_dir: str, dataset_scope: str = ""):
        self.skill_dir = skill_dir
        self.dataset_scope = dataset_scope
        self.user_skills: List[UserProfileSkill] = []
        self.functional_skills: List[FunctionalSkill] = []
        self.manifest_path = os.path.join(self.skill_dir, "skill_bank.json")
        os.makedirs(self.skill_dir, exist_ok=True)
        self._load_if_exists()

    def _load_if_exists(self):
        if not os.path.exists(self.manifest_path):
            return
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.dataset_scope = payload.get("dataset_scope", self.dataset_scope)
        self.user_skills = [
            UserProfileSkill(**row) for row in payload.get("user_skills", [])
        ]
        self.functional_skills = [
            FunctionalSkill(**row) for row in payload.get("functional_skills", [])
        ]

    def save(self):
        os.makedirs(self.skill_dir, exist_ok=True)
        payload = {
            "dataset_scope": self.dataset_scope,
            "user_skills": [asdict(skill) for skill in self.user_skills],
            "functional_skills": [asdict(skill) for skill in self.functional_skills],
        }
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self._export_markdown_snapshots()

    def _export_markdown_snapshots(self):
        user_dir = os.path.join(self.skill_dir, "user_skills")
        functional_dir = os.path.join(self.skill_dir, "functional_skills")
        os.makedirs(user_dir, exist_ok=True)
        os.makedirs(functional_dir, exist_ok=True)

        for filename in os.listdir(user_dir):
            if filename.endswith(".md"):
                os.remove(os.path.join(user_dir, filename))
        for filename in os.listdir(functional_dir):
            if filename.endswith(".md"):
                os.remove(os.path.join(functional_dir, filename))

        for skill in self.user_skills:
            filename = self._safe_snapshot_filename(skill.name)
            with open(
                os.path.join(user_dir, f"{filename}.md"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(self._user_skill_to_markdown(skill))

        for skill in self.functional_skills:
            filename = self._safe_snapshot_filename(skill.name)
            with open(
                os.path.join(functional_dir, f"{filename}.md"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(self._functional_skill_to_markdown(skill))

    @staticmethod
    def _safe_snapshot_filename(name: str) -> str:
        unsafe = '<>:"/\\|?*'
        filename = "".join(
            "_" if char in unsafe or ord(char) < 32 else char for char in str(name)
        )
        filename = filename.strip(" .")
        if filename in {"", ".", ".."}:
            filename = "unnamed_skill"
        return filename[:180]

    @staticmethod
    def _user_skill_to_markdown(skill: UserProfileSkill) -> str:
        return (
            f"# {skill.name}\n\n"
            f"## Skill Name\n{skill.name}\n\n"
            f"## Description\n{skill.description}\n\n"
            "## Core Preferences\n"
            + "\n".join(f"- {x}" for x in skill.core_preference)
            + "\n\n"
            + "## Secondary Preferences\n"
            + "\n".join(f"- {x}" for x in skill.secondary_preference)
            + "\n\n"
            + "## Exploration Boundary\n"
            + "\n".join(f"- {x}" for x in skill.exploration_boundary)
            + "\n\n"
            + "## Ranking Guidance\n"
            + "\n".join(f"- {x}" for x in skill.ranking_guidance)
            + "\n\n"
            + "## Constraints\n"
            + "\n".join(f"- {x}" for x in skill.constraints)
            + "\n"
        )

    @staticmethod
    def _functional_skill_to_markdown(skill: FunctionalSkill) -> str:
        return (
            f"# {skill.name}\n\n"
            f"- Skill ID: {skill.skill_id}\n"
            f"- Skill Type: {skill.skill_type}\n"
            f"- Dataset Scope: {skill.dataset_scope}\n"
            f"- Version: {skill.version}\n\n"
            f"## Description\n{skill.description}\n\n"
            f"## Purpose\n{skill.purpose}\n\n"
            "## When to Use\n"
            + "\n".join(f"- {x}" for x in skill.when_to_use)
            + "\n\n"
            + "## Input Signals\n"
            + "\n".join(f"- {x}" for x in skill.input_signals)
            + "\n\n"
            + "## How to Apply\n"
            + "\n".join(f"- {x}" for x in skill.how_to_apply)
            + "\n\n"
            + "## Constraints\n"
            + "\n".join(f"- {x}" for x in skill.constraints)
            + "\n"
        )

    def set_user_skills(
        self,
        skills: Iterable[UserProfileSkill],
        reason: str = "init",
    ):
        self.user_skills = list(skills)
        for skill in self.user_skills:
            skill.last_update_reason = reason
        self.save()

    def set_functional_skills(
        self,
        skills: Iterable[FunctionalSkill],
        reason: str = "init",
    ):
        self.functional_skills = list(skills)
        for skill in self.functional_skills:
            skill.last_update_reason = reason
        self.save()

    def upsert_user_skill(self, skill: UserProfileSkill, reason: str):
        self._upsert_skill(skill, self.user_skills, reason)

    def upsert_functional_skill(self, skill: FunctionalSkill, reason: str):
        self._upsert_skill(skill, self.functional_skills, reason)

    def _upsert_skill(self, skill, skill_list, reason: str):
        replaced = False
        for index, existing in enumerate(skill_list):
            if existing.skill_id == skill.skill_id or existing.name == skill.name:
                skill.version = existing.version + 1
                skill_list[index] = skill
                replaced = True
                break
        if not replaced:
            skill_list.append(skill)
        skill.last_update_reason = reason
        self.save()

    def register_outcome(self, selected_skill_names: List[str], success: bool):
        if not selected_skill_names:
            return
        selected = set(selected_skill_names)
        for skill in self.user_skills + self.functional_skills:
            if skill.name in selected:
                skill.register_outcome(success)
        self.save()

    def has_user_skills(self) -> bool:
        return bool(self.user_skills)

    def has_functional_skills(self) -> bool:
        return bool(self.functional_skills)

    def build_default_functional_skills(self):
        """Fallback functional skills used only if LLM initialization fails.

        Even the fallback skills use open-ended natural-language applicability
        conditions so that routing follows the same rule/condition-matching
        mechanism as LLM-initialized and failure-evolved skills.
        """

        if self.functional_skills:
            return

        defaults = [
            FunctionalSkill(
                skill_id="functional_preference_alignment",
                name="preference_alignment",
                skill_type="functional",
                dataset_scope=self.dataset_scope,
                description=(
                    "Prioritize candidates that align with stable recurring "
                    "preference evidence in the user's history."
                ),
                purpose=(
                    "Keep recommendations aligned with strong recurring preference signals."
                ),
                when_to_use=[
                    "the user history provides sufficiently stable recurring preference evidence"
                ],
                input_signals=[
                    "recurring preference cues in the interaction history",
                    "repeated product functions or categories",
                    "consistent interaction patterns",
                ],
                how_to_apply=[
                    "identify the strongest recurring preference cues",
                    "rank candidates satisfying multiple strong cues above one-off matches",
                    "use the prior recommendation only as supporting evidence",
                ],
                constraints=[
                    "do not overfit to a single cue",
                    "do not ignore explicit negative evidence",
                ],
                source="bootstrap_fallback",
            ),
            FunctionalSkill(
                skill_id="functional_short_history_conservative",
                name="short_history_conservative",
                skill_type="functional",
                dataset_scope=self.dataset_scope,
                description=(
                    "Use conservative candidate comparison when preference evidence "
                    "is sparse or uncertain."
                ),
                purpose="Reduce risky recommendations under sparse preference evidence.",
                when_to_use=[
                    "the observed interaction history is too sparse to support a stable fine-grained preference judgment"
                ],
                input_signals=[
                    "few observed interactions",
                    "weak or inconsistent preference evidence",
                ],
                how_to_apply=[
                    "prefer broadly compatible candidates over unsupported niche matches",
                    "avoid fine-grained claims not supported by the observed history",
                    "use available prior evidence only as auxiliary guidance",
                ],
                constraints=[
                    "do not fabricate detailed preferences from sparse history"
                ],
                source="bootstrap_fallback",
            ),
            FunctionalSkill(
                skill_id="functional_negative_feedback_avoidance",
                name="negative_feedback_avoidance",
                skill_type="functional",
                dataset_scope=self.dataset_scope,
                description=(
                    "Suppress candidate patterns that have recently led to explicit "
                    "rejection or repeated recommendation failure."
                ),
                purpose="Avoid repeating recommendation mistakes after negative feedback.",
                when_to_use=[
                    "recent interaction evidence contains explicit rejection or a similar recommendation pattern has failed repeatedly"
                ],
                input_signals=[
                    "recent rejection or dislike",
                    "repeated failed recommendation pattern",
                ],
                how_to_apply=[
                    "identify the attributes associated with recent failed recommendations",
                    "demote candidates sharing clearly rejected attributes",
                    "prefer alternatives that still respect stable user preferences",
                ],
                constraints=[
                    "do not infer dislikes beyond the observed failure evidence"
                ],
                source="bootstrap_fallback",
            ),
            FunctionalSkill(
                skill_id="functional_diversity_when_homogeneous",
                name="diversity_when_homogeneous",
                skill_type="functional",
                dataset_scope=self.dataset_scope,
                description=(
                    "Introduce controlled differentiation when the candidate set is "
                    "highly homogeneous."
                ),
                purpose=(
                    "Differentiate similar candidates without leaving the user's "
                    "preference boundary."
                ),
                when_to_use=[
                    "the candidate set is highly homogeneous and core preference alignment alone is insufficient to distinguish candidates"
                ],
                input_signals=[
                    "highly similar candidate set",
                    "limited differentiation among candidates",
                ],
                how_to_apply=[
                    "first satisfy the user's core preference constraints",
                    "then prefer a candidate that adds useful differentiation within the acceptable boundary",
                ],
                constraints=[
                    "do not diversify beyond the user's exploration boundary"
                ],
                source="bootstrap_fallback",
            ),
        ]

        self.functional_skills = defaults
        self.save()


class PreferenceTemplateBuilder:
    """Fallback utility for constructing a user-profile skill without an LLM."""

    @staticmethod
    def summarize_history(history_items: Sequence[str]) -> Dict[str, List[str]]:
        tokens = []
        for item in history_items:
            tokens.extend(tokenize(item))
        counts = Counter(tokens)
        ranked = [token for token, _ in counts.most_common(12)]
        core = ranked[:4]
        secondary = ranked[4:8]
        boundary = [f"stay close to: {token}" for token in core[:2]]
        guidance = []
        if core:
            guidance.append(
                f"rank items matching multiple core signals: {', '.join(core[:3])}"
            )
        if secondary:
            guidance.append(
                f"use secondary signals only as tie-breakers: {', '.join(secondary[:3])}"
            )
        description = "Prototype profile derived from recurring signals in user history."
        return {
            "description": description,
            "core_preference": core or ["general preference inferred from history"],
            "secondary_preference": secondary,
            "exploration_boundary": boundary
            or ["avoid abrupt jumps away from the dominant history pattern"],
            "ranking_guidance": guidance
            or ["prefer candidates consistent with the strongest history patterns"],
            "applicable_patterns": core + secondary,
        }

    @staticmethod
    def build_skill(
        skill_index: int,
        dataset_scope: str,
        history_items: Sequence[str],
        source: str = "generated",
    ) -> UserProfileSkill:
        summary = PreferenceTemplateBuilder.summarize_history(history_items)
        name_suffix = "_".join(summary["core_preference"][:2]) or f"cluster_{skill_index}"
        name = f"user_profile_{skill_index}_{name_suffix}"[:80]
        return UserProfileSkill(
            skill_id=f"user_profile_{skill_index}",
            name=name,
            skill_type="user_profile",
            dataset_scope=dataset_scope,
            description=summary["description"],
            core_preference=summary["core_preference"],
            secondary_preference=summary["secondary_preference"],
            exploration_boundary=summary["exploration_boundary"],
            ranking_guidance=summary["ranking_guidance"],
            applicable_patterns=summary["applicable_patterns"],
            prototype_histories=[" | ".join(history_items[:10])],
            source=source,
        )
