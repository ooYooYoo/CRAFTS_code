import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List


@dataclass
class FailureCase:
    user_id: str
    stage: str
    user_history: str
    candidates: str
    interaction_history: List[Dict]
    final_recommendation: str
    ground_truth: str
    user_feedback: str
    selected_skills: List[str] = field(default_factory=list)
    preference_query: str = ""
    state_summary: Dict[str, object] = field(default_factory=dict)
    difficulty: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


class FailureBuffer:
    """Store failures and expose a fresh batch for each evolution step.

    ``cases`` keeps the bounded failure history for inspection, while
    ``pending_cases`` contains only failures that have not yet been analyzed by
    Skill Bank evolution. This avoids repeatedly evolving the Skill Bank from the
    same old failures.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = int(max_size)
        self.cases: List[FailureCase] = []
        self.pending_cases: List[FailureCase] = []
        self.total_added = 0
        self.last_evolved_total = 0

    def add_case(self, case: FailureCase):
        self.cases.append(case)
        self.pending_cases.append(case)
        self.total_added += 1

        if len(self.cases) > self.max_size:
            self.cases = sorted(
                self.cases,
                key=lambda row: row.difficulty,
                reverse=True,
            )[: self.max_size]

        if len(self.pending_cases) > self.max_size:
            self.pending_cases = sorted(
                self.pending_cases,
                key=lambda row: row.difficulty,
                reverse=True,
            )[: self.max_size]

    def should_trigger(self, interval: int) -> bool:
        return len(self.pending_cases) >= max(1, int(interval))

    def mark_evolved(self):
        self.last_evolved_total = self.total_added
        self.pending_cases = []

    def top_cases(self, limit: int) -> List[FailureCase]:
        return sorted(
            self.pending_cases,
            key=lambda row: row.difficulty,
            reverse=True,
        )[: max(1, int(limit))]

    def stats(self) -> Dict[str, object]:
        if not self.cases:
            return {
                "size": 0,
                "pending_size": 0,
                "total_added": self.total_added,
            }
        return {
            "size": len(self.cases),
            "pending_size": len(self.pending_cases),
            "total_added": self.total_added,
            "avg_difficulty": sum(case.difficulty for case in self.cases)
            / max(1, len(self.cases)),
        }

    def save(self, filepath: str):
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "max_size": self.max_size,
                    "total_added": self.total_added,
                    "last_evolved_total": self.last_evolved_total,
                    "cases": [case.to_dict() for case in self.cases],
                    "pending_cases": [
                        case.to_dict() for case in self.pending_cases
                    ],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, filepath: str) -> "FailureBuffer":
        with open(filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        buffer = cls(max_size=payload.get("max_size", 200))
        buffer.total_added = payload.get("total_added", 0)
        buffer.last_evolved_total = payload.get("last_evolved_total", 0)
        buffer.cases = [
            FailureCase(**row) for row in payload.get("cases", [])
        ]
        buffer.pending_cases = [
            FailureCase(**row) for row in payload.get("pending_cases", [])
        ]
        return buffer
