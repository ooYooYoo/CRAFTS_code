import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

from .skill_bank import SkillBank, UserProfileSkill


@dataclass
class SkillRetrievalRecord:
    query_text: str
    positive_skill_text: str
    negative_skill_text: str
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        payload = asdict(self)
        if payload.get("metadata") is None:
            payload["metadata"] = {}
        return payload


class PairwiseSkillDataset(Dataset):
    def __init__(self, records: Sequence[SkillRetrievalRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> SkillRetrievalRecord:
        return self.records[idx]


class ProjectionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TrainableSkillRetriever(nn.Module):
    """Frozen sentence encoder + trainable projection for user-profile skills."""

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.text_encoder = SentenceTransformer(self.encoder_name)
        if hasattr(self.text_encoder, "eval"):
            self.text_encoder.eval()
        if hasattr(self.text_encoder, "parameters"):
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

        try:
            encoder_dim = int(self.text_encoder.get_sentence_embedding_dimension())
        except Exception:
            test_vec = self.text_encoder.encode(
                ["test"],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            encoder_dim = int(test_vec.shape[1])

        self.projector = ProjectionMLP(
            input_dim=encoder_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout,
        ).to(self.device)

    @staticmethod
    def user_skill_to_text(skill: UserProfileSkill) -> str:
        parts = [
            f"Skill Name: {skill.name}",
            f"Description: {skill.description}",
            "Core Preferences: " + ", ".join(skill.core_preference),
            "Secondary Preferences: " + ", ".join(skill.secondary_preference),
            "Exploration Boundary: " + ", ".join(skill.exploration_boundary),
            "Ranking Guidance: " + ", ".join(skill.ranking_guidance),
            "Constraints: " + ", ".join(getattr(skill, "constraints", [])),
        ]
        return "\n".join(p for p in parts if p and p.strip())

    def _encode_texts_base(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must not be empty")
        vecs = self.text_encoder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return torch.tensor(vecs, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        self.projector.eval()
        base = self._encode_texts_base([text])
        return self.projector(base)[0]

    @torch.no_grad()
    def encode_skills(self, texts: Sequence[str]) -> torch.Tensor:
        self.projector.eval()
        base = self._encode_texts_base(texts)
        return self.projector(base)

    @torch.no_grad()
    def score(self, query_text: str, skill_texts: Sequence[str]) -> torch.Tensor:
        q = self.encode_query(query_text).unsqueeze(0)
        s = self.encode_skills(skill_texts)
        return torch.matmul(s, q.transpose(0, 1)).squeeze(-1)

    @torch.no_grad()
    def retrieve_topk(
        self,
        query_text: str,
        skills: Sequence[UserProfileSkill],
        top_k: int = 2,
        similarity_threshold: Optional[float] = None,
        use_success_bias: bool = False,
    ) -> List[Tuple[float, UserProfileSkill]]:
        if not skills:
            return []

        skill_texts = [self.user_skill_to_text(skill) for skill in skills]
        scores = self.score(query_text, skill_texts).detach().cpu().numpy().tolist()

        ranked = []
        for score, skill in zip(scores, skills):
            final_score = float(score)
            if use_success_bias:
                final_score += 0.05 * float(getattr(skill, "success_rate", 0.0))
            if similarity_threshold is None or final_score >= similarity_threshold:
                ranked.append((final_score, skill))

        ranked.sort(key=lambda row: row[0], reverse=True)
        return ranked[:top_k]

    def save_projector(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "encoder_name": self.encoder_name,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "projector_state_dict": self.projector.state_dict(),
        }
        torch.save(payload, path)

    def load_projector(self, path: str, strict: bool = True):
        payload = torch.load(path, map_location=self.device)
        self.projector.load_state_dict(
            payload["projector_state_dict"],
            strict=strict,
        )
        self.projector.to(self.device)
        self.projector.eval()


def build_pairwise_records_from_logs(
    interaction_logs: Sequence[Dict],
    skill_bank: SkillBank,
    min_query_len: int = 8,
    negatives_per_positive: int = 2,
    random_seed: int = 2018,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    negative_strategy: str = "bottom",
    bottom_ratio: float = 0.3,
    middle_ratio: float = 0.3,
    positive_similarity_threshold: Optional[float] = None,
    use_success_only: bool = True,
) -> List[SkillRetrievalRecord]:
    """Build (q, s+, s-) supervision from successful train interactions.

    Only user-profile skills participate in this retriever. Positive skills are the
    user-profile skills selected for successful trajectories; negatives are sampled
    from the remaining user-profile skills.
    """

    rng = random.Random(random_seed)
    skill_by_name = {skill.name: skill for skill in skill_bank.user_skills}
    all_user_skill_names = list(skill_by_name.keys())
    records: List[SkillRetrievalRecord] = []

    if len(all_user_skill_names) < 2:
        return records

    encoder = SentenceTransformer(encoder_name)
    if hasattr(encoder, "eval"):
        encoder.eval()

    skill_text_by_name = {
        name: TrainableSkillRetriever.user_skill_to_text(skill)
        for name, skill in skill_by_name.items()
    }
    all_skill_texts = [skill_text_by_name[name] for name in all_user_skill_names]
    all_skill_vecs = encoder.encode(
        all_skill_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    name_to_index = {
        name: idx for idx, name in enumerate(all_user_skill_names)
    }

    def _rank_skills_by_query(query_text: str) -> List[Tuple[str, float]]:
        query_vec = encoder.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        scores = np.dot(all_skill_vecs, query_vec)
        ranked_pairs = [
            (name, float(scores[name_to_index[name]]))
            for name in all_user_skill_names
        ]
        ranked_pairs.sort(key=lambda row: row[1], reverse=True)
        return ranked_pairs

    def _sample_negatives(
        ranked_pairs: List[Tuple[str, float]],
        positive_names: Sequence[str],
        candidate_skill_names: Sequence[str],
    ) -> List[str]:
        positive_set = set(positive_names)
        candidate_set = (
            set(candidate_skill_names)
            if candidate_skill_names
            else set(all_user_skill_names)
        )

        ordered = [
            name
            for name, _ in ranked_pairs
            if name in candidate_set and name not in positive_set
        ]
        if not ordered:
            ordered = [
                name for name, _ in ranked_pairs if name not in positive_set
            ]
        if not ordered:
            return []

        strategy = str(negative_strategy or "bottom").lower()
        if strategy == "random":
            pool = ordered
        elif strategy == "mixed":
            bottom_n = max(
                negatives_per_positive,
                int(len(ordered) * bottom_ratio),
            )
            bottom_pool = ordered[-bottom_n:]

            middle_n = max(
                negatives_per_positive,
                int(len(ordered) * middle_ratio),
            )
            center = len(ordered) // 2
            half = max(1, middle_n // 2)
            left = max(0, center - half)
            right = min(len(ordered), center + half)
            middle_pool = ordered[left:right]
            pool = list(dict.fromkeys(bottom_pool + middle_pool))
        else:
            bottom_n = max(
                negatives_per_positive,
                int(len(ordered) * bottom_ratio),
            )
            pool = ordered[-bottom_n:]

        if not pool:
            pool = ordered
        return rng.sample(
            pool,
            k=min(negatives_per_positive, len(pool)),
        )

    for row in interaction_logs:
        query_text = str(row.get("preference_query", "")).strip()
        if len(query_text) < min_query_len:
            continue

        success = bool(row.get("success", False))
        if use_success_only and not success:
            continue

        selected_skill_names = [
            name
            for name in row.get("selected_skills", [])
            if name in skill_by_name
        ]
        if not selected_skill_names:
            continue

        candidate_skill_names = [
            name
            for name in row.get("candidate_skill_names", [])
            if name in skill_by_name
        ]
        if not candidate_skill_names:
            candidate_skill_names = all_user_skill_names

        ranked_pairs = _rank_skills_by_query(query_text)
        score_by_name = dict(ranked_pairs)

        positive_names = selected_skill_names
        if positive_similarity_threshold is not None:
            positive_names = [
                name
                for name in positive_names
                if score_by_name.get(name, -1.0)
                >= positive_similarity_threshold
            ]
            if not positive_names:
                continue

        sampled_negatives = _sample_negatives(
            ranked_pairs=ranked_pairs,
            positive_names=positive_names,
            candidate_skill_names=candidate_skill_names,
        )
        if not sampled_negatives:
            continue

        for pos_name in positive_names:
            pos_text = skill_text_by_name[pos_name]
            for neg_name in sampled_negatives:
                records.append(
                    SkillRetrievalRecord(
                        query_text=query_text,
                        positive_skill_text=pos_text,
                        negative_skill_text=skill_text_by_name[neg_name],
                        metadata={
                            "positive_skill_name": pos_name,
                            "negative_skill_name": neg_name,
                            "success": success,
                            "negative_strategy": negative_strategy,
                            "positive_similarity": score_by_name.get(pos_name),
                            "negative_similarity": score_by_name.get(neg_name),
                            "user_id": row.get("user_id"),
                            "round_idx": row.get("round_idx"),
                        },
                    )
                )

    return records


def _pairwise_collate_fn(
    batch: Sequence[SkillRetrievalRecord],
) -> Dict[str, List[str]]:
    return {
        "query_texts": [x.query_text for x in batch],
        "positive_skill_texts": [x.positive_skill_text for x in batch],
        "negative_skill_texts": [x.negative_skill_text for x in batch],
        "metadata": [x.metadata or {} for x in batch],
    }


def train_skill_retriever(
    retriever: TrainableSkillRetriever,
    records: Sequence[SkillRetrievalRecord],
    save_path: Optional[str] = None,
    batch_size: int = 32,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    margin: Optional[float] = None,
    temperature: float = 1.0,
    grad_clip_norm: Optional[float] = 1.0,
    num_workers: int = 0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    if not records:
        raise ValueError("No training records provided.")

    print("[SkillRetriever] Training started...")
    print(f"[SkillRetriever] Number of records: {len(records)}")

    dataset = PairwiseSkillDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_pairwise_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        retriever.projector.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {"epoch_loss": []}
    retriever.projector.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 0

        for batch in loader:
            q_base = retriever._encode_texts_base(batch["query_texts"])
            p_base = retriever._encode_texts_base(
                batch["positive_skill_texts"]
            )
            n_base = retriever._encode_texts_base(
                batch["negative_skill_texts"]
            )

            q = retriever.projector(q_base)
            p = retriever.projector(p_base)
            n = retriever.projector(n_base)

            pos_scores = (q * p).sum(dim=-1) / max(temperature, 1e-6)
            neg_scores = (q * n).sum(dim=-1) / max(temperature, 1e-6)

            if margin is None:
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            else:
                target = torch.ones_like(pos_scores)
                loss = F.margin_ranking_loss(
                    pos_scores,
                    neg_scores,
                    target,
                    margin=margin,
                )

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    retriever.projector.parameters(),
                    grad_clip_norm,
                )
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        history["epoch_loss"].append(avg_loss)
        if verbose:
            print(
                f"[SkillRetriever] epoch={epoch}/{epochs} "
                f"loss={avg_loss:.6f}"
            )

    retriever.projector.eval()
    if save_path:
        retriever.save_projector(save_path)

    return history


def save_records_jsonl(
    records: Sequence[SkillRetrievalRecord],
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def load_records_jsonl(path: str) -> List[SkillRetrievalRecord]:
    records: List[SkillRetrievalRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(SkillRetrievalRecord(**json.loads(line)))
    return records
