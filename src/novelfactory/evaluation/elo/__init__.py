"""Elo 评分模块。

参考 DebateQD (NeurIPS 2025 Workshop, arXiv:2510.05909)。
提供 Persuasion Elo 和 Truth Elo 两种评分模式，用于辩论系统的收敛判定和角色能力追踪。
"""

from novelfactory.evaluation.elo.elo import (
    INITIAL_RATING,
    K_FACTOR,
    EloSystem,
    PersuasionElo,
    TruthElo,
    expected_score,
    update_elo,
    update_elo_tie,
)

__all__ = [
    "EloSystem",
    "PersuasionElo",
    "TruthElo",
    "expected_score",
    "update_elo",
    "update_elo_tie",
    "K_FACTOR",
    "INITIAL_RATING",
]
