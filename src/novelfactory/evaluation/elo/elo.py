"""Elo 评分系统核心实现。

参考 DebateQD (NeurIPS 2025 Workshop) 的 PersuasionElo 和 TruthElo 设计。
适用于辩论系统的双方能力追踪和收敛判定。

用法:
    elo = PersuasionElo(k_factor=32)
    elo.register_player("editor")
    elo.register_player("reader")
    elo.record_match("editor", "reader", winner="editor")
    rating = elo.get_rating("editor")  # 1000 + 16 = 1016
"""

from __future__ import annotations

import math

K_FACTOR = 32
INITIAL_RATING = 1000


def expected_score(rating_a: float, rating_b: float) -> float:
    """计算 A 对 B 的预期胜率。

    Args:
        rating_a: 玩家 A 的 Elo 评分
        rating_b: 玩家 B 的 Elo 评分

    Returns:
        玩家 A 的预期胜率 (0-1)
    """
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def update_elo(
    winner_rating: float,
    loser_rating: float,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """更新 Elo 评分，胜者加分败者扣分。

    Args:
        winner_rating: 胜者当前评分
        loser_rating: 败者当前评分
        k: K 因子（影响评分波动幅度）

    Returns:
        (新胜者评分, 新败者评分)
    """
    expected_winner = expected_score(winner_rating, loser_rating)
    expected_loser = 1.0 - expected_winner
    new_winner = winner_rating + k * (1.0 - expected_winner)
    new_loser = loser_rating + k * (0.0 - expected_loser)
    return new_winner, new_loser


def update_elo_tie(
    rating_a: float,
    rating_b: float,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """平局时更新 Elo 评分。

    Args:
        rating_a: 玩家 A 当前评分
        rating_b: 玩家 B 当前评分
        k: K 因子

    Returns:
        (新玩家A评分, 新玩家B评分)
    """
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    new_a = rating_a + k * (0.5 - expected_a)
    new_b = rating_b + k * (0.5 - expected_b)
    return new_a, new_b


class EloSystem:
    """基础 Elo 评分系统。

    管理多个玩家的评分，支持注册、记录比赛、查询评分。
    """

    def __init__(
        self, k_factor: float = K_FACTOR, initial_rating: float = INITIAL_RATING
    ):
        self._ratings: dict[str, float] = {}
        self._match_count: dict[str, int] = {}
        self._k = k_factor
        self._initial = initial_rating

    def register_player(
        self, player_id: str, initial_rating: float | None = None
    ) -> None:
        """注册一个新玩家。

        Args:
            player_id: 玩家唯一标识
            initial_rating: 初始评分，默认 1000
        """
        if player_id not in self._ratings:
            self._ratings[player_id] = initial_rating or self._initial
            self._match_count[player_id] = 0

    def get_rating(self, player_id: str) -> float:
        """获取玩家当前评分。"""
        return self._ratings.get(player_id, self._initial)

    def get_match_count(self, player_id: str) -> int:
        """获取玩家比赛场次。"""
        return self._match_count.get(player_id, 0)

    def get_all_ratings(self) -> dict[str, float]:
        """获取所有玩家评分。"""
        return dict(self._ratings)

    def expected_win_probability(self, player_a: str, player_b: str) -> float:
        """计算 A 对 B 的预期胜率。"""
        return expected_score(
            self._ratings.get(player_a, self._initial),
            self._ratings.get(player_b, self._initial),
        )

    def rating_spread(self, *player_ids: str) -> float:
        """计算指定玩家间的最大评分差距。

        用于辩论收敛判定：差距 < 阈值时说明辩论效果趋同。
        """
        ratings = [self._ratings.get(pid, self._initial) for pid in player_ids]
        if not ratings:
            return 0.0
        return max(ratings) - min(ratings)

    def reset(self, player_id: str | None = None) -> None:
        """重置评分。"""
        if player_id:
            self._ratings[player_id] = self._initial
            self._match_count[player_id] = 0
        else:
            for pid in self._ratings:
                self._ratings[pid] = self._initial
                self._match_count[pid] = 0


class PersuasionElo(EloSystem):
    """Persuasion Elo — 策略间胜负追踪。

    适用于 Editor vs Reader 的辩论胜率追踪。
    每次辩论判定谁"说服"了对方（Persuasion Elo 不关注谁是对的，只关注谁赢了）。
    """

    def record_match(
        self,
        winner: str,
        loser: str,
    ) -> None:
        """记录一场比赛结果。

        Args:
            winner: 胜者 ID
            loser: 败者 ID
        """
        for pid in (winner, loser):
            if pid not in self._ratings:
                self.register_player(pid)

        new_winner, new_loser = update_elo(
            self._ratings[winner],
            self._ratings[loser],
            self._k,
        )
        self._ratings[winner] = new_winner
        self._ratings[loser] = new_loser
        self._match_count[winner] += 1
        self._match_count[loser] += 1

    def record_tie(self, player_a: str, player_b: str) -> None:
        """记录一场平局。"""
        for pid in (player_a, player_b):
            if pid not in self._ratings:
                self.register_player(pid)

        new_a, new_b = update_elo_tie(
            self._ratings[player_a],
            self._ratings[player_b],
            self._k,
        )
        self._ratings[player_a] = new_a
        self._ratings[player_b] = new_b
        self._match_count[player_a] += 1
        self._match_count[player_b] += 1


class TruthElo(EloSystem):
    """Truth Elo — 团队协作正确率追踪。

    适用于 Editor+Reader 合作帮助 quality_gate 做正确决策。
    使用双 Elo（团队能力 + 问题难度）建模。
    """

    def __init__(
        self, k_factor: float = K_FACTOR, initial_rating: float = INITIAL_RATING
    ):
        super().__init__(k_factor, initial_rating)
        self._question_ratings: dict[str, float] = {}  # 问题难度评分

    def register_question(
        self, question_id: str, difficulty: float | None = None
    ) -> None:
        """注册一个问题。"""
        if question_id not in self._question_ratings:
            self._question_ratings[question_id] = difficulty or self._initial

    def record_judgment(
        self,
        team_id: str,
        question_id: str,
        correct: bool,
    ) -> tuple[float, float]:
        """记录一次评审判决。

        Args:
            team_id: 评审团队 ID
            question_id: 问题 ID
            correct: 是否判断正确

        Returns:
            (新团队评分, 新问题难度评分)
        """
        if team_id not in self._ratings:
            self.register_player(team_id)
        if question_id not in self._question_ratings:
            self.register_question(question_id)

        team_rating = self._ratings[team_id]
        question_rating = self._question_ratings[question_id]

        # 计算预期正确率
        expected = expected_score(team_rating, question_rating)

        # 更新
        actual = 1.0 if correct else 0.0
        new_team = team_rating + self._k * (actual - expected)
        new_question = question_rating + self._k * (expected - actual)  # 反向更新

        self._ratings[team_id] = new_team
        self._question_ratings[question_id] = new_question
        self._match_count[team_id] = self._match_count.get(team_id, 0) + 1

        return new_team, new_question

    def get_question_rating(self, question_id: str) -> float:
        """获取问题难度评分。"""
        return self._question_ratings.get(question_id, self._initial)
