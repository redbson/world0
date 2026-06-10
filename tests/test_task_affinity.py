"""Tests: graded task affinity (dynamics/affinity.py).

The contract: exact substring hits keep scoring 1.0 (legacy behavior
preserved bit-for-bit), partial token overlap earns a proportional
score, and trivial overlap stays below the activation engine's
MIN_TASK_AFFINITY gate.
"""

from __future__ import annotations

from world0.dynamics.activation import MIN_TASK_AFFINITY
from world0.dynamics.affinity import task_affinity, task_tokens


class TestSubstringEquivalence:
    def test_exact_substring_scores_one(self):
        assert task_affinity("debug latency", ["debug latency spike"]) == 1.0

    def test_identical_task_scores_one(self):
        assert task_affinity("bootstrap", ["bootstrap"]) == 1.0

    def test_substring_wins_over_partial_entries(self):
        # One entry is a partial overlap, another is a substring hit;
        # the substring must dominate regardless of iteration order.
        history = ["latency tuning", "full debug latency run"]
        assert task_affinity("debug latency", history) == 1.0


class TestGradedOverlap:
    def test_token_overlap_scores_fraction(self):
        # "debug latency spike" vs "latency debugging" — no substring,
        # but 'latency' (1 of 3 signature tokens) overlaps.
        score = task_affinity(
            "debug latency spike", ["latency profiling session"]
        )
        assert 0.0 < score < 1.0
        assert abs(score - 1 / 3) < 1e-9

    def test_full_token_overlap_without_substring(self):
        # All task tokens present, but word order prevents substring.
        score = task_affinity("latency debug", ["debug the latency issue"])
        assert score == 1.0

    def test_best_entry_wins(self):
        history = ["unrelated work", "debug latency now"]
        assert task_affinity("debug latency", history) == 1.0


class TestThresholdAndEdges:
    def test_no_overlap_scores_zero(self):
        assert task_affinity("debug latency", ["design review"]) == 0.0

    def test_empty_task_scores_zero(self):
        assert task_affinity("", ["anything"]) == 0.0

    def test_empty_history_scores_zero(self):
        assert task_affinity("debug latency", []) == 0.0

    def test_single_shared_token_stays_below_gate(self):
        # One of four tokens shared → 0.25, below MIN_TASK_AFFINITY:
        # coincidental overlap must not fire the boost.
        score = task_affinity(
            "debug cache layer eviction", ["cache warming strategy"]
        )
        assert 0.0 < score < MIN_TASK_AFFINITY


class TestCJKTokens:
    def test_cjk_task_substring(self):
        assert task_affinity("延迟调试", ["生产环境延迟调试记录"]) == 1.0

    def test_task_tokens_handles_cjk(self):
        # tokenize_signature must not blow up on CJK text; behavior
        # (token granularity) is delegated to the shared tokenizer.
        toks = task_tokens("延迟 调试 latency")
        assert "latency" in toks
