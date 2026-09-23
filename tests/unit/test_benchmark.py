"""Tests for benchmark.py"""

import pytest

from rag.evaluator.benchmark import (
    load_benchmark,
    profile_to_chunks,
    sanitize_chunks,
)
from rag.evaluator.faithfulness_checker import FaithfulnessChecker
from rag.evaluator.relevance_scorer import RelevanceScorer


@pytest.mark.unit
class TestBenchmark:
    """Benchmark fixture loading and corpus construction."""

    @pytest.fixture
    def profile(self):
        """A minimal profile fixture with one repository."""
        return {
            "github_username": "example-dev",
            "portfolio_url": "https://example.dev",
            "repos": [
                {
                    "name": "widget-api",
                    "description": "A small API for widgets",
                    "language": "Python",
                    "stargazers_count": 5,
                    "topics": ["fastapi", "widgets"],
                    "readme_content": "# widget-api\n\nA small API for widgets.\n",
                }
            ],
        }

    def test_sanitize_coerces_none_text_to_empty_string(self):
        assert sanitize_chunks([{"text": None}]) == [{"text": ""}]

    def test_sanitize_adds_missing_text_key(self):
        result = sanitize_chunks([{"id": "a"}])
        assert result[0]["text"] == ""
        assert result[0]["id"] == "a"

    def test_sanitize_does_not_mutate_input(self):
        original = {"text": None}
        sanitize_chunks([original])
        assert original == {"text": None}

    def test_sanitized_chunks_are_safe_for_both_scorers(self):
        """The runner, not the scorers, is where null text is neutralised."""
        chunks = sanitize_chunks([{"text": None}, {"text": "python expert"}])

        relevance = RelevanceScorer().score("python", chunks)
        faithfulness = FaithfulnessChecker().check("This developer has Python experience.", chunks)

        assert 0.0 <= relevance <= 1.0
        assert 0.0 <= faithfulness <= 1.0

    def test_profile_to_chunks_emits_overview_plus_two_chunks_per_repo(self, profile):
        assert len(profile_to_chunks(profile)) == 1 + 2 * len(profile["repos"])

    def test_profile_to_chunks_ids_are_unique(self, profile):
        ids = [chunk["id"] for chunk in profile_to_chunks(profile)]
        assert len(ids) == len(set(ids))

    def test_profile_to_chunks_summary_includes_name_language_and_topics(self, profile):
        summary = next(
            c for c in profile_to_chunks(profile) if c["id"] == "repo::widget-api::summary"
        )
        assert "widget-api" in summary["text"]
        assert "Python" in summary["text"]
        assert "fastapi" in summary["text"]

    def test_profile_to_chunks_handles_null_readme_and_description(self):
        profile = {
            "github_username": "example-dev",
            "repos": [{"name": "quiet", "description": None, "readme_content": None}],
        }
        assert all(isinstance(c["text"], str) for c in profile_to_chunks(profile))

    def test_profile_to_chunks_with_no_repos_returns_overview_only(self):
        chunks = profile_to_chunks({"github_username": "example-dev", "repos": []})
        assert len(chunks) == 1
        assert chunks[0]["id"] == "profile-overview"

    def test_load_benchmark_returns_profiles_for_every_case(self):
        profiles = load_benchmark()

        assert len(profiles) >= 3
        for entry in profiles:
            assert entry.cases
            assert all(isinstance(chunk["text"], str) for chunk in entry.chunks)
            for case in entry.cases:
                assert case.query
                assert case.reference_feedback

    def test_load_benchmark_case_ids_are_unique(self):
        ids = [case.case_id for entry in load_benchmark() for case in entry.cases]
        assert len(ids) == len(set(ids))
