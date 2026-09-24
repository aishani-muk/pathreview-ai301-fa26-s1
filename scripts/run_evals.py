"""Run the RAG evaluation suite against benchmark portfolios."""

import json
from pathlib import Path

from rag.evaluator.runner import run_benchmark

OUTPUT_PATH = Path("eval_results.json")


def main() -> None:
    """Execute the full evaluation pipeline and output results."""
    print("Running RAG evaluation suite...")

    report = run_benchmark()
    OUTPUT_PATH.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    summary = report.summary
    print(
        f"Scored {summary['cases']} cases across {summary['profiles']} profiles: "
        f"relevance {summary['mean_relevance']}, "
        f"faithfulness {summary['mean_faithfulness']}, "
        f"overall {summary['mean_overall']}."
    )
    print(f"Evaluation complete. Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
