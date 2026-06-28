import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.core.config import get_settings
from app.services.graph_rag_service import YouMedGraphRAGService
from tools.eval.youmed_graphrag_evaluator import evaluate_all, load_json, print_summary, run_graph_tests_for_eval, save_json


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run graph-only evaluation against FastAPI GraphRAG service classes.")
    parser.add_argument("--cases", required=True, help="Path to eval cases JSON")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--results", default="graph_results.json")
    parser.add_argument("--report", default="graph_eval_report.json")
    args = parser.parse_args()

    eval_cases = load_json(args.cases)
    service = YouMedGraphRAGService(get_settings())

    results = run_graph_tests_for_eval(
        eval_cases,
        ask_graph_func=service.ask_graph,
        limit=args.limit,
        sleep_seconds=args.sleep,
        save_path=args.results,
    )
    report = evaluate_all(eval_cases[: args.limit if args.limit else None], results)
    print_summary(report)
    save_json(report, args.report)


if __name__ == "__main__":
    main()
