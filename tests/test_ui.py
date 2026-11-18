"""
Automated Verification Tests for Member 4's Explainable Streamlit Dashboard & Ablation Data.
"""

import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_app_syntax_and_functions():
    """Verify app/streamlit_app.py exists, parses cleanly via AST, and defines core XAI functions."""
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    assert app_path.exists(), "app/streamlit_app.py is missing"

    source_code = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source_code, filename="streamlit_app.py")

    defined_functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    expected_functions = {
        "load_corpus_chunks",
        "load_benchmark_data",
        "compute_live_hybrid_retrieval",
        "execute_explainable_agent_workflow",
    }
    assert expected_functions.issubset(defined_functions)


def test_streamlit_dashboard_exposes_all_4_explainable_tabs():
    """Verify the UI exposes the Agentic Loop, Math Inspector, RAG Triad Benchmark, and Conflict Matrix."""
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    source_code = app_path.read_text(encoding="utf-8")

    assert "Explainable Case Advisor" in source_code
    assert "Mathematical IR Inspector" in source_code
    assert "RAG Triad Benchmark" in source_code
    assert "Conflict Matrix" in source_code


def test_ablation_study_and_benchmark_json_artifact():
    """Verify data/evaluation/benchmark_results.json has 3 ablation modes and 32 case results."""
    bench_path = PROJECT_ROOT / "data" / "evaluation" / "benchmark_results.json"
    assert bench_path.exists(), "benchmark_results.json is missing"

    data = json.loads(bench_path.read_text(encoding="utf-8"))
    assert data["summary"]["total_cases"] == 32
    assert len(data["ablation_study"]) == 3
    assert len(data["case_results"]) == 32