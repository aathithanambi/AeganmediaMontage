"""Backward-compatible entry point — delegates to workers/orchestrator.

worker.py invokes `python -m webapp.pipeline_runner`; this shim keeps
that command working without any configuration change.
"""
from webapp.workers.orchestrator import main, run_pipeline
from webapp.workers.shared import get_api_usage

__all__ = ["run_pipeline", "main", "get_api_usage"]

if __name__ == "__main__":
    main()
