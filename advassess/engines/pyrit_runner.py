"""Runs one PyRIT attack job in its own process.

    python -m advassess.engines.pyrit_runner job.json result.json

Kept separate from the main program so PyRIT stays optional, its async code and memory
database live in a throwaway process, and a crash or hang cannot take the assessment down.
The job file holds endpoints and the NAME of the environment variable for each key, never a key.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time


def _target(spec: dict, max_tokens: int):
    from pyrit.prompt_target import OpenAIChatTarget

    key = os.environ.get(spec["api_key_env"], "") if spec.get("api_key_env") else ""
    return OpenAIChatTarget(model_name=spec["model"], endpoint=spec["endpoint"], api_key=key or "not-needed",
                            max_completion_tokens=max_tokens)


def _scorer(spec: dict, judge_target):
    from pyrit.score import SelfAskTrueFalseScorer, SubStringScorer, TrueFalseQuestion

    if spec["type"] == "substring":
        return SubStringScorer(substring=spec["substring"], categories=["objective"])
    q = spec["question"]
    return SelfAskTrueFalseScorer.from_question(
        chat_target=judge_target,
        question=TrueFalseQuestion(true_description=q["true_description"],
                                   false_description=q["false_description"], category="objective"))


def _text(piece) -> str:
    return str(getattr(piece, "converted_value", None) or getattr(piece, "original_value", "") or "")


async def _run(job: dict) -> dict:
    from pyrit.executor.attack import (AttackAdversarialConfig, AttackScoringConfig, CrescendoAttack,
                                        PromptSendingAttack, RedTeamingAttack)
    from pyrit.memory import CentralMemory
    from pyrit.setup import IN_MEMORY, initialize_pyrit_async

    await initialize_pyrit_async(memory_db_type=IN_MEMORY)
    memory = CentralMemory.get_memory_instance()
    tokens = job["max_completion_tokens"]
    objective_target = _target(job["objective_target"], tokens)
    adversarial = _target(job["adversarial_target"], tokens) if job.get("adversarial_target") else None
    judge = _target(job["scoring_target"], tokens) if job.get("scoring_target") else adversarial
    scoring = AttackScoringConfig(objective_scorer=_scorer(job["scoring"], judge))
    strategy = job["strategy"]

    def build():
        if strategy == "prompt_sending":
            return PromptSendingAttack(objective_target=objective_target, attack_scoring_config=scoring)
        adv = AttackAdversarialConfig(target=adversarial)
        if strategy == "red_teaming":
            return RedTeamingAttack(objective_target=objective_target, attack_adversarial_config=adv,
                                    attack_scoring_config=scoring, max_turns=job["max_turns"])
        return CrescendoAttack(objective_target=objective_target, attack_adversarial_config=adv,
                               attack_scoring_config=scoring, max_turns=job["max_turns"])

    outcomes = []
    for objective in job["objectives"]:
        started = time.perf_counter()
        row = {"objective": objective, "achieved": False, "outcome": "ERROR", "turns": 0, "transcript": [],
               "reason": "", "error": None}
        try:
            result = await build().execute_async(objective=objective)
            row["outcome"] = result.outcome.name
            row["achieved"] = result.outcome.name == "SUCCESS"
            row["turns"] = int(result.executed_turns or 0)
            row["reason"] = str(result.outcome_reason or "")[:300]
            pieces = sorted(memory.get_message_pieces(conversation_id=result.conversation_id),
                            key=lambda p: (getattr(p, "sequence", 0), getattr(p, "timestamp", 0)))
            row["transcript"] = [{"role": str(p.api_role), "text": _text(p)[:2000]} for p in pieces
                                 if str(p.api_role) in ("user", "assistant")]
            if result.error_message:
                row["error"] = str(result.error_message)[:300]
        except Exception as exc:  # one failed objective must not lose the others
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        row["seconds"] = round(time.perf_counter() - started, 1)
        outcomes.append(row)
    return {"strategy": strategy, "objectives": outcomes}


def main() -> int:
    job_path, out_path = sys.argv[1], sys.argv[2]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    result = asyncio.run(_run(job))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
