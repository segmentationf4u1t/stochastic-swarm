
from typing import List, Dict, Any, Optional, Sequence, Mapping
from dataclasses import dataclass
import time, os, logging
import concurrent.futures as cf
from openai import OpenAI

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/sonoma-sky-alpha")
MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "30000"))  # per-call output cap

# Module logger (unconfigured by default; configured in CLI when executed as script)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__all__ = [
    # New public API
    "StochasticSwarmConfig",
    "StochasticSwarmResult",
    "stochasticswarm",
    "run_stochasticswarm",
]

@dataclass
class StochasticSwarmConfig:
    prompt: str
    n_runs: int
    system_prompt: Optional[str] = None
    stop: Optional[Sequence[str]] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    max_workers: Optional[int] = None
    timeout: Optional[float] = None
    max_retries: int = 3
    backoff_initial: float = 0.5
    backoff_multiplier: float = 2.0
    return_meta: bool = True
    api_key: Optional[str] = None

@dataclass
class StochasticSwarmResult:
    final: str
    candidates: List[str]
    meta: Optional[Mapping[str, Any]] = None

 

def _extract_text(resp) -> str:
    """Extract assistant text from a Chat Completions result.

    Handles multiple SDK content shapes and normalizes to a stripped string.
    """
    try:
        content = resp.choices[0].message.content  # type: ignore[attr-defined]
        if isinstance(content, list):
            normalized: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    # Common shapes: {"text": "..."} or {"type":"text","text":{"value":"..."}}
                    if "text" in part:
                        text_val = part.get("text")
                        if isinstance(text_val, dict) and "value" in text_val:
                            normalized.append(str(text_val.get("value", "")))
                        else:
                            normalized.append(str(text_val))
                        continue
                    if "content" in part:
                        normalized.append(str(part.get("content", "")))
                        continue
                normalized.append(str(part))
            return "".join(normalized).strip()
        if content is None:
            # Fallback to legacy .text field
            text_fallback = getattr(resp.choices[0], "text", "")  # type: ignore[attr-defined]
            return str(text_fallback).strip()
        return str(content).strip()
    except Exception:
        try:
            return str(getattr(resp.choices[0], "text", "") or "").strip()  # type: ignore[attr-defined]
        except Exception:
            return ""


def _one_completion(
    client: OpenAI,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    model: str,
    max_tokens: int,
    top_p: float = 1.0,
    stop: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
    max_retries: int = 3,
    backoff_initial: float = 0.5,
    backoff_multiplier: float = 2.0,
) -> str:
    """Single non-streaming completion (Chat Completions) with retry/backoff.

    Raises the last exception after exhausting retries.
    """
    delay = backoff_initial
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                timeout=timeout,
            )
            return _extract_text(resp)
        except Exception as e:
            last_exc = e
            logger.warning(
                "completion attempt %s/%s failed: %s",
                attempt + 1,
                max_retries,
                getattr(e, "message", repr(e)),
            )
            if attempt == max_retries - 1:
                break
            time.sleep(delay)
            delay *= backoff_multiplier
    assert last_exc is not None
    raise last_exc

def _build_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages

def _build_synthesis_inputs(candidates: List[str]) -> tuple[str, str]:
    """Returns (instructions, user_input) for a synthesis pass with stronger guidance.

    - Filters out empty/whitespace-only candidates
    - Provides concise, directive rules for synthesis
    """
    nonempty = [
        (i + 1, txt.strip()) for i, txt in enumerate(candidates) if txt and txt.strip()
    ]
    if not nonempty:
        numbered = "<cand 1>\n\n</cand 1>"
        count = 0
    else:
        numbered = "\n\n".join(
            f"<cand {idx}>\n{txt}\n</cand {idx}>" for idx, txt in nonempty
        )
        count = len(nonempty)

    instructions = (
        "You are an expert editor and synthesizer. Produce a single, self-contained final answer "
        "that merges the strongest content across the candidate answers while correcting mistakes, "
        "resolving contradictions, and removing redundancy. Follow these rules:\n"
        "1) Do not reference candidates or the editing process.\n"
        "2) Preserve factual accuracy; prefer precise, verifiable details.\n"
        "3) Maintain consistent style and terminology; be concise and clear.\n"
        "4) If candidates disagree, choose the most coherent and technically correct approach; "
        "fill small gaps with well-known best practices when safe.\n"
        "5) If code is appropriate, provide a single complete code block; avoid extraneous prose.\n"
        "6) Match the implied output format in the candidates (bullets, code, steps) when clear; "
        "otherwise use a short, direct style.\n"
        "7) Do not include analysis or chain-of-thought; output only the final answer."
    )
    user = (
        f"You are given {count} candidate answers delimited by <cand i> tags.\n\n"
        f"{numbered}\n\nReturn only the single best final answer."
    )
    return instructions, user

def stochasticswarm(
    prompt: str,
    n_runs: int,
    openrouter_api_key: str | None = None,
    *,
    system_prompt: Optional[str] = None,
    stop: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    max_retries: int = 3,
    backoff_initial: float = 0.5,
    backoff_multiplier: float = 2.0,
    return_meta: bool = True,
) -> Dict[str, Any]:
    """
    Fan out n_runs parallel generations at T=0.9 and synthesize a final answer at T=0.2
    using OpenRouter via the OpenAI Chat Completions API.

    Auth: provide openrouter_api_key or set OPENROUTER_API_KEY env var.
    Returns: {"final": str, "candidates": List[str], "meta"?: Dict[str, Any]}
    """
    assert n_runs >= 1, "n_runs must be >= 1"

    model_to_use = model or MODEL
    max_tokens_to_use = int(max_tokens or MAX_TOKENS)

    api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key not provided. Set OPENROUTER_API_KEY env var or pass api_key."
        )
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )

    started_at = time.monotonic()

    # Parallel candidate generations (threaded)
    effective_workers = max_workers if max_workers is not None else min(n_runs, 16)
    candidates: List[str] = [""] * n_runs  # preserve order
    messages = _build_messages(prompt, system_prompt)
    with cf.ThreadPoolExecutor(max_workers=effective_workers) as ex:
        fut_to_idx = {
            ex.submit(
                _one_completion,
                client,
                messages,
                temperature=0.9,
                model=model_to_use,
                max_tokens=max_tokens_to_use,
                top_p=1.0,
                stop=stop,
                timeout=timeout,
                max_retries=max_retries,
                backoff_initial=backoff_initial,
                backoff_multiplier=backoff_multiplier,
            ): i
            for i in range(n_runs)
        }
        for fut in cf.as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                candidates[i] = fut.result()
            except Exception as e:
                logger.error("candidate %s failed: %s", i + 1, getattr(e, "message", repr(e)))
                candidates[i] = ""

    gen_done_at = time.monotonic()

    # Synthesis pass (use instructions as system message)
    instructions, user = _build_synthesis_inputs(candidates)
    try:
        final_resp = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            top_p=1,
            max_tokens=max_tokens_to_use,
            timeout=timeout,
        )
        final = _extract_text(final_resp)
        synthesis_failed = False
    except Exception as e:
        logger.error("synthesis failed: %s", getattr(e, "message", repr(e)))
        nonempty = [c for c in candidates if c and c.strip()]
        final = max(nonempty, key=lambda s: len(s)) if nonempty else ""
        synthesis_failed = True

    finished_at = time.monotonic()

    result: Dict[str, Any] = {"final": final, "candidates": candidates}
    if return_meta:
        meta: Dict[str, Any] = {
            "model": model_to_use,
            "n_runs": n_runs,
            "max_tokens": max_tokens_to_use,
            "duration_candidates_sec": round(gen_done_at - started_at, 4),
            "duration_total_sec": round(finished_at - started_at, 4),
            "synthesis_failed": synthesis_failed,
        }
        # Attempt to extract token usage if available
        try:
            meta["usage"] = getattr(final_resp, "usage", None)  # type: ignore[name-defined]
        except Exception:
            pass
        result["meta"] = meta
    return result

def run_stochasticswarm(cfg: StochasticSwarmConfig) -> StochasticSwarmResult:
    out = stochasticswarm(
        prompt=cfg.prompt,
        n_runs=cfg.n_runs,
        openrouter_api_key=cfg.api_key,
        system_prompt=cfg.system_prompt,
        stop=cfg.stop,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        max_workers=cfg.max_workers,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        backoff_initial=cfg.backoff_initial,
        backoff_multiplier=cfg.backoff_multiplier,
        return_meta=cfg.return_meta,
    )
    return StochasticSwarmResult(
        final=out.get("final", ""),
        candidates=list(out.get("candidates", [])),
        meta=out.get("meta"),
    )

 

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="StochasticSwarm: parallel candidates + synthesis using OpenRouter")
    parser.add_argument("-p", "--prompt", type=str, help="User prompt text. If omitted, read from stdin.")
    parser.add_argument("-n", "--n-runs", type=int, default=4, help="Number of parallel candidates")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--stop", type=str, default=None, help="Comma-separated stop strings")
    parser.add_argument("--model", type=str, default=None, help="Model id (default from env OPENROUTER_MODEL)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for each call (default from env)")
    parser.add_argument("--max-workers", type=int, default=None, help="Thread pool size (default=min(n_runs,16))")
    parser.add_argument("--timeout", type=float, default=None, help="Per-request timeout in seconds")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (else uses env)")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Print full JSON result")
    parser.add_argument("--print-candidates", action="store_true", help="Print candidates to stdout")

    args = parser.parse_args()

    if args.verbose:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    prompt_arg = args.prompt
    if not prompt_arg:
        data = sys.stdin.read()
        prompt_arg = (data or "").strip()
    if not prompt_arg:
        parser.error("prompt is required via --prompt or stdin")

    stop_list: Optional[List[str]] = None
    if args.stop:
        stop_list = [s for s in (x.strip() for x in args.stop.split(",")) if s]

    result = stochasticswarm(
        prompt=prompt_arg,
        n_runs=args.n_runs,
        openrouter_api_key=args.api_key,
        system_prompt=args.system,
        stop=stop_list,
        model=args.model,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        timeout=args.timeout,
    )

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("final", ""))
        if args.print_candidates:
            print("\n--- candidates ---\n")
            for idx, cand in enumerate(result.get("candidates", []), start=1):
                print(f"<cand {idx}>\n{cand}\n</cand {idx}>\n")
