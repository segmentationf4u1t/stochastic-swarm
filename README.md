## StochasticSwarm

StochasticSwarm is a tiny Python utility that fans out multiple LLM candidates in parallel and synthesizes them into a single final answer. It talks to OpenRouter via the OpenAI Chat Completions API surface.

### Why

- **Breadth + quality**: Sample several diverse candidates, then synthesize a concise, accurate final.
- **Fast**: Parallel fan-out with a synthesis pass. Retries with backoff built in.
- **Flexible**: System prompt, stop tokens, timeouts, model and token caps, and more.

## Requirements

- Python 3.10+
- Python package: `openai` (v1+)

```bash
pip install openai
```

## Configuration

Set an OpenRouter key and optionally override defaults via environment variables.

- **OPENROUTER_API_KEY**: required (e.g., `sk-or-v1-...`)
- **OPENROUTER_BASE_URL**: defaults to `https://openrouter.ai/api/v1`
- **OPENROUTER_MODEL**: defaults to `openrouter/sonoma-sky-alpha`
- **OPENROUTER_MAX_TOKENS**: defaults to `30000` (per-call cap)
- **LOG_LEVEL**: when `--verbose` is used (e.g., `INFO`, `DEBUG`)

## CLI Usage

```bash
export OPENROUTER_API_KEY=...  # or set in your shell/OS env

python main.py \
  --prompt "Locate the error and fix it {code}" \
  --n-runs 6 \
  --system "Be concise and precise." \
  --stop "</END>" \
  --model openrouter/sonoma-sky-alpha \
  --max-tokens 2048 \
  --timeout 30 \
  --verbose \
  --print-candidates
```

Print full JSON (includes `meta` when enabled):

```bash
python main.py -p "Design a REST API for notes" -n 5 --json
```

See all options:

```bash
python main.py -h
```

## Library Usage

### Minimal

```python
from main import stochasticswarm

result = stochasticswarm(
    prompt="Summarize the topic.",
    n_runs=5,
)

print(result["final"])          # synthesized best answer
print(len(result["candidates"])) # all raw candidates
print(result.get("meta"))         # timings, usage (when present), etc.
```

### With config dataclass

```python
from main import StochasticSwarmConfig, run_stochasticswarm

cfg = StochasticSwarmConfig(
    prompt="Generate a TypeScript function and tests.",
    n_runs=6,
    system_prompt="Write robust, readable code and explain briefly.",
    stop=["</END>"],
    model="openrouter/sonoma-sky-alpha",
    max_tokens=2048,
    timeout=30,
)

res = run_stochasticswarm(cfg)
print(res.final)
```

