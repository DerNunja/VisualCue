# Troubleshooting

## LM Studio Context Length

The error below is caused by the context window used when the model is loaded in LM Studio, not by VisualCue's completion token cap:

```text
The number of tokens to keep from the initial prompt is greater than the context length (n_keep: ... >= n_ctx: 9472)
```

The local `lms load --help` command exposes the setting as:

```text
-c, --context-length <length>
```

For long VisualCue requests, load the VLM with an explicit context length large enough for image tokens, prompt text, candidate lists, and completion budget. A practical starting point is `32768`:

```bash
lms unload --all && lms load --context-length 32768 --identifier google/gemma-4-26b-a4b -y google/gemma-4-26b-a4b
```

If `32768` does not fit in VRAM, lower the value and record the largest stable value for the run. The maintenance command in long-run configs should always include `--context-length`; otherwise each reload can fall back to LM Studio's default context window.

## VLM Request Failures

VisualCue skips individual samples instead of aborting the full evaluation when the VLM backend returns token-limit or request errors. Skips are logged in `samples.jsonl` with `skip_reason`.

Relevant skip reasons:
- `vlm_token_limit_exceeded`
- `vlm_request_error`

Use `scripts/analyze_skips.py` to summarize skip categories and correlate them with scene density and prompt token usage.
