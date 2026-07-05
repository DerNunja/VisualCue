# VisualCue

## Run The Demo

Start an OpenAI-compatible VLM server first, for example LM Studio at `http://localhost:1234/v1`, and load the model you want to use.

Then run the interactive Streamlit demo:

```bash
streamlit run app/streamlit_app.py
```

The app reuses the same `SequentialPipeline` and `AgenticPipeline` as the evaluation harness. Falcon is cached as a warm Streamlit resource, so switching systems, prompt style, `max_steps`, reasoning, or VLM connection settings does not reload the segmentation model.

## Run Evaluation Jobs

Run all commands from the repository root.

### Prerequisites

Install the Python environment and make sure CUDA is available for Falcon Perception. For VLM-based systems, start an OpenAI-compatible server such as LM Studio and load the model configured in the YAML file.

Prepare the local datasets once:

```bash
uv run python scripts/download_refcocog.py data/refcocog --split val --split-by umd
uv run python scripts/download_fsc147.py data/fsc147
```

Dataset files are ignored by Git and stay local.

### Development Runs

SequentialPipeline, 10 random samples per dataset:

```bash
uv run python scripts/run_eval.py --config configs/sequential_dev.yaml
```

AgenticPipeline, 10 random samples per dataset:

```bash
uv run python scripts/run_eval.py --config configs/agentic_dev.yaml
```

The development configs write summaries and sample traces to:

```text
results/dev_sequential_logging/
results/dev_agentic/
```

Each run produces one summary JSON and one `__samples.jsonl` file per system/dataset pair. The `samples.jsonl` files include `intermediate`, including prompt traces and Agentic loop steps.

### Full Falcon Baseline

The committed full Falcon baseline summaries were produced with the Falcon-only system over the full RefCOCOg val and FSC-147 test splits. To reproduce a full baseline run, use a config with `sample_limit` omitted or set to `null`, `system: falcon_only`, and `out_dir: results/falcon_full`, then run:

```bash
uv run python scripts/run_eval.py --config <your-full-falcon-config.yaml>
```

Full runs can take substantially longer than development runs. Keep qualitative overlays local; JSON summaries and `__samples.jsonl` logs are tracked.

### Outputs

For a system named `agentic_pipeline` on RefCOCOg, the harness writes:

```text
results/<out_dir>/agentic_pipeline__refcocog.json
results/<out_dir>/agentic_pipeline__refcocog__samples.jsonl
```

The summary JSON contains aggregate metrics, latency, the config hash, and `metrics.metadata.system_config`. The sample log contains per-example predictions and `intermediate` debugging information.
