# VisualCue

## Run The Demo

Start an OpenAI-compatible VLM server first, for example LM Studio at `http://localhost:1234/v1`, and load the model you want to use.

Then run the interactive Streamlit demo:

```bash
streamlit run app/streamlit_app.py
```

The app reuses the same `SequentialPipeline` and `AgenticPipeline` as the evaluation harness. Falcon is cached as a warm Streamlit resource, so switching systems, prompt style, `max_steps`, reasoning, or VLM connection settings does not reload the segmentation model.
