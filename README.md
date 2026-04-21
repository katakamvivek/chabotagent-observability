# MMS Customer Service Chatbot — with Observability

A LangGraph-powered document extraction agent for MMS with full observability using **Langfuse**, **LangSmith**, and **OpenTelemetry (OTel)**.

---

## What the App Does

Users interact with a Gradio chatbot to:
- Ask questions about MMS services (novated leasing, salary packaging)
- Upload a **car repair invoice PDF** → agent extracts it, checks the customer, creates a claim, and sends a push notification
- Upload a **payslip PDF** → agent calculates novated lease salary figures

```
User (Gradio UI)
       │
       ▼
   chatbot node  ──── general questions ────► END
       │
       │ PDF detected
       ▼
  supervisor node (ReAct loop)
       ├── extract_document()       ← pdfplumber + PII masking
       ├── process_car_claim()      ← customer lookup + CSV record
       ├── send_claim_notification() ← Pushover push notification
       └── calculate_salary()       ← novated lease figures
```

---

## Project Structure

```
document_extraction/
├── app.py              # Gradio UI
├── agents.py           # LangGraph graph + all observability wiring
├── main.py             # PDF extraction, PII masking, customer lookup, claim creation
├── pii_config.json     # PII masking rules (regex patterns)
├── requirements.txt    # Python dependencies
├── customer_data/
│   ├── customers.csv   # Registered customer records
│   └── car_claim.csv   # Created automatically on first claim
├── output/             # Extracted JSON files (git-ignored)
└── otel_traces.csv     # OTel span output (git-ignored)
```

---

## Setup

### 1. Create and activate environment

```bash
conda create -n document_agent python=3.11
conda activate document_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install opentelemetry-sdk openinference-instrumentation-langchain langsmith
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=mms-document-agent

# Pushover (optional — for push notifications)
PUSHOVER_USER=...
PUSHOVER_TOKEN=...
```

### 4. Run the app

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

---

## Observability

This project implements **three observability tools simultaneously** — all running on every agent invocation.

---

### 1. Langfuse

**What it is:** Open-source LLM observability platform with a rich UI for sessions, users, cost, and prompt inspection.

**How it works:** Langfuse uses a LangChain **callback handler**. It is explicitly passed inside `config["callbacks"]` when invoking the graph. LangChain calls it at each step — before an LLM call, after a tool returns, when a chain finishes.

```python
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

_langfuse_handler = LangfuseCallbackHandler()

config = {
    "callbacks": [_langfuse_handler],       # ← Langfuse plugs in here
    "metadata": {
        "langfuse_session_id": session_id,
        "langfuse_user_id":    user_id,
    }
}

app.invoke(initial_state, config=config)
get_langfuse_client().flush()               # ensure all data is sent
```

**What it captures:**
- Every LLM call with full prompt and response
- Token counts and cost per generation (automatic, uses built-in pricing tables)
- Tool calls with inputs and outputs
- Session and user grouping

**Where to view:** [cloud.langfuse.com](https://cloud.langfuse.com) → your project

---

### 2. LangSmith

**What it is:** The official LangChain observability platform, built directly into the LangChain SDK. Requires the least code of all three tools.

**How it works:** LangSmith is baked into LangChain itself. Setting `LANGCHAIN_TRACING_V2=true` in `.env` is enough — LangChain automatically ships every run to LangSmith in the background. No handler or callback class is needed.

```python
# No code needed — just .env:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=lsv2_...
# LANGCHAIN_PROJECT=mms-document-agent

# Only addition: merge session/user into metadata for filtering in the UI
config.setdefault("metadata", {}).update({
    "session_id": session_id,
    "user_id":    user_id,
})
config["tags"] = [f"session:{session_id}"]
```

**What it captures:**
- Full LangGraph run tree (each node as a child span)
- LLM calls with prompts, responses, token counts, and cost (automatic)
- Tool calls with inputs and outputs
- Latency per node

**Where to view:** [smith.langchain.com](https://smith.langchain.com) → Projects → `mms-document-agent`

---

### 3. OpenTelemetry (OTel)

**What it is:** A vendor-neutral, open-source observability standard maintained by the CNCF. Used industry-wide for microservices, Kubernetes, and databases. LLM-specific attributes are added via the **OpenInference** convention.

**How it works:** OTel uses monkey-patching. `LangChainInstrumentor().instrument()` wraps LangChain's internals at import time. Every subsequent LangChain call automatically emits an OTel span — no config changes, no callbacks.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor

# 1. Set up provider → processor → exporter chain
_otel_provider = TracerProvider(resource=Resource.create({"service.name": "mms-document-agent"}))
_otel_provider.add_span_processor(BatchSpanProcessor(CSVSpanExporter()))
trace.set_tracer_provider(_otel_provider)

# 2. Patch LangChain once at import time — no further changes needed
LangChainInstrumentor().instrument()

# 3. Open a root span per run to group all child spans under one trace_id
with _otel_tracer.start_as_current_span("mms-agent-run") as root_span:
    root_span.set_attribute("session.id", session_id)
    root_span.set_attribute("user.id", user_id)
    result = app.invoke(initial_state, config=config)

_otel_provider.force_flush()  # flush buffered spans to CSV
```

**Custom CSV Exporter:**

Rather than sending spans to Jaeger or a cloud collector, this project includes a custom `CSVSpanExporter` that writes one row per span to `otel_traces.csv`. This makes it easy to inspect traces in Excel or any CSV viewer.

```python
class CSVSpanExporter(SpanExporter):
    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            # extract model, tokens, prompts, tool inputs from OpenInference attributes
            # calculate cost manually using a pricing dict
            # write one CSV row
```

**CSV columns:**

| Column | Description |
|---|---|
| `timestamp` | When the span started (UTC) |
| `trace_id` | Groups all spans for one agent run |
| `span_id` | Unique ID for this span |
| `parent_span_id` | Links child to parent span |
| `span_kind` | `LLM` / `TOOL` / `CHAIN` / `AGENT` |
| `name` | e.g. `ChatOpenAI`, `extract_document` |
| `duration_ms` | How long this span took |
| `model` | LLM model name |
| `prompt_tokens` | Tokens sent to the LLM |
| `completion_tokens` | Tokens returned by the LLM |
| `total_tokens` | prompt + completion |
| `cost_usd` | Calculated from a local pricing table |
| `input_preview` | First 300 chars of the prompt |
| `output_preview` | First 300 chars of the response |
| `status` | `OK` or `ERROR` |
| `session_id` | Groups spans across one user conversation |
| `user_id` | Identifies the end user |

**Where to view:** `E:\document_extraction\otel_traces.csv` — open in Excel, Google Sheets, or any CSV viewer.

---

### How all three run together

```
app.invoke(initial_state, config=config)
         │
         ├── Langfuse   → reads config["callbacks"] → Langfuse cloud UI
         │
         ├── LangSmith  → reads LANGCHAIN_TRACING_V2 env var → LangSmith cloud UI
         │
         └── OTel       → LangChainInstrumentor patches at import → otel_traces.csv
```

A single `app.invoke()` call feeds all three tools simultaneously.

---

### Comparison

| | Langfuse | LangSmith | OTel (CSV) |
|---|---|---|---|
| **Integration** | Explicit callback | Env vars only | Monkey-patch at import |
| **Token tracking** | Automatic | Automatic | Automatic (OpenInference) |
| **Cost tracking** | Automatic | Automatic | Manual pricing dict |
| **Storage** | Cloud or self-hosted | Cloud only | Local CSV file |
| **UI** | Langfuse dashboard | LangSmith dashboard | Excel / CSV viewer |
| **Vendor lock-in** | Low (open source) | Medium (LangChain) | None |
| **Free tier** | Generous | 5,000 traces/month | Fully free |
| **Best for** | Privacy-focused, self-hosted teams | LangChain projects, zero-effort setup | Local dev, enterprise OTel pipelines |

---

## Key Features

- **PII Masking** — sensitive fields (name, email, phone) are redacted before any data is stored or sent to the LLM
- **Customer Lookup** — checks if the document owner exists in `customers.csv` before creating a claim
- **Car Claim Creation** — appends a new row to `car_claim.csv` with claim ID, status, and masked document data
- **Novated Lease Calculation** — extracts net income from payslips and calculates salary-after-lease and annual tax savings
- **Push Notifications** — sends a Pushover notification on claim submission (requires Pushover credentials)

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `LANGFUSE_PUBLIC_KEY` | Yes | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Yes | Langfuse secret key |
| `LANGFUSE_HOST` | Yes | Langfuse host URL |
| `LANGCHAIN_TRACING_V2` | Yes | Set to `true` to enable LangSmith |
| `LANGCHAIN_API_KEY` | Yes | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name (default: `default`) |
| `PUSHOVER_USER` | No | Pushover user key for notifications |
| `PUSHOVER_TOKEN` | No | Pushover app token for notifications |
