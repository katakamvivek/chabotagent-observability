"""
LangGraph multi-agent system for document extraction.

Entry point: chatbot
  - General message  -> chatbot (LLM with MMS guardrails) -> END
  - PDF uploaded     -> chatbot -> supervisor (ReAct loop with ToolNode)
                        supervisor calls tools until done -> END
"""

import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Sequence

from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langfuse import get_client as get_langfuse_client

# OTel core
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import Resource

# OpenInference auto-instruments LangChain — adds LLM-specific attributes to spans
from openinference.instrumentation.langchain import LangChainInstrumentor

load_dotenv()

_HERE = Path(__file__).parent

# ── OTel CSV Exporter ──────────────────────────────────────────────────────────
#
# Each LangChain / LangGraph call produces OTel spans.  OpenInference enriches
# those spans with LLM-specific attributes (model, tokens, prompts, responses).
# This exporter flattens every span into one CSV row so it is easy to read.
#
# CSV columns:
#   timestamp       – when the span started (UTC ISO-8601)
#   trace_id        – groups all spans for one agent run
#   span_id         – unique ID for this span
#   parent_span_id  – links child → parent (blank = root span)
#   span_kind       – LLM | TOOL | CHAIN | AGENT (from openinference)
#   name            – human-readable span name (e.g. "ChatOpenAI", "extract_document")
#   duration_ms     – how long this span took
#   model           – LLM model name (blank for non-LLM spans)
#   prompt_tokens   – tokens sent to the LLM
#   completion_tokens – tokens returned by the LLM
#   total_tokens    – prompt + completion
#   input_preview   – first 300 chars of the prompt / tool input
#   output_preview  – first 300 chars of the response / tool output
#   status          – OK | ERROR
#   session_id      – groups spans across one user conversation
#   user_id         – identifies the end user

CSV_PATH = _HERE / "otel_traces.csv"

CSV_COLUMNS = [
    "timestamp", "trace_id", "span_id", "parent_span_id",
    "span_kind", "name", "duration_ms",
    "model", "prompt_tokens", "completion_tokens", "total_tokens",
    "cost_usd",
    "input_preview", "output_preview",
    "status", "session_id", "user_id",
]

# Pricing per 1 000 tokens (USD) — update when OpenAI changes rates.
# Format: model_substring -> (input_per_1k, output_per_1k)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":        (0.000150, 0.000600),
    "gpt-4o":             (0.002500, 0.010000),
    "gpt-4-turbo":        (0.010000, 0.030000),
    "gpt-4":              (0.030000, 0.060000),
    "gpt-3.5-turbo":      (0.000500, 0.001500),
    "claude-opus-4":      (0.015000, 0.075000),
    "claude-sonnet-4":    (0.003000, 0.015000),
    "claude-haiku-4":     (0.000800, 0.004000),
}


def _calculate_cost(model: str, prompt_tokens, completion_tokens) -> str:
    """
    Return cost in USD as a formatted string (e.g. '$0.001234').
    Matches model name by substring so 'gpt-4o-2024-11-20' hits 'gpt-4o'.
    Returns '' if model is unknown or tokens are missing.
    """
    if not model or prompt_tokens == "" or completion_tokens == "":
        return ""

    model_lower = model.lower()
    # match longest key first to avoid 'gpt-4' swallowing 'gpt-4o'
    for key in sorted(_MODEL_PRICING, key=len, reverse=True):
        if key in model_lower:
            input_rate, output_rate = _MODEL_PRICING[key]
            cost = (int(prompt_tokens) * input_rate / 1000) + \
                   (int(completion_tokens) * output_rate / 1000)
            return f"${cost:.6f}"

    return ""  # unknown model


def _ns_to_iso(nanoseconds: int) -> str:
    """Convert OTel nanosecond timestamp to a readable UTC string."""
    dt = datetime.fromtimestamp(nanoseconds / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # trim to milliseconds


def _extract_input(attrs: dict) -> str:
    """Pull the most useful input text from span attributes."""
    # LLM span: openinference stores messages as indexed attributes
    parts = []
    i = 0
    while True:
        role = attrs.get(f"llm.input_messages.{i}.message.role", "")
        content = attrs.get(f"llm.input_messages.{i}.message.content", "")
        if not role and not content:
            break
        if content:
            parts.append(f"[{role}] {content}")
        i += 1

    if parts:
        full = " | ".join(parts)
        return full[:300] + ("…" if len(full) > 300 else "")

    # Tool / chain span
    raw = attrs.get("input.value", attrs.get("input", ""))
    return str(raw)[:300] if raw else ""


def _extract_output(attrs: dict) -> str:
    """Pull the most useful output text from span attributes."""
    # LLM span
    parts = []
    i = 0
    while True:
        content = attrs.get(f"llm.output_messages.{i}.message.content", "")
        if not content:
            break
        parts.append(content)
        i += 1

    if parts:
        full = " | ".join(parts)
        return full[:300] + ("…" if len(full) > 300 else "")

    # Tool / chain span
    raw = attrs.get("output.value", attrs.get("output", ""))
    return str(raw)[:300] if raw else ""


class CSVSpanExporter(SpanExporter):
    """
    Writes one CSV row per OTel span.
    Creates the file with headers on first write.
    """

    def __init__(self, filepath: Path = CSV_PATH):
        self.filepath = filepath
        self._write_header_if_needed()

    def _write_header_if_needed(self):
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    def export(self, spans: Sequence) -> SpanExportResult:
        rows = []
        for span in spans:
            attrs = dict(span.attributes or {})

            # nanosecond timestamps → ms duration
            duration_ms = round((span.end_time - span.start_time) / 1_000_000, 2)

            parent_id = ""
            if span.parent and span.parent.span_id:
                parent_id = format(span.parent.span_id, "016x")

            model            = attrs.get("llm.model_name", "")
            prompt_tokens    = attrs.get("llm.token_count.prompt", "")
            completion_tokens = attrs.get("llm.token_count.completion", "")

            rows.append({
                "timestamp":          _ns_to_iso(span.start_time),
                "trace_id":           format(span.context.trace_id, "032x"),
                "span_id":            format(span.context.span_id, "016x"),
                "parent_span_id":     parent_id,
                "span_kind":          attrs.get("openinference.span.kind", ""),
                "name":               span.name,
                "duration_ms":        duration_ms,
                "model":              model,
                "prompt_tokens":      prompt_tokens,
                "completion_tokens":  completion_tokens,
                "total_tokens":       attrs.get("llm.token_count.total", ""),
                "cost_usd":           _calculate_cost(model, prompt_tokens, completion_tokens),
                "input_preview":      _extract_input(attrs),
                "output_preview":     _extract_output(attrs),
                "status":             span.status.status_code.name,
                "session_id":         attrs.get("session.id", ""),
                "user_id":            attrs.get("user.id", ""),
            })

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerows(rows)

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


# ── OTel provider setup ────────────────────────────────────────────────────────
#
# Resource tags appear in every span — useful for filtering in a real collector.
_otel_resource = Resource.create({"service.name": "mms-document-agent"})
_otel_provider = TracerProvider(resource=_otel_resource)
_otel_provider.add_span_processor(BatchSpanProcessor(CSVSpanExporter()))
trace.set_tracer_provider(_otel_provider)

# Auto-instrument LangChain: patches LLM calls, tool calls, and chains
# so they emit OTel spans with OpenInference attributes automatically.
LangChainInstrumentor().instrument()

_otel_tracer = trace.get_tracer("mms-agent")


# ── Langfuse observability ─────────────────────────────────────────────────────

_langfuse_handler = LangfuseCallbackHandler()


def build_langfuse_config(session_id: str = "", user_id: str = "", run_name: str = "") -> dict:
    """
    Build a LangChain run config that attaches Langfuse observability.
    session_id / user_id flow through LangChain metadata so Langfuse 4.x
    callback handler captures them via langfuse_session_id / langfuse_user_id keys.
    """
    metadata: dict = {}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if user_id:
        metadata["langfuse_user_id"] = user_id

    config: dict = {"callbacks": [_langfuse_handler]}
    if metadata:
        config["metadata"] = metadata
    if run_name:
        config["run_name"] = run_name
    return config


# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # append-only; ToolNode results are merged in correctly
    pdf_path: str                            # path to the uploaded PDF (empty if not provided)


# ── Tools (real implementations) ──────────────────────────────────────────────

@tool
def extract_document(pdf_path: str) -> str:
    """
    Extract a PDF file and convert it to structured JSON.
    Call this first whenever a PDF has been uploaded.
    Returns the document type and a summary of extracted content.
    """
    from main import pdf_to_json, mask_pii

    print(f"[tool:extract_document] Extracting: '{pdf_path}'")

    try:
        raw_json = pdf_to_json(pdf_path)
    except Exception as e:
        return f"ERROR: Could not extract PDF '{pdf_path}': {e}"

    pages = raw_json.get("pages", [])
    has_content = any(p.get("fields") or p.get("tables") or p.get("text") for p in pages)
    if not has_content:
        return (
            f"ERROR: '{Path(pdf_path).name}' was opened but no readable text found. "
            "It may be a scanned image. Please provide a text-based PDF."
        )

    masked_json = mask_pii(raw_json)

    out_dir = _HERE / "output"
    out_dir.mkdir(exist_ok=True)
    stem = Path(pdf_path).stem
    (out_dir / f"{stem}.json").write_text(
        json.dumps(raw_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"{stem}_masked.json").write_text(
        json.dumps(masked_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Detect document type from content
    all_text = str(raw_json).lower()
    if any(k in all_text for k in ["insurer", "insurance_claim", "police_report", "repair", "claim"]):
        doc_type = "car_claim"
    elif "net income" in all_text:
        doc_type = "payslip"
    else:
        doc_type = "unknown"

    print(f"[tool:extract_document] Done. Pages: {raw_json.get('total_pages', 0)}, type: {doc_type}")
    return json.dumps({
        "status":          "success",
        "document_type":   doc_type,
        "total_pages":     raw_json.get("total_pages", 0),
        "json_path":       str(out_dir / f"{stem}.json"),
        "pii_note":        "Raw PDF contained PII data. All PII fields have been masked before processing. The masked_data below is what was used.",
        "masked_data":     masked_json,
    })


@tool
def process_car_claim(pdf_path: str) -> str:
    """
    Submit a car insurance claim for an already-extracted document.
    Checks if the customer exists, creates the claim record, and sends a notification.
    Call this after extract_document returns document_type = 'car_claim'.
    """
    from main import customer_exists, add_car_claim, send_notification

    stem = Path(pdf_path).stem
    json_path   = str(_HERE / "output" / f"{stem}.json")
    masked_path = _HERE / "output" / f"{stem}_masked.json"

    print(f"[tool:process_car_claim] Checking customer for '{pdf_path}'")

    try:
        masked_json = json.loads(masked_path.read_text(encoding="utf-8"))
    except Exception:
        masked_json = {}

    found = customer_exists(json_path)
    if not found:
        return (
            "Customer not found in system. "
            "Please ensure your details are registered with MMS before submitting a claim."
        )

    claim_id = add_car_claim(json_path, masked_json, pdf_path)

    print(f"[tool:process_car_claim] Claim {claim_id} created")
    return (
        f"Car claim submitted successfully. "
        f"Claim reference: {claim_id}. "
        f"Our team will review it and be in touch shortly."
    )


@tool
def send_claim_notification(claim_id: str, pdf_name: str) -> str:
    """
    Send a push notification confirming a car claim was successfully submitted.
    Call this after process_car_claim returns a successful claim_id.
    """
    from main import send_notification

    print(f"[tool:send_claim_notification] Sending notification for {claim_id}")
    success = send_notification(
        title="New Car Claim Submitted",
        message=f"Claim {claim_id} has been successfully submitted for '{pdf_name}'.",
    )

    if success:
        return f"Push notification sent successfully for claim {claim_id}."
    return f"Push notification could not be delivered for claim {claim_id} — please check Pushover credentials."


@tool
def calculate_salary(pdf_path: str) -> str:
    """
    Calculate novated lease salary figures from an already-extracted payslip.
    Call this after extract_document returns document_type = 'payslip'.
    """
    from main import novated_lease

    stem = Path(pdf_path).stem
    json_path = str(_HERE / "output" / f"{stem}.json")

    print(f"[tool:calculate_salary] Calculating novated lease for '{pdf_path}'")

    try:
        result = novated_lease(json_path)
    except Exception as e:
        return f"ERROR: Could not calculate salary: {e}"

    print(f"[tool:calculate_salary] Result: {result}")
    return (
        f"Novated Lease Calculation:\n"
        f"  Current Net Income:  ${result['net_income']:,.2f}\n"
        f"  Salary After Lease:  ${result['new_salary_after_lease']:,.2f}\n"
        f"  Annual Tax Savings:  ${result['tax_savings']:,.2f}\n"
        f"Based on a monthly lease contribution of $500."
    )


SUPERVISOR_TOOLS = [extract_document, process_car_claim, send_claim_notification, calculate_salary]
supervisor_llm   = llm.bind_tools(SUPERVISOR_TOOLS)

SUPERVISOR_SYSTEM = """You are a document processing agent for McMillan Shakespeare Limited.

When a PDF is provided, follow these steps using the available tools:
1. Call extract_document with the pdf_path to extract and identify the document.
2. Read the returned document_type:
   - "car_claim"  -> call process_car_claim with the same pdf_path, then call
                     send_claim_notification with the claim_id and pdf file name from the result
   - "payslip"    -> call calculate_salary with the same pdf_path
   - "unknown"    -> do NOT call more tools; tell the user the document type is not supported
3. After all tools complete, return a clear and friendly response summarising the outcome.

Always use the tools in order. Do not skip extraction."""


# ── Chatbot system prompt (MMS guardrails) ─────────────────────────────────────

CHATBOT_SYSTEM = """You are a friendly and professional customer service assistant for
McMillan Shakespeare Limited (MMS) — Australia's leading provider of novated leasing,
salary packaging, and asset management services.

You may ONLY assist with:
1. Questions about McMillan Shakespeare Limited and its products or services
   (novated leasing, salary packaging, fleet management, asset management)
2. Submitting or enquiring about car insurance claims
3. Novated lease salary calculations and packaging benefits

IMPORTANT RULE — Car claim questions:
If the customer asks anything about how to submit, apply for, lodge, or start a car
insurance claim, you MUST respond ONLY with this exact message and nothing else:
"To submit your car insurance claim, simply upload your car repair invoice or insurance
claim document using the upload button below. Our system will automatically process it
and register your claim. Please ensure your details are registered with MMS before submitting."
Do NOT provide steps, instructions, or any other information about the claims process.

If a customer asks about anything outside these topics, your response MUST start with the
word "Sorry" and explain that you can only assist with MMS-related questions. Example:
"Sorry, I can only assist with McMillan Shakespeare Limited related questions such as
novated leasing, salary packaging, car claims, or salary calculations. Is there anything
along those lines I can help you with today?"

Never discuss competitors, politics, general finance advice unrelated to MMS,
or any topic unrelated to MMS services. Keep responses concise and professional."""


# ── Chatbot node (entry point) ─────────────────────────────────────────────────

def chatbot_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Entry point for all interactions.
    - PDF present -> hand off to supervisor
    - General message -> call LLM with MMS guardrails
    """
    pdf_path = state.get("pdf_path", "")
    messages = state.get("messages", [])

    # PDF present — hand off to supervisor
    if pdf_path and Path(pdf_path).exists():
        print(f"[chatbot] PDF detected: '{pdf_path}' -> supervisor")
        return state

    # General message — call LLM with MMS guardrails
    lc_messages = [SystemMessage(content=CHATBOT_SYSTEM)]
    for msg in messages:
        lc_messages.append(HumanMessage(content=msg) if isinstance(msg, str) else msg)

    response = llm.invoke(lc_messages, config=config)
    print(f"[chatbot] {response.content}")
    return {**state, "messages": list(messages) + [response]}


def route_chatbot(state: AgentState) -> str:
    if state.get("pdf_path") and Path(state["pdf_path"]).exists():
        return "supervisor"
    return END


# ── Supervisor node (ReAct loop) ───────────────────────────────────────────────

def supervisor_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    ReAct agent: calls tools to process the document, loops until done.
    On the first call, injects the PDF path into the conversation so the LLM
    knows which file to process.
    """
    messages = list(state.get("messages", []))
    pdf_path = state.get("pdf_path", "")

    # Inject PDF context on the first supervisor call (no prior ToolMessages)
    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    if not has_tool_results and pdf_path:
        user_msg = messages[-1].content if messages else "process this document"
        messages = messages + [
            HumanMessage(content=f"PDF file to process: {pdf_path}\nUser request: {user_msg}")
        ]

    lc_messages = [SystemMessage(content=SUPERVISOR_SYSTEM)] + messages

    print(f"[supervisor] Invoking LLM...")
    response = supervisor_llm.invoke(lc_messages, config=config)

    if response.tool_calls:
        print(f"[supervisor] Tool selected: {response.tool_calls[0]['name']}")
    else:
        print(f"[supervisor] Final answer ready")

    return {**state, "messages": list(state.get("messages", [])) + [response]}


def route_supervisor(state: AgentState) -> str:
    """Route to ToolNode if the LLM picked a tool, otherwise end."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph():
    tool_node = ToolNode(SUPERVISOR_TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("chatbot",    chatbot_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tools",      tool_node)

    graph.set_entry_point("chatbot")

    graph.add_conditional_edges("chatbot",    route_chatbot,    {"supervisor": "supervisor", END: END})
    graph.add_conditional_edges("supervisor", route_supervisor, {"tools": "tools", END: END})
    graph.add_edge("tools", "supervisor")   # always loop back after tool execution

    return graph.compile()


# ── Entry point ────────────────────────────────────────────────────────────────

def run(pdf_path: str = "", message: str = "",
        session_id: str = "", user_id: str = "") -> AgentState:
    """
    session_id: groups all traces for one customer conversation (auto-generated if omitted).
    user_id:    customer identifier shown in Langfuse dashboard and OTel CSV.
    """
    app = build_graph()

    if not session_id:
        session_id = f"session-{uuid.uuid4().hex[:8]}"

    initial_state: AgentState = {
        "messages": [HumanMessage(content=message)] if message else [],
        "pdf_path": pdf_path,
    }

    # ── Langfuse (cloud UI) ────────────────────────────────────────────────────
    langfuse_config = build_langfuse_config(
        session_id=session_id,
        user_id=user_id,
        run_name=f"mms-agent-{session_id}",
    )
    print(f"[langfuse] session_id={session_id} user_id={user_id or 'unknown'}")

    # ── LangSmith — merge session/user metadata into the same config ──────────
    # LangChain reads LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY from .env
    # automatically. We only need to add metadata for session/user grouping.
    langfuse_config.setdefault("metadata", {}).update({
        "session_id": session_id,
        "user_id":    user_id or "unknown",
    })
    langfuse_config["tags"] = [f"session:{session_id}"]
    print(f"[langsmith] project={os.getenv('LANGCHAIN_PROJECT', 'default')}")

    # ── OTel (local CSV) ───────────────────────────────────────────────────────
    # Open a root span for this entire agent run.
    # session.id and user.id are stamped here; child spans (LLM calls, tool
    # calls) inherit the trace_id automatically via OTel context propagation.
    with _otel_tracer.start_as_current_span("mms-agent-run") as root_span:
        root_span.set_attribute("session.id", session_id)
        root_span.set_attribute("user.id", user_id or "unknown")
        root_span.set_attribute("pdf_path", pdf_path or "none")

        result = app.invoke(initial_state, config=langfuse_config)

    # Flush both backends
    get_langfuse_client().flush()
    _otel_provider.force_flush()

    print(f"[otel] Spans written → {CSV_PATH}")
    return result


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else ""
    msg = sys.argv[2] if len(sys.argv) > 2 else ""
    result = run(pdf_path=pdf, message=msg)
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            print(f"\nAssistant: {m.content}")
            break
