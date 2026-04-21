"""
Shared helpers and golden test data for all DeepEval test files.
"""

import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.messages import AIMessage, ToolMessage

from agents import run


# ── Agent runner ───────────────────────────────────────────────────────────────

def run_agent(pdf_path: str = "", message: str = "") -> tuple[str, list[ToolCall]]:
    """
    Run the MMS agent and return:
      - final_response: last assistant message shown to user
      - tools_called:   list of DeepEval ToolCall objects with name, args, output
    """
    result = run(pdf_path=pdf_path, message=message)
    messages = result.get("messages", [])

    # Final assistant response (last AIMessage without tool_calls)
    final_response = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            final_response = m.content
            break

    # Collect tool calls + their outputs from the message history
    tools_called: list[ToolCall] = []
    for i, m in enumerate(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                # Find the ToolMessage that corresponds to this tool call
                tool_output = ""
                for j in range(i + 1, len(messages)):
                    if isinstance(messages[j], ToolMessage) and \
                       messages[j].tool_call_id == tc["id"]:
                        tool_output = messages[j].content
                        break
                tools_called.append(ToolCall(
                    name=tc["name"],
                    input_parameters=tc.get("args", {}),
                    output=tool_output,
                ))

    return final_response, tools_called


# ── PDF paths ──────────────────────────────────────────────────────────────────

ROOT           = Path(__file__).parent.parent
CAR_CLAIM_PDF  = str(ROOT / "car_repair_claim_invoice.pdf")
PAYSLIP_PDF    = str(ROOT / "oct_2022.pdf")
BAD_PDF        = str(ROOT / "bad_file.pdf")


# ── Golden expected tool sequences ────────────────────────────────────────────

EXPECTED_CAR_CLAIM_TOOLS = [
    ToolCall(name="extract_document"),
    ToolCall(name="process_car_claim"),
    ToolCall(name="send_claim_notification"),
]

EXPECTED_PAYSLIP_TOOLS = [
    ToolCall(name="extract_document"),
    ToolCall(name="calculate_salary"),
]
