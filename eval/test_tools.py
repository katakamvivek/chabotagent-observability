"""
Individual tool evaluation — extraction, PII masking, car claim, salary calc.

Covers:
  - FaithfulnessMetric   : final response is grounded in tool outputs (no hallucination)
  - HallucinationMetric  : salary figures and claim IDs match tool output
  - Direct assertion     : known PII values absent from masked structured fields
  - GEval (custom)       : document type detection accuracy
  - GEval (custom)       : PII masking confirmation note present
"""

import json
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    FaithfulnessMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from dataset import run_agent, CAR_CLAIM_PDF, PAYSLIP_PDF


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def car_claim_run():
    response, tools = run_agent(pdf_path=CAR_CLAIM_PDF, message="submit my claim")
    return response, tools

@pytest.fixture(scope="module")
def payslip_run():
    response, tools = run_agent(pdf_path=PAYSLIP_PDF, message="calculate my novated lease")
    return response, tools


# ── extract_document: document type detection ──────────────────────────────────

def test_car_claim_document_type_detected(car_claim_run):
    """extract_document must identify car_repair_claim_invoice.pdf as 'car_claim'."""
    _, tools = car_claim_run
    extract_tool = next((t for t in tools if t.name == "extract_document"), None)
    assert extract_tool is not None, "extract_document was not called"

    output = json.loads(extract_tool.output)
    assert output.get("document_type") == "car_claim", (
        f"Expected 'car_claim', got '{output.get('document_type')}'"
    )


def test_payslip_document_type_detected(payslip_run):
    """extract_document must identify oct_2022.pdf as 'payslip'."""
    _, tools = payslip_run
    extract_tool = next((t for t in tools if t.name == "extract_document"), None)
    assert extract_tool is not None, "extract_document was not called"

    output = json.loads(extract_tool.output)
    assert output.get("document_type") == "payslip", (
        f"Expected 'payslip', got '{output.get('document_type')}'"
    )


# ── extract_document: PII masking ─────────────────────────────────────────────

def test_pii_note_present_in_extract_output(car_claim_run):
    """extract_document output must include a pii_note confirming masking was applied."""
    _, tools = car_claim_run
    extract_tool = next((t for t in tools if t.name == "extract_document"), None)
    assert extract_tool is not None

    output = json.loads(extract_tool.output)
    assert "pii_note" in output, "pii_note missing from extract_document output"
    assert "masked" in output["pii_note"].lower(), "pii_note does not mention masking"


def test_masked_data_has_no_pii_leakage(car_claim_run):
    """
    PII masking must replace the claimant's personal identifiers in structured
    fields and table cells.  Known PII from car_repair_claim_invoice.pdf:
      - email : arjun.v.patel@gmail.com
      - phone : 0412 738 965
      - name  : arjun vikram patel (case-insensitive)
    """
    _, tools = car_claim_run
    extract_tool = next((t for t in tools if t.name == "extract_document"), None)
    assert extract_tool is not None

    output = json.loads(extract_tool.output)
    masked_data = output.get("masked_data", {})

    # Collect only structured fields and table cells (not raw text lines)
    structured_values = {}
    for page in masked_data.get("pages", []):
        structured_values.update(page.get("fields", {}))
        for table in page.get("tables", []):
            for row in table:
                structured_values.update(row)

    masked_fields_str = json.dumps(structured_values).lower()

    known_pii = [
        "arjun.v.patel@gmail.com",
        "0412 738 965",
        "arjun vikram patel",
    ]
    for pii in known_pii:
        assert pii.lower() not in masked_fields_str, (
            f"PII not masked in structured fields: '{pii}'"
        )


# ── process_car_claim: faithfulness ───────────────────────────────────────────

def test_car_claim_response_is_faithful(car_claim_run):
    """
    Final car claim response must be grounded in tool outputs —
    claim ID and success message must come from process_car_claim, not invented.
    """
    response, tools = car_claim_run
    tool_outputs = [t.output for t in tools if t.output]

    test_case = LLMTestCase(
        input="Submit my car insurance claim",
        actual_output=response,
        retrieval_context=tool_outputs,
    )
    assert_test(test_case, [FaithfulnessMetric(threshold=0.7)])


def test_car_claim_response_no_hallucination(car_claim_run):
    """Car claim response must not invent claim IDs or facts not in tool outputs."""
    response, tools = car_claim_run
    tool_outputs = [t.output for t in tools if t.output]

    test_case = LLMTestCase(
        input="Submit my car insurance claim",
        actual_output=response,
        context=tool_outputs,
    )
    assert_test(test_case, [HallucinationMetric(threshold=0.5)])


# ── calculate_salary: figures accuracy ────────────────────────────────────────

def test_salary_response_contains_correct_figures(payslip_run):
    """Salary calculation response must contain the key figures from calculate_salary tool."""
    response, tools = payslip_run
    salary_tool = next((t for t in tools if t.name == "calculate_salary"), None)
    assert salary_tool is not None, "calculate_salary was not called"

    # Parse figures directly from tool output and verify they appear in the response
    tool_output = salary_tool.output
    tool_data = json.loads(tool_output) if tool_output.strip().startswith("{") else {}

    if tool_data:
        # Structured JSON output — check each numeric figure appears in the response
        for key in ("net_income", "new_salary_after_lease", "tax_savings"):
            if key in tool_data:
                value = str(tool_data[key]).rstrip("0").rstrip(".")
                assert value in response, (
                    f"Expected {key} value '{value}' not found in response"
                )
    else:
        # Tool returned a text summary — verify at least one dollar amount carries over
        import re as _re
        amounts = _re.findall(r"\$[\d,]+(?:\.\d+)?", tool_output)
        assert any(amt in response for amt in amounts), (
            f"No salary amounts from tool output found in response.\n"
            f"Tool output: {tool_output}\nResponse: {response}"
        )


def test_salary_response_no_hallucination(payslip_run):
    """Salary figures in response must match tool output — no invented numbers."""
    response, tools = payslip_run
    salary_tool = next((t for t in tools if t.name == "calculate_salary"), None)
    assert salary_tool is not None

    test_case = LLMTestCase(
        input="Calculate my novated lease salary",
        actual_output=response,
        context=[salary_tool.output],
    )
    assert_test(test_case, [HallucinationMetric(threshold=0.5)])


# ── send_claim_notification: output check ─────────────────────────────────────

def test_notification_tool_called_after_claim(car_claim_run):
    """send_claim_notification must be called after process_car_claim succeeds."""
    _, tools = car_claim_run
    tool_names = [t.name for t in tools]
    assert "send_claim_notification" in tool_names, "send_claim_notification was not called"

    # Notification must come after process_car_claim
    claim_idx  = next(i for i, t in enumerate(tools) if t.name == "process_car_claim")
    notify_idx = next(i for i, t in enumerate(tools) if t.name == "send_claim_notification")
    assert notify_idx > claim_idx, "send_claim_notification called before process_car_claim"


def test_notification_success_in_output(car_claim_run):
    """send_claim_notification tool output must confirm notification was sent."""
    _, tools = car_claim_run
    notify_tool = next((t for t in tools if t.name == "send_claim_notification"), None)
    assert notify_tool is not None

    test_case = LLMTestCase(
        input="Send push notification for submitted claim",
        actual_output=notify_tool.output,
    )
    metric = GEval(
        name="Notification Success",
        criteria="The output must confirm that the push notification was sent successfully.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )
    assert_test(test_case, [metric])
