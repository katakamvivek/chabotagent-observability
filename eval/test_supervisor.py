"""
Supervisor agent evaluation — tool routing and task completion.

Covers:
  - ToolCorrectnessMetric : right tools called in right order
  - TaskCompletionMetric  : end-to-end task completed successfully
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric, TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall

from dataset import (
    run_agent,
    CAR_CLAIM_PDF,
    PAYSLIP_PDF,
    BAD_PDF,
    EXPECTED_CAR_CLAIM_TOOLS,
    EXPECTED_PAYSLIP_TOOLS,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def car_claim_run():
    response, tools = run_agent(pdf_path=CAR_CLAIM_PDF, message="submit my claim")
    return response, tools

@pytest.fixture(scope="module")
def payslip_run():
    response, tools = run_agent(pdf_path=PAYSLIP_PDF, message="calculate my novated lease")
    return response, tools

@pytest.fixture(scope="module")
def bad_pdf_run():
    response, tools = run_agent(pdf_path=BAD_PDF, message="process this")
    return response, tools


# ── Tool correctness tests ─────────────────────────────────────────────────────

def test_car_claim_tool_sequence(car_claim_run):
    """
    Car claim flow must call tools in order:
    extract_document → process_car_claim → send_claim_notification
    """
    response, tools = car_claim_run
    test_case = LLMTestCase(
        input="Submit car insurance claim",
        actual_output=response,
        tools_called=tools,
        expected_tools=EXPECTED_CAR_CLAIM_TOOLS,
    )
    assert_test(test_case, [
        ToolCorrectnessMetric(
            threshold=0.7,
            should_consider_ordering=True,
            include_reason=True,
        )
    ])


def test_payslip_tool_sequence(payslip_run):
    """
    Payslip flow must call tools in order:
    extract_document → calculate_salary
    """
    response, tools = payslip_run
    test_case = LLMTestCase(
        input="Calculate my novated lease salary",
        actual_output=response,
        tools_called=tools,
        expected_tools=EXPECTED_PAYSLIP_TOOLS,
    )
    assert_test(test_case, [
        ToolCorrectnessMetric(
            threshold=0.7,
            should_consider_ordering=True,
            include_reason=True,
        )
    ])


def test_bad_pdf_no_unnecessary_tools(bad_pdf_run):
    """
    For an unreadable PDF, only extract_document should be called —
    no claim or salary tools should be invoked.
    """
    response, tools = bad_pdf_run
    tool_names = [t.name for t in tools]
    assert "process_car_claim"     not in tool_names, "process_car_claim called on bad PDF"
    assert "calculate_salary"      not in tool_names, "calculate_salary called on bad PDF"
    assert "send_claim_notification" not in tool_names, "send_claim_notification called on bad PDF"


# ── Task completion tests ──────────────────────────────────────────────────────

def test_car_claim_task_completed(car_claim_run):
    """Car claim submission task must be completed end-to-end."""
    response, tools = car_claim_run
    test_case = LLMTestCase(
        input="Submit my car insurance claim",
        actual_output=response,
        tools_called=tools,
    )
    assert_test(test_case, [
        TaskCompletionMetric(
            task="Submit a car insurance claim by extracting the PDF, verifying the customer, creating the claim record, and sending a push notification.",
            threshold=0.7,
        )
    ])


def test_payslip_task_completed(payslip_run):
    """Novated lease salary calculation task must be completed end-to-end."""
    response, tools = payslip_run
    test_case = LLMTestCase(
        input="Calculate my novated lease salary",
        actual_output=response,
        tools_called=tools,
    )
    assert_test(test_case, [
        TaskCompletionMetric(
            task="Calculate the novated lease salary figures (net income, salary after lease, annual tax savings) from the uploaded payslip.",
            threshold=0.7,
        )
    ])


def test_bad_pdf_task_gracefully_handled(bad_pdf_run):
    """Agent must return a user-friendly error for unreadable PDFs — not crash."""
    response, _ = bad_pdf_run
    assert response, "No response returned for bad PDF"
    assert len(response) > 10, "Response too short — likely empty or silent failure"
