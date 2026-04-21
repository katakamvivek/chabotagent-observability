"""
MMS Customer Service Chatbot — Gradio UI
"""

import uuid
from pathlib import Path

import gradio as gr
from langchain_core.messages import AIMessage

from agents import run


# ── Helper ─────────────────────────────────────────────────────────────────────

def get_last_reply(result: dict) -> str:
    """Extract the final assistant message from the agent result."""
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            return m.content
    return "Sorry, I was unable to process your request. Please try again."


# ── Chat handler ───────────────────────────────────────────────────────────────

def chat(message: str, pdf_file, history: list, session_id: str):
    """
    Called on every user submission.
    - message:    text typed by the user (may be empty if only a PDF was uploaded)
    - pdf_file:   uploaded file path string from gr.File, or None
    - history:    current chat history as list of {"role": ..., "content": ...}
    - session_id: persists for the lifetime of the browser session
    """
    if not session_id:
        session_id = f"session-{uuid.uuid4().hex[:8]}"

    pdf_path = pdf_file if pdf_file else ""

    # Require at least a message or a PDF
    if not message.strip() and not pdf_path:
        return history, "", None, session_id

    # Display label when only a PDF is uploaded with no text
    display_msg = message.strip() or f"[Uploaded: {Path(pdf_path).name}]"

    result = run(pdf_path=pdf_path, message=message.strip(), session_id=session_id)
    reply  = get_last_reply(result)

    history = history + [
        {"role": "user",      "content": display_msg},
        {"role": "assistant", "content": reply},
    ]

    # Clear the text box and PDF upload after processing
    return history, "", None, session_id


# ── UI ─────────────────────────────────────────────────────────────────────────

CSS = """
.header { text-align: center; padding: 16px 0 8px; }
.header h1 { font-size: 1.8rem; font-weight: 700; color: #1a3c6e; margin: 0; }
.header p  { color: #555; margin: 4px 0 0; font-size: 0.95rem; }
footer { display: none !important; }
"""

with gr.Blocks(title="MMS Customer Assistant") as demo:

    # ── Session state (unique per browser tab) ─────────────────────────────────
    session_id = gr.State(value=lambda: f"session-{uuid.uuid4().hex[:8]}")

    # ── Header ─────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div class="header">
            <h1>McMillan Shakespeare Limited</h1>
            <p>Customer Service Assistant — Novated Leasing &amp; Car Claims</p>
        </div>
    """)

    # ── Chat area ──────────────────────────────────────────────────────────────
    chatbot = gr.Chatbot(
        label="MMS Assistant",
        height=460,
        avatar_images=(None, "https://ui-avatars.com/api/?name=MMS&background=1a3c6e&color=fff&size=64"),
        placeholder="Ask me about novated leasing, salary packaging, or upload a PDF to submit a claim.",
    )

    # ── Input row ──────────────────────────────────────────────────────────────
    with gr.Row():
        pdf_upload = gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
            scale=1,
        )
        msg = gr.Textbox(
            label="Your message",
            placeholder="e.g. What is novated leasing? or upload a payslip / car claim PDF above.",
            lines=2,
            scale=4,
        )

    with gr.Row():
        clear_btn  = gr.Button("Clear chat", variant="secondary", scale=1)
        submit_btn = gr.Button("Send",       variant="primary",   scale=1)

    # ── Suggested prompts ──────────────────────────────────────────────────────
    gr.Examples(
        examples=[
            ["What is novated leasing?",           None],
            ["How do I apply for a car insurance claim?", None],
            ["What salary packaging options does MMS offer?", None],
        ],
        inputs=[msg, pdf_upload],
        label="Quick questions",
    )

    # ── Event wiring ───────────────────────────────────────────────────────────
    submit_inputs  = [msg, pdf_upload, chatbot, session_id]
    submit_outputs = [chatbot, msg, pdf_upload, session_id]

    submit_btn.click(chat, inputs=submit_inputs, outputs=submit_outputs)
    msg.submit(   chat, inputs=submit_inputs, outputs=submit_outputs)
    clear_btn.click(lambda: ([], "", None), outputs=[chatbot, msg, pdf_upload])


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
