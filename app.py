import os
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
from pipeline import DocumentIntelligencePipeline
import tempfile
import yaml
        
load_dotenv()

pipeline = None
current_document_id = None
current_document_name = None

def initialize_pipeline():
    global pipeline
    if pipeline is None:
        config_path = os.environ.get("PIPELINE_CONFIG_PATH")
        pipeline = DocumentIntelligencePipeline(config_path=config_path)
        os.unlink(config_path)
    return pipeline

def process_document(file):
    global current_document_id, current_document_name
    
    if file is None:
        return "Please upload a document first."
    try:
        pipe = initialize_pipeline()
        file_path = file.name
        current_document_name = Path(file_path).name
        document = pipe.process_document(file_path)
        current_document_id = document.document_id
        status = f"""✅ Document processed successfully!
                            **{current_document_name}**
                            - Pages: {document.num_pages}
                            - Layout chunks: {len(document.chunks)}
                            - Status: Ready for questions
                            """

        
        agentic_meta = document.metadata.get('agentic')
        skip_reason = document.metadata.get('agentic_skip_reason')
        if agentic_meta and agentic_meta.get('decision'):
            decision = agentic_meta['decision']
            winner = decision.get('winner', 'ocr').upper()
            confidence = float(decision.get('confidence') or 0.0)
            reason = decision.get('reason', 'n/a')
            status += f"\n **Agentic Winner:** {winner} (confidence {confidence:.2f})\nReason: {reason}\n"
            if decision.get('notes'):
                status += f"Notes: {decision['notes']}\n"
        elif skip_reason:
            status += f"\n Agentic stage skipped: {skip_reason}\n"
        return status, ""
    
    except Exception as e:
        return f"Error processing document: {str(e)}", ""

def answer_question(question):
    global current_document_id, current_document_name
    
    if current_document_id is None:
        return "Please upload and process a document first."
    
    if not question or not question.strip():
        return "Please enter a question."
    
    try:
        answer = pipeline.ask_question(
            question,
            n_results=10
        )
        return answer
    
    except Exception as e:
        return f"Error: {str(e)}"

def clear_all():
    global current_document_id, current_document_name
    current_document_id = None
    current_document_name = None
    return None, "Ready to process a new document.", "", ""

with gr.Blocks(title="Document Q&A", theme=gr.themes.Soft()) as demo:
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ Upload Document")
            file_input = gr.File(
                label="Upload PDF or Image",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
            )
            process_btn = gr.Button("📄 Process Document", variant="primary", size="lg")
            
            status_output = gr.Markdown(
                value="Ready to process a document.",
                label="Status"
            )
            
            gr.Markdown("---")
            clear_btn = gr.Button("🗑️ Clear & Start Over", variant="secondary")
        
        with gr.Column(scale=2):
            # Question answering section
            gr.Markdown("### 2️⃣ Ask Questions")
            
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about the document...",
                lines=2
            )
            
            ask_btn = gr.Button("💡 Get Answer", variant="primary", size="lg")
            
            answer_output = gr.Textbox(
                label="Answer",
                lines=10,
            )
    process_btn.click(
        fn=process_document,
        inputs=[file_input],
        outputs=[status_output, answer_output]
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output]
    )
    
    question_input.submit(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output]
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[file_input, status_output, question_input, answer_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
