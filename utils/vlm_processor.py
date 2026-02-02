import os
import base64
import json
from typing import List, Dict, Optional, Any
import time
from openai import OpenAI, Timeout
from PIL import Image
import io
from .models import LayoutChunk, LayoutType

class VLMProcessor:
    def __init__(self,
                 api_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None):

        self.api_endpoint = api_endpoint or os.environ.get("VLLM_API_ENDPOINT")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY")
        self.model_name = model_name or os.environ.get("VLM_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
        
        if not self.api_endpoint or not self.api_key:
            raise ValueError("VLLM_API_ENDPOINT and VLLM_API_KEY must be provided or set as environment variables")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.api_endpoint,
            api_key=self.api_key,
            timeout=Timeout(300.0),
            max_retries=0
        )
        
        print(f"✓ VLM Processor initialized with model: {self.model_name}")
    
    def analyze_chunk(self, image_path: str, chunk: LayoutChunk) -> Dict[str, Any]:

        if not chunk.bbox:
            return {"error": "No bounding box available"}
        
        # Crop the chunk from the image
        image = Image.open(image_path)
        box = (int(chunk.bbox.x1), int(chunk.bbox.y1), 
               int(chunk.bbox.x2), int(chunk.bbox.y2))
        cropped = image.crop(box)
        
        base64_image = self._image_to_base64(cropped)
        prompt = self._create_chunk_prompt(chunk.layout_type)

        result = self._call_vlm(base64_image, prompt)
        
        return result
    
    def analyze_full_document(self, image_path: str) -> List[Dict[str, Any]]:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        prompt = """
            You are an OCR and document understanding model.
            Given an image of a document page, analyze its structure and return a structured JSON representation of the content in natural reading order.
 
            For each detected element, include:
            - "type": e.g., text, heading, table, image, caption, list, chart, checkbox, signature, handwritten_input
            - "content": textual or structured data depending on the element type
 
            If the document is an **application form** or any structured form, detect and extract:
            - Key-value pairs where printed labels (keys) correspond to filled or handwritten values (values)
            - Checkboxes, marking them as `"checked": true` or `"checked": false`
            - Handwritten inputs (names, dates, comments, etc.)
            - Signatures as `"type": "signature"`
            - Unfilled fields as `"value": null`
 
            Follow these format rules for each element:
            - **Text** → return the text in markdown format inside `"content"`
            - **Heading** → return HTML in `"content"`
            - **List** → return markdown in `"content"`
            - **Image** → return image description in `"content"`
            - **Chart** → return chart data in HTML table format in `"content"`
            - **Table** → must return table data in HTML table format in `"content"`
                
            Sample output format:
            ```json
            [
                {
                    "type": "heading" | "text" | "table" | "image" | "caption" | "list" | "chart" | "checkbox",
                    "content": "textual content in markdown format" | "table data in HTML format" | "image description" | "chart data in HTML" | "checkbox label: checked/unchecked"
                },
                ...
            ]
            ```
        """
        
        result = self._call_vlm(base64_image, prompt)
        try:
            if isinstance(result.get('content'), str):
                # Try to extract JSON from markdown code blocks
                content = result['content']
                if '```json' in content:
                    json_str = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    json_str = content.split('```')[1].split('```')[0].strip()
                else:
                    json_str = content
                
                elements = json.loads(json_str)
                return elements
            return []
        except Exception as e:
            print(f"Error parsing VLM response: {e}")
            return [{"type": "error", "content": str(result)}]
    
    def enhance_chunk_text(self, image_path: str, chunk: LayoutChunk, ocr_text: str) -> str:
        if not chunk.bbox:
            return ocr_text
        
        # Crop the chunk
        image = Image.open(image_path)
        box = (int(chunk.bbox.x1), int(chunk.bbox.y1), 
               int(chunk.bbox.x2), int(chunk.bbox.y2))
        cropped = image.crop(box)
        
        base64_image = self._image_to_base64(cropped)
        
        prompt = f"""
        You are an expert OCR corrector. 
        
        The traditional OCR extracted this text:
        "{ocr_text}"
        
        Please analyze the image and:
        1. If the OCR is accurate, return it as-is
        2. If there are errors, return the corrected text
        3. If it's handwritten or unclear, provide your best interpretation
        
        Return ONLY the text content, no explanations.
        """
        
        result = self._call_vlm(base64_image, prompt)
        enhanced_text = result.get('content', ocr_text)
        
        # Clean up the response
        if isinstance(enhanced_text, str):
            enhanced_text = enhanced_text.strip()
            # Remove markdown if present
            if enhanced_text.startswith('```') and enhanced_text.endswith('```'):
                enhanced_text = enhanced_text.split('```')[1].strip()
        
        return enhanced_text if enhanced_text else ocr_text
    
    def get_semantic_summary(self, chunks: List[LayoutChunk]) -> str:
        sorted_chunks = sorted(chunks, key=lambda x: (x.page_number, x.reading_order))
        doc_text = []
        for chunk in sorted_chunks:
            if chunk.text.strip():
                doc_text.append(f"[{chunk.layout_type.value}] {chunk.text}")
        
        combined_text = "\n\n".join(doc_text[:50]) 
        
        prompt = f"""
        Analyze this document and provide a concise summary covering:
        1. Document type and purpose
        2. Key information and entities
        3. Main topics or sections
        
        Document content:
        {combined_text}
        
        Provide a brief 2-3 sentence summary.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response.choices[0].message.content
            return summary.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""
    
    def _call_vlm(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        try:
            data_url = f"data:image/jpeg;base64,{base64_image}"
            
            payload = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
            
            start_ts = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=payload,
            )
            
            elapsed = time.perf_counter() - start_ts
            
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "model": self.model_name,
                "processing_time": elapsed
            }
            
        except Exception as e:
            print(f"VLM API error: {str(e)}")
            return {
                "error": str(e),
                "content": ""
            }
    
    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _create_chunk_prompt(self, layout_type: LayoutType) -> str:        
        base_prompt = "You are an expert document analyzer. Extract and structure the content from this image element."
        
        prompts = {
            LayoutType.TABLE: base_prompt + "\n\nThis is a TABLE. Extract the data and return it as an HTML table with proper structure.",
            LayoutType.FIGURE: base_prompt + "\n\nThis is a FIGURE/IMAGE. Provide a detailed description of what you see.",
            LayoutType.TITLE: base_prompt + "\n\nThis is a TITLE/HEADING. Extract the exact text.",
            LayoutType.TEXT: base_prompt + "\n\nThis is TEXT content. Extract all text accurately, preserving formatting if any.",
            LayoutType.LIST: base_prompt + "\n\nThis is a LIST. Extract all items in markdown list format.",
            LayoutType.EQUATION: base_prompt + "\n\nThis is an EQUATION/FORMULA. Extract it in LaTeX format if possible, or describe it.",
            LayoutType.HANDWRITTEN: base_prompt + "\n\nThis is HANDWRITTEN text. Carefully interpret and extract the text, even if partially unclear.",
        }
        
        return prompts.get(layout_type, base_prompt)


class VLMEnhancedRetriever:
    def __init__(self, vector_store, vlm_processor: VLMProcessor):
        self.vector_store = vector_store
        self.vlm = vlm_processor
    
    def retrieve_with_context(self, 
                            query: str,
                            n_results: int = 5,
                            enhance_results: bool = True) -> List[Dict[str, Any]]:
        results = self.vector_store.query(query, n_results)
        
        if not enhance_results:
            return [r.to_dict() for r in results]

        enhanced_results = []
        
        for result in results:
            enhanced = result.to_dict()

            chunk = result.chunk
            if chunk.metadata.get('image_path'):
                try:
                    vlm_analysis = self.vlm.analyze_chunk(
                        chunk.metadata['image_path'],
                        chunk
                    )
                    enhanced['vlm_analysis'] = vlm_analysis
                except Exception as e:
                    print(f"VLM enhancement failed: {e}")
            
            enhanced_results.append(enhanced)
        
        return enhanced_results
    
    def answer_question(self, 
                       query: str,
                       n_results: int = 10) -> str:

        results = self.vector_store.query(query, n_results)
        
        if not results:
            return "No relevant documents found to answer your question."

        context_parts = []
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            chunk_info = f"[Excerpt {i}]"
            chunk_info += f"\nType: {chunk.layout_type.value}"
            chunk_info += f"\nPage: {chunk.page_number}"
            if chunk.metadata.get('detected_class'):
                chunk_info += f"\nSection: {chunk.metadata['detected_class']}"
            chunk_info += f"\nContent: {chunk.text}"
            chunk_info += f"\n"
            
            context_parts.append(chunk_info)
        
        context = "\n".join(context_parts)

        prompt = f"""You are an expert document analyst. Answer the user's question based on the provided document excerpts.

                    Question: {query}

                    Document Excerpts:
                    {context}

                    Instructions:
                    1. Provide a direct, clear answer based on the excerpts above
                    2. If the information is not in the excerpts, say "The provided excerpts do not contain enough information to answer this question."
                    3. If you can answer, be specific and reference relevant details from the excerpts
                    4. Synthesize information across multiple excerpts if needed
                    5. Do not make assumptions beyond what's in the excerpts

                    Answer:"""
        
        try:
            response = self.vlm.client.chat.completions.create(
                model=self.vlm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            return answer.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
