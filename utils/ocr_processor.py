import os
from typing import List, Dict, Optional
import time
from PIL import Image
import io
import numpy as np
import cv2
import base64
from openai import OpenAI, Timeout
from .models import LayoutChunk, LayoutType


class OCRProcessor:
    def __init__(self, 
                 api_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None):
        
        self.api_endpoint = api_endpoint or os.environ.get("PROD_VLLM2_API_ENDPOINT")
        self.api_key = api_key or os.environ.get("PROD_VLLM2_API_KEY")
        self.model_name = model_name or os.environ.get("PROD_VLLM2_MODEL")
        
        if not self.api_endpoint or not self.api_key:
            print("Warning: Primary OCR not found. Will use fallback OCR.")
            self.client = None
        else:
            self.client = OpenAI(
                base_url=self.api_endpoint,
                api_key=self.api_key,
                timeout=Timeout(300.0),
                max_retries=0
            )
            print(f"OCR initialized via VLLM API: {self.model_name}")
    
    def process_chunk(self, image_path: str, chunk: LayoutChunk, 
                     enhance_quality: bool = True) -> LayoutChunk:
        if not chunk.bbox:
            return chunk
        
        cropped_image = self._crop_image(image_path, chunk.bbox)        

        if enhance_quality:
            processed_image = self._preprocess_image(cropped_image, chunk.layout_type)
        else:
            processed_image = cropped_image
        
        ocr_result = self._call_nanonets_model(processed_image)
        
        chunk.text = ocr_result.get("text", "")
        chunk.ocr_confidence = ocr_result.get("confidence", 0.0)
        chunk.is_handwritten = self._detect_handwriting(ocr_result, chunk.layout_type)        
        chunk.metadata["ocr_raw"] = ocr_result
        chunk.metadata["ocr_engine"] = "nanonets-hf"
        
        return chunk
    
    def process_chunks(self, image_path: str, chunks: List[LayoutChunk], 
                      batch_size: int = 5, delay: float = 0.1) -> List[LayoutChunk]:
        
        processed_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            for chunk in batch:
                try:
                    enhance = chunk.layout_type in [LayoutType.HANDWRITTEN, LayoutType.TABLE]
                    processed_chunk = self.process_chunk(image_path, chunk, enhance_quality=enhance)
                    processed_chunks.append(processed_chunk)
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk.layout_id}: {str(e)}")
                    chunk.metadata["ocr_error"] = str(e)
                    processed_chunks.append(chunk)
            
            if i + batch_size < len(chunks):
                time.sleep(delay)
        
        return processed_chunks
    
    def _crop_image(self, image_path: str, bbox) -> np.ndarray:
        image = cv2.imread(image_path)
        
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(image.shape[1], int(bbox.x2))
        y2 = min(image.shape[0], int(bbox.y2))
        
        cropped = image[y1:y2, x1:x2]
        return cropped
    
    def _preprocess_image(self, image: np.ndarray, layout_type: LayoutType) -> np.ndarray:
        """
        - TEXT: Denoise, binarize
        - TABLE: Enhance grid lines
        - HANDWRITTEN: Light denoising only
        - EQUATION: High contrast
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if layout_type == LayoutType.HANDWRITTEN:
            processed = cv2.fastNlMeansDenoising(gray, h=10)
            
        elif layout_type == LayoutType.TABLE:
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif layout_type == LayoutType.EQUATION:
            processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
        else:
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            processed = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        return processed
    
    def _call_nanonets_model(self, image: np.ndarray) -> Dict:

        if self.client is None:
            return {"text": "", "confidence": 0.0, "error": "OCR API not configured"}
        
        try:
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image).convert("RGB")
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{base64_image}"
            prompt = "Extract all text from this image. Return only the text content, no explanations."
            payload = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=payload,
            )
            
            text = response.choices[0].message.content.strip()            
            confidence = 0.95 if text else 0.0
            
            return {
                "text": text,
                "confidence": confidence,
                "model": self.model_name
            }
            
        except Exception as e:
            print(f"Nanonets OCR API error: {str(e)}")
            return {"text": "", "confidence": 0.0, "error": str(e)}
    
    def _parse_nanonets_response(self, response: Dict) -> Dict:
        return response
    
    def _detect_handwriting(self, ocr_result: Dict, layout_type: LayoutType) -> bool:
        if layout_type == LayoutType.HANDWRITTEN:
            return True
        
        confidence = ocr_result.get("confidence", 1.0)
        if confidence < 0.6:
            return True
        
        return False
    
    def process_with_fallback(self, image_path: str, chunk: LayoutChunk) -> LayoutChunk:
        try:
            result = self.process_chunk(image_path, chunk, enhance_quality=True)
            if result.ocr_confidence < 0.5 and result.text.strip():
                print(f"Low confidence ({result.ocr_confidence}), trying enhanced preprocessing...")
                enhanced_result = self.process_chunk(image_path, chunk, enhance_quality=True)
                
                if enhanced_result.ocr_confidence > result.ocr_confidence:
                    return enhanced_result
            
            return result
            
        except Exception as e:
            print(f"OCR failed for chunk {chunk.layout_id}: {str(e)}")
            chunk.metadata["ocr_error"] = str(e)
            return chunk


class FallbackOCRProcessor(OCRProcessor):
    def __init__(self, 
                 api_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 use_tesseract: bool = True, 
                 use_easyocr: bool = True):
        super().__init__(api_endpoint, api_key, model_name)
        
        self.use_tesseract = use_tesseract
        self.use_easyocr = use_easyocr
        
        if use_easyocr:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False) 
                print("✓ EasyOCR initialized as fallback")
            except ImportError:
                print("EasyOCR not available. Install with: pip install easyocr")
                self.use_easyocr = False
            except Exception as e:
                print(f"EasyOCR initialization failed: {e}")
                self.use_easyocr = False
    
    def process_chunk(self, image_path: str, chunk: LayoutChunk, 
                     enhance_quality: bool = True) -> LayoutChunk:

        if self.client is not None:
            try:
                return super().process_chunk(image_path, chunk, enhance_quality)
            except Exception as e:
                print(f"Nanonets API failed: {e}, using fallback...")
        
        if self.use_easyocr:
            try:
                return self._process_with_easyocr(image_path, chunk)
            except Exception as e:
                print(f"EasyOCR failed: {e}")
        
        if self.use_tesseract:
            try:
                return self._process_with_tesseract(image_path, chunk)
            except Exception as e:
                print(f"Tesseract failed: {e}")
        
        return chunk
    
    def _process_with_easyocr(self, image_path: str, chunk: LayoutChunk) -> LayoutChunk:

        cropped = self._crop_image(image_path, chunk.bbox)
        results = self.easyocr_reader.readtext(cropped)
        
        texts = [text for (_, text, _) in results]
        confidences = [conf for (_, _, conf) in results]
        
        chunk.text = " ".join(texts)
        chunk.ocr_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        chunk.metadata["ocr_engine"] = "easyocr"
        
        return chunk
    
    def _process_with_tesseract(self, image_path: str, chunk: LayoutChunk) -> LayoutChunk:

        try:
            import pytesseract
        except ImportError:
            raise ImportError("pytesseract not available. Install with: pip install pytesseract")
        
        cropped = self._crop_image(image_path, chunk.bbox)
        processed = self._preprocess_image(cropped, chunk.layout_type)
        
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        
        texts = [text for text in data['text'] if text.strip()]
        confidences = [conf for conf, text in zip(data['conf'], data['text']) 
                      if text.strip() and conf != -1]
        
        chunk.text = " ".join(texts)
        chunk.ocr_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        chunk.metadata["ocr_engine"] = "tesseract"
        
        return chunk
