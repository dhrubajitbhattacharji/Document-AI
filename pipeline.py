import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import json
from tqdm import tqdm
from dotenv import load_dotenv
from utils.models import Document, LayoutChunk, QueryResult, LayoutType
from utils.layout_extractor import LayoutExtractor
from utils.ocr_processor import OCRProcessor, FallbackOCRProcessor
from utils.vector_store import VectorStore, HybridVectorStore
from utils.vlm_processor import VLMProcessor, VLMEnhancedRetriever
from utils.agentic_evaluator import AgenticExtractionEvaluator
from utils.doctr_orientation import orientation_correct_paths

load_dotenv()
class DocumentIntelligencePipeline:
    def __init__(self, config_path: Optional[str] = None):
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()

        layout_model_path = self.config.get('layout', {}).get('model_path')
        layout_confidence = self.config.get('layout', {}).get('confidence_threshold', 0.25)
        layout_iou = self.config.get('layout', {}).get('iou_threshold', 0.45)
        self.layout_extractor = LayoutExtractor(
            model_path=layout_model_path,
            confidence_threshold=layout_confidence,
            iou_threshold=layout_iou
        )
        
        ocr_api_endpoint = self.config.get('ocr', {}).get('api_endpoint')
        ocr_api_key = self.config.get('ocr', {}).get('api_key')
        ocr_model_name = self.config.get('ocr', {}).get('model_name')
        use_fallback = self.config.get('ocr', {}).get('use_fallback', True)
        
        if use_fallback:
            self.ocr_processor = FallbackOCRProcessor(
                api_endpoint=ocr_api_endpoint,
                api_key=ocr_api_key,
                model_name=ocr_model_name,
                use_tesseract=True,
                use_easyocr=True
            )
        else:
            self.ocr_processor = OCRProcessor(
                api_endpoint=ocr_api_endpoint,
                api_key=ocr_api_key,
                model_name=ocr_model_name
            )
        
        vector_config = self.config.get('vector_store', {})
        use_hybrid = vector_config.get('use_hybrid', True)
        
        if use_hybrid:
            self.vector_store = HybridVectorStore(
                persist_directory=vector_config.get('persist_directory', './chroma_db'),
                collection_name=vector_config.get('collection_name', 'document_layouts'),
                embedding_model=vector_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            )
        else:
            self.vector_store = VectorStore(
                persist_directory=vector_config.get('persist_directory', './chroma_db'),
                collection_name=vector_config.get('collection_name', 'document_layouts'),
                embedding_model=vector_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            )

        vlm_config = self.config.get('vlm', {})
        self.use_vlm = vlm_config.get('enabled', False)
        
        if self.use_vlm:
            try:
                self.vlm_processor = VLMProcessor(
                    api_endpoint=vlm_config.get('api_endpoint'),
                    api_key=vlm_config.get('api_key'),
                    model_name=vlm_config.get('model_name')
                )
                self.vlm_retriever = VLMEnhancedRetriever(self.vector_store, self.vlm_processor)
            except Exception as e:
                print(f"Warning: VLM initialization failed: {e}")
                self.use_vlm = False
                self.vlm_processor = None
                self.vlm_retriever = None
        else:
            self.vlm_processor = None
            self.vlm_retriever = None
            print("VLM enhancement disabled (can be enabled in config)")

        self.agentic_evaluator = AgenticExtractionEvaluator(
            self.vlm_processor,
            self.config.get('agentic', {})
        )
        
        print("Pipeline initialized successfully!")
    
    def process_document(self, 
                        file_path: str,
                        document_id: Optional[str] = None,
                        visualize: bool = False,
                        save_intermediate: bool = False) -> Document:
        print(f"\n{'='*60}")
        print(f"Processing document: {file_path}")
        print(f"{'='*60}\n")

        file_ext = Path(file_path).suffix.lower()

        orient_cfg = self.config.get('orientation', {})
        orientation_enabled = orient_cfg.get('enabled', True)
        orientation_save = orient_cfg.get('save_images', True)
        angle_threshold = orient_cfg.get('angle_threshold', 1.0)
        det_arch = orient_cfg.get('det_arch', 'db_resnet50')
        reco_arch = orient_cfg.get('reco_arch', 'parseq')

        corrected_paths = []
        orientation_meta = []

        print("Stage 0: Orientation Detection & Correction")
        print("-" * 40)
        if orientation_enabled:
            try:
                if file_ext == '.pdf':

                    temp_pages = self.layout_extractor.render_pdf_to_images(file_path) if hasattr(self.layout_extractor, 'render_pdf_to_images') else []
                    corrected = orientation_correct_paths(temp_pages, './output/orientation', det_arch=det_arch, reco_arch=reco_arch, angle_threshold=angle_threshold)
                    for c in corrected:
                        corrected_paths.append(c['effective_path'])
                    orientation_meta = corrected
                elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                    corrected = orientation_correct_paths([file_path], './output/orientation', det_arch=det_arch, reco_arch=reco_arch, angle_threshold=angle_threshold)
                    corrected_paths = [corrected[0]['effective_path']]
                    orientation_meta = corrected
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
                print(f"✓ Orientation processed for {len(corrected_paths)} page(s)")
            except Exception as e:
                print(f"⚠️  Orientation correction failed: {e}. Proceeding without correction.")
                corrected_paths = []
        else:
            print("Orientation correction disabled by config")


        print("\nStage 1: Layout Detection")
        print("-" * 40)

        if file_ext == '.pdf':
            if corrected_paths:
                document = self.layout_extractor.extract_layouts_from_images(corrected_paths, original_pdf_path=file_path)
            else:
                document = self.layout_extractor.extract_layouts_from_pdf(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            working_path = corrected_paths[0] if corrected_paths else file_path
            chunks = self.layout_extractor.extract_layouts(working_path)
            document = Document(
                file_path=file_path,
                num_pages=1,
                chunks=chunks
            )
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        if document_id:
            document.document_id = document_id
        
        print(f"✓ Detected {len(document.chunks)} layout elements across {document.num_pages} page(s)")

        page_image_paths = self._resolve_page_images(file_path, corrected_paths, document)

        # Attach orientation metadata to document
        if orientation_meta:
            document.metadata['orientation'] = orientation_meta
        
        # Visualize layouts if requested
        if visualize and file_ext != '.pdf':
            output_vis_path = f"./output/layout_viz_{document.document_id}.png"
            os.makedirs('./output', exist_ok=True)
            self.layout_extractor.visualize_layout(file_path, document.chunks, output_vis_path)
            print(f"✓ Layout visualization saved to {output_vis_path}")
        
        # Stage 2: OCR Processing
        print("\nStage 2: OCR Processing")
        print("-" * 40)
        
        # For PDFs, we need to process page by page
        if file_ext == '.pdf':
            document = self._process_pdf_ocr(file_path, document)
        else:
            document.chunks = self._process_chunks_ocr(file_path, document.chunks)
        
        # Calculate statistics
        total_text_length = sum(len(chunk.text) for chunk in document.chunks)
        chunks_with_text = sum(1 for chunk in document.chunks if chunk.text.strip())
        avg_confidence = sum(chunk.ocr_confidence for chunk in document.chunks) / len(document.chunks) if document.chunks else 0
        
        print(f"✓ Extracted text from {chunks_with_text}/{len(document.chunks)} chunks")
        print(f"✓ Total text length: {total_text_length} characters")
        print(f"✓ Average OCR confidence: {avg_confidence:.2%}")
        
        # Stage 3: Vector Storage
        print("\nStage 3: Vector Storage")
        print("-" * 40)
        
        success = self.vector_store.add_document(document)
        
        if success:
            print(f"✓ Document indexed in vector database")
            print(f"✓ Document ID: {document.document_id}")
        else:
            print("✗ Failed to index document")

        print("\nStage 4: Agentic Evaluation")
        print("-" * 40)
        agentic_result = self._run_agentic_stage(document, page_image_paths)
        if agentic_result.get('decision'):
            decision = agentic_result['decision']
            winner = decision.get('winner', 'ocr').upper()
            conf = decision.get('confidence', 0.0)
            print(f"✓ Agentic winner: {winner} (confidence {conf:.2f})")
            print(f"Reason: {decision.get('reason', 'n/a')}")
        else:
            reason = agentic_result.get('skip_reason', 'Agentic evaluation skipped')
            print(f"⚠️  Agentic evaluation skipped: {reason}")
        
        # Save intermediate results if requested
        if save_intermediate:
            self._save_intermediate_results(document)
        
        print(f"\n{'='*60}")
        print("Document processing completed!")
        print(f"{'='*60}\n")
        
        return document
    
    def process_batch(self, file_paths: List[str], **kwargs) -> List[Document]:
        documents = []
        
        print(f"\nProcessing batch of {len(file_paths)} documents...")
        
        for file_path in tqdm(file_paths, desc="Processing documents"):
            try:
                document = self.process_document(file_path, **kwargs)
                documents.append(document)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        print(f"\nBatch processing completed: {len(documents)}/{len(file_paths)} successful")
        
        return documents
    
    def query(self,
             query_text: str,
             n_results: int = 5,
             layout_types: Optional[List[LayoutType]] = None,
             use_hybrid: bool = True,
             use_vlm: bool = None) -> List[QueryResult]:
        if use_vlm is None:
            use_vlm = self.use_vlm
        
        # If VLM is requested and available, use VLM-enhanced retrieval
        if use_vlm and self.vlm_retriever:
            enhanced_results = self.vlm_retriever.retrieve_with_context(
                query_text, n_results, enhance_results=True
            )
            # Convert back to QueryResult objects (simplified)
            return self._standard_query(query_text, n_results, layout_types, use_hybrid)
        
        return self._standard_query(query_text, n_results, layout_types, use_hybrid)
    
    def _standard_query(self, query_text, n_results, layout_types, use_hybrid):
        """Standard query without VLM enhancement"""
        if layout_types:
            return self.vector_store.query_by_layout_type(
                query_text, layout_types, n_results
            )
        
        if use_hybrid and isinstance(self.vector_store, HybridVectorStore):
            return self.vector_store.hybrid_query(query_text, n_results)
        
        return self.vector_store.query(query_text, n_results)
    
    def ask_question(self, question: str, n_results: int = 10) -> str:
        if not self.use_vlm or not self.vlm_retriever:
            return "VLM is not enabled. Please enable it in the configuration to use question answering."
        
        return self.vlm_retriever.answer_question(question, n_results)
    
    def enhance_chunk_with_vlm(self, document_id: str, chunk_id: str) -> Dict:
        if not self.use_vlm or not self.vlm_processor:
            return {"error": "VLM not enabled"}
        
        # Get the chunk
        chunks = self.vector_store.get_document_chunks(document_id)
        chunk = next((c for c in chunks if c.layout_id == chunk_id), None)
        
        if not chunk or not chunk.metadata.get('image_path'):
            return {"error": "Chunk not found or no image available"}
        
        # Analyze with VLM
        analysis = self.vlm_processor.analyze_chunk(
            chunk.metadata['image_path'],
            chunk
        )
        
        return {
            "chunk": chunk.to_dict(),
            "vlm_analysis": analysis
        }
    
    def query_with_formatting(self,
                             query_text: str,
                             n_results: int = 5,
                             include_metadata: bool = True) -> str:
        results = self.query(query_text, n_results)
        
        output = []
        output.append(f"\nQuery: {query_text}")
        output.append(f"Found {len(results)} results:\n")
        output.append("=" * 80)
        
        for i, result in enumerate(results, 1):
            output.append(f"\nResult {i} (Similarity: {result.similarity_score:.2%})")
            output.append("-" * 80)
            
            if include_metadata:
                output.append(f"Document ID: {result.document_id}")
                output.append(f"Layout Type: {result.chunk.layout_type.value}")
                output.append(f"Page: {result.chunk.page_number}")
                output.append(f"Reading Order: {result.chunk.reading_order}")
                output.append(f"OCR Confidence: {result.chunk.ocr_confidence:.2%}")
                output.append("")
            
            output.append(f"Text: {result.chunk.text}")
            
            if result.context_chunks:
                output.append(f"\nContext ({len(result.context_chunks)} surrounding chunks):")
                for ctx_chunk in result.context_chunks:
                    output.append(f"  [{ctx_chunk.layout_type.value}] {ctx_chunk.text[:100]}...")
            
            output.append("")
        
        return "\n".join(output)
    
    def get_document_summary(self, document_id: str) -> Dict:
        chunks = self.vector_store.get_document_chunks(document_id)
        
        if not chunks:
            return {"error": "Document not found"}
        
        # Calculate statistics
        layout_type_counts = {}
        for chunk in chunks:
            lt = chunk.layout_type.value
            layout_type_counts[lt] = layout_type_counts.get(lt, 0) + 1
        
        total_text_length = sum(len(chunk.text) for chunk in chunks)
        avg_confidence = sum(chunk.ocr_confidence for chunk in chunks) / len(chunks)
        
        pages = set(chunk.page_number for chunk in chunks)
        
        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "total_pages": len(pages),
            "layout_type_distribution": layout_type_counts,
            "total_text_length": total_text_length,
            "average_ocr_confidence": avg_confidence,
            "chunks_with_text": sum(1 for chunk in chunks if chunk.text.strip())
        }
    
    def export_document(self, document_id: str, output_path: str, format: str = 'json'):

        chunks = self.vector_store.get_document_chunks(document_id)
        
        if not chunks:
            print(f"Document {document_id} not found")
            return
        
        if format == 'json':
            data = {
                "document_id": document_id,
                "chunks": [chunk.to_dict() for chunk in chunks]
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'markdown':
            with open(output_path, 'w') as f:
                f.write(f"# Document: {document_id}\n\n")
                
                current_page = None
                for chunk in chunks:
                    if chunk.page_number != current_page:
                        current_page = chunk.page_number
                        f.write(f"\n## Page {current_page}\n\n")
                    
                    f.write(f"### {chunk.layout_type.value.title()}\n\n")
                    f.write(f"{chunk.text}\n\n")
        
        elif format == 'txt':
            with open(output_path, 'w') as f:
                for chunk in chunks:
                    f.write(f"{chunk.text}\n\n")
        
        print(f"Document exported to {output_path}")
    
    def _process_chunks_ocr(self, image_path: str, chunks: List[LayoutChunk]) -> List[LayoutChunk]:
        batch_size = self.config.get('ocr', {}).get('batch_size', 5)
        return self.ocr_processor.process_chunks(image_path, chunks, batch_size=batch_size)
    
    def _resolve_page_images(self, file_path: str, corrected_paths: List[str], document: Document) -> List[str]:
        if corrected_paths:
            return corrected_paths
        return self._render_document_images(file_path, document)

    def _render_document_images(self, file_path: str, document: Document) -> List[str]:
        file_ext = Path(file_path).suffix.lower()
        if file_ext == '.pdf':
            output_root = self.config.get('processing', {}).get('output_directory', './output')
            page_dir = Path(output_root) / 'pdf_pages' / document.document_id
            page_dir.mkdir(parents=True, exist_ok=True)
            try:
                return self.layout_extractor.render_pdf_to_images(
                    file_path,
                    dpi=self.config.get('layout', {}).get('pdf_dpi', 300),
                    output_dir=str(page_dir)
                )
            except Exception as e:
                print(f"⚠️  Failed to render PDF pages for agentic stage: {e}")
                return []
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return [file_path]
        return []

    def _run_agentic_stage(self, document: Document, page_image_paths: List[str]) -> Dict:
        if not self.agentic_evaluator:
            reason = "Agentic evaluator not initialized"
            document.metadata['agentic_skip_reason'] = reason
            return {'skip_reason': reason}

        if not self.agentic_evaluator.enabled:
            reason = "Agentic evaluator disabled via config"
            document.metadata['agentic_skip_reason'] = reason
            return {'skip_reason': reason}

        if not page_image_paths:
            reason = "No page images available for agentic evaluation"
            print(reason)
            document.metadata['agentic_skip_reason'] = reason
            return {'skip_reason': reason}

        print(f"Running agentic evaluator on {len(page_image_paths)} page image(s)")
        result = {}
        try:
            result = self.agentic_evaluator.orchestrate(document, page_image_paths)
        except Exception as e:
            reason = f"Agentic evaluator failed: {e}"
            print(reason)
            document.metadata['agentic_skip_reason'] = reason
            return {'skip_reason': reason}

        if not result:
            reason = "Agentic evaluator returned empty payload"
            document.metadata['agentic_skip_reason'] = reason
            return {'skip_reason': reason}

        if result.get('decision'):
            document.metadata['agentic'] = result
            document.metadata['agentic_winner'] = result['decision'].get('winner')
            document.metadata['agentic_best_content'] = result['decision'].get('best_content')
            document.metadata.pop('agentic_skip_reason', None)
            self._persist_agentic_outputs(document, result)
            return result

        reason = result.get('skip_reason', 'Agentic evaluator did not return a decision')
        document.metadata['agentic_skip_reason'] = reason
        result['skip_reason'] = reason
        return result

    def _persist_agentic_outputs(self, document: Document, agentic_result: Dict):
        results_dir = Path(
            self.config.get('processing', {}).get('results_directory', './results')
        )
        doc_dir = results_dir / document.document_id
        try:
            doc_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️  Could not create agentic results directory {doc_dir}: {e}")
            return

        def _write_text(path: Path, content: str):
            try:
                path.write_text(content or '', encoding='utf-8')
            except Exception as exc:
                print(f"⚠️  Failed to write {path}: {exc}")

        def _write_json(path: Path, payload: Any):
            try:
                path.write_text(json.dumps(payload or {}, indent=2), encoding='utf-8')
            except Exception as exc:
                print(f"⚠️  Failed to write {path}: {exc}")

        ocr_candidate = agentic_result.get('ocr_candidate') or {}
        vlm_candidate = agentic_result.get('vlm_candidate') or {}

        if ocr_candidate:
            _write_text(doc_dir / 'ocr_candidate.txt', ocr_candidate.get('content', ''))
            _write_json(doc_dir / 'ocr_metadata.json', ocr_candidate.get('metadata', {}))

        if vlm_candidate:
            _write_text(doc_dir / 'vlm_candidate.txt', vlm_candidate.get('content', ''))
            _write_json(doc_dir / 'vlm_metadata.json', vlm_candidate.get('metadata', {}))

        vlm_payload = agentic_result.get('vlm_payload')
        if vlm_payload:
            _write_json(doc_dir / 'vlm_payload.json', vlm_payload)

        _write_json(doc_dir / 'agentic_decision.json', agentic_result.get('decision', {}))
        print(f"✓ Agentic artifacts saved to {doc_dir}")

    def _process_pdf_ocr(self, pdf_path: str, document: Document) -> Document:
        try:
            import pdf2image
        except ImportError:
            raise ImportError("pdf2image is required for PDF processing")
        
        # Group chunks by page
        pages = {}
        for chunk in document.chunks:
            if chunk.page_number not in pages:
                pages[chunk.page_number] = []
            pages[chunk.page_number].append(chunk)
        
        # Convert PDF pages to images and process
        pdf_images = pdf2image.convert_from_path(pdf_path, dpi=300)
        
        processed_chunks = []
        for page_num, page_chunks in tqdm(pages.items(), desc="OCR processing pages"):
            # Save page as temporary image
            temp_image_path = f"/tmp/page_{page_num}.png"
            pdf_images[page_num - 1].save(temp_image_path)
            
            # Process chunks
            processed_page_chunks = self._process_chunks_ocr(temp_image_path, page_chunks)
            processed_chunks.extend(processed_page_chunks)
            
            # Clean up
            os.remove(temp_image_path)
        
        document.chunks = processed_chunks
        return document
    
    def _save_intermediate_results(self, document: Document):
        """Save intermediate processing results"""
        output_dir = './output/intermediate'
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{document.document_id}.json")
        
        with open(output_file, 'w') as f:
            json.dump(document.to_dict(), f, indent=2)
        
        print(f"✓ Intermediate results saved to {output_file}")
    
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'layout': {
                'model_path': None,
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'ocr': {
                'api_endpoint': os.environ.get('PROD_VLLM2_API_ENDPOINT'),
                'api_key': os.environ.get('PROD_VLLM2_API_KEY'),
                'model_name': os.environ.get('PROD_VLLM2_MODEL', 'nanonets/Nanonets-OCR2-3B'),
                'use_fallback': True,
                'batch_size': 5
            },
                'vector_store': {
                'persist_directory': './chroma_db',
                'collection_name': 'document_layouts',
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'use_hybrid': True
            },
                'processing': {
                    'save_intermediate': True,
                    'output_directory': './output',
                    'visualize_layouts': True,
                    'results_directory': './results'
                },
            'vlm': {
                'enabled': False,  # Set to True to enable VLM
                'api_endpoint': os.environ.get('VLLM_API_ENDPOINT'),
                'api_key': os.environ.get('VLLM_API_KEY'),
                'model_name': os.environ.get('VLM_NAME', 'Qwen/Qwen2.5-VL-7B-Instruct')
            },
            'agentic': {
                'enabled': False,
                'run_vlm_after_layout': True,
                'max_candidate_chars': 6000,
                'min_ocr_confidence': 0.55,
                'tie_breaker': 'ocr',
                'evaluation_prompt': (
                    "You are an extraction arbiter comparing two document captures. "
                    "Return JSON with fields winner ('ocr'|'vlm'), confidence (0-1), "
                    "and reason (<=20 words)."
                )
            }
        }
    
    def get_statistics(self) -> Dict:
        """Get overall pipeline statistics"""
        return self.vector_store.get_statistics()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Intelligence Pipeline")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--process', type=str, help='Path to document to process')
    parser.add_argument('--query', type=str, help='Query text')
    parser.add_argument('--visualize', action='store_true', help='Visualize layouts')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentIntelligencePipeline(config_path=args.config)
    
    # Process document
    if args.process:
        pipeline.process_document(args.process, visualize=args.visualize, save_intermediate=True)
    
    # Query
    if args.query:
        print(pipeline.query_with_formatting(args.query))
    
    # Show statistics
    if not args.process and not args.query:
        stats = pipeline.get_statistics()
        print("\nPipeline Statistics:")
        print(json.dumps(stats, indent=2))
