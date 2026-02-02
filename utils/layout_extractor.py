"""
Layout extraction using DocLayout-YOLO.
Detects and assigns IDs to layout elements in documents.
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision
from huggingface_hub import snapshot_download
from doclayout_yolo import YOLOv10

from .models import LayoutChunk, LayoutType, BoundingBox, Document


class LayoutExtractor:
    """Extract layout elements from documents using DocLayout-YOLO"""
    
    # Mapping from DocLayout-YOLO class IDs to names
    ID_TO_NAMES = {
        0: 'title', 
        1: 'text',  # plain text
        2: 'abandon',  # abandon
        3: 'figure', 
        4: 'text',  # figure_caption
        5: 'table', 
        6: 'text',  # table_caption
        7: 'text',  # table_footnote
        8: 'text',  # isolate_formula
        9: 'text'  # formula_caption
    }
    
    # Mapping from names to LayoutType
    LAYOUT_TYPE_MAPPING = {
        "title": LayoutType.TITLE,
        "text": LayoutType.TEXT,
        "plain text": LayoutType.TEXT,
        "table": LayoutType.TABLE,
        "figure": LayoutType.FIGURE,
        "caption": LayoutType.CAPTION,
        "header": LayoutType.HEADER,
        "footer": LayoutType.FOOTER,
        "list": LayoutType.LIST,
        "equation": LayoutType.EQUATION,
        "formula": LayoutType.EQUATION,
        "handwritten": LayoutType.HANDWRITTEN,
        "abandon": LayoutType.OTHER,
    }
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25, 
                 iou_threshold: float = 0.45):
        """
        Initialize the layout extractor.
        
        Args:
            model_path: Path to DocLayout-YOLO model weights. If None, downloads from HuggingFace.
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        
        if model_path and os.path.exists(model_path):
            print(f"Loading DocLayout-YOLO model from: {model_path}")
            self.model = YOLOv10(model_path)
        else:
            # Download from HuggingFace Hub
            print("Downloading DocLayout-YOLO model from HuggingFace...")
            model_dir = snapshot_download(
                'juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501',
                local_dir='./models/DocLayout-YOLO-DocStructBench-imgsz1280-2501'
            )
            model_path = os.path.join(model_dir, "doclayout_yolo_docstructbench_imgsz1280_2501.pt")
            print(f"Loading model from: {model_path}")
            self.model = YOLOv10(model_path)
    
    def extract_layouts(self, image_path: str, page_number: int = 1) -> List[LayoutChunk]:
        """
        Extract layout elements from a single image/page.
        
        Args:
            image_path: Path to the image file
            page_number: Page number in the document
            
        Returns:
            List of LayoutChunk objects with assigned IDs and bounding boxes
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run detection with DocLayout-YOLO
        det_res = self.model.predict(
            image_path,
            imgsz=1280,
            conf=self.confidence_threshold,
            device=self.device
        )[0]
        
        # Move to CPU once for post-processing
        boxes = det_res.boxes.xyxy.detach().cpu()
        classes = det_res.boxes.cls.detach().cpu().to(torch.int64)
        scores = det_res.boxes.conf.detach().cpu()
        
        # Return empty if no detections
        if boxes.numel() == 0:
            return []
        
        # Apply NMS (Non-Maximum Suppression)
        keep = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=self.iou_threshold)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        
        # Single-detection safety (ensure shape [N, 4])
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
            classes = classes.unsqueeze(0)
        
        chunks = []
        for i in range(len(boxes)):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = boxes[i].tolist()
            confidence = float(scores[i])
            class_id = int(classes[i])
            
            # Map to class name and then to LayoutType
            class_name = self.ID_TO_NAMES.get(class_id, 'text')
            layout_type = self._map_layout_type(class_name)
            
            # Create bounding box
            bbox = BoundingBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2)
            )
            
            # Create layout chunk
            chunk = LayoutChunk(
                layout_type=layout_type,
                bbox=bbox,
                page_number=page_number,
                confidence=confidence,
                metadata={
                    "detected_class": class_name,
                    "class_id": class_id,
                    "image_path": image_path
                }
            )
            
            chunks.append(chunk)
        
        # Calculate reading order based on spatial positions
        chunks = self._calculate_reading_order(chunks)
        
        # Establish spatial relationships
        chunks = self._establish_relationships(chunks)
        
        return chunks
    
    def extract_layouts_from_pdf(self, pdf_path: str, dpi: int = 300) -> Document:
        """
        Extract layouts from a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for PDF to image conversion
            
        Returns:
            Document object with all layout chunks
        """
        try:
            import pdf2image
        except ImportError:
            raise ImportError("pdf2image is required for PDF processing. Install with: pip install pdf2image")
        
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        
        document = Document(
            file_path=pdf_path,
            num_pages=len(images)
        )
        
        # Process each page
        for page_num, image in enumerate(images, start=1):
            # Save temporary image
            temp_image_path = f"/tmp/page_{page_num}.png"
            image.save(temp_image_path)
            
            # Extract layouts from the page
            page_chunks = self.extract_layouts(temp_image_path, page_number=page_num)
            document.chunks.extend(page_chunks)
            
            # Clean up temporary file
            os.remove(temp_image_path)
        
        return document

    def extract_layouts_from_images(
        self,
        image_paths: List[str],
        original_pdf_path: Optional[str] = None,
    ) -> Document:
        """Extract layouts from a list of already-rendered page images."""
        document = Document(
            file_path=original_pdf_path or (image_paths[0] if image_paths else ""),
            num_pages=len(image_paths),
        )

        for page_num, image_path in enumerate(image_paths, start=1):
            page_chunks = self.extract_layouts(image_path, page_number=page_num)
            document.chunks.extend(page_chunks)

        return document

    def render_pdf_to_images(
        self,
        pdf_path: str,
        dpi: int = 300,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """Render PDF pages to image files for downstream processing."""
        try:
            import pdf2image
        except ImportError as exc:
            raise ImportError("pdf2image is required to render PDF pages") from exc

        target_dir = (
            Path(output_dir)
            if output_dir
            else Path("./output/pdf_pages") / Path(pdf_path).stem
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        saved_paths: List[str] = []
        for idx, image in enumerate(images, start=1):
            path = target_dir / f"page_{idx}.png"
            image.save(path)
            saved_paths.append(str(path))
        return saved_paths
    
    def _map_layout_type(self, class_name: str) -> LayoutType:
        """Map detected class name to LayoutType enum"""
        for key, value in self.LAYOUT_TYPE_MAPPING.items():
            if key in class_name:
                return value
        return LayoutType.OTHER
    
    def _calculate_reading_order(self, chunks: List[LayoutChunk]) -> List[LayoutChunk]:
        """
        Calculate reading order based on spatial position.
        Uses top-to-bottom, left-to-right ordering.
        """
        if not chunks:
            return chunks
        
        # Sort by vertical position (top to bottom) with some tolerance for same-line elements
        def sort_key(chunk: LayoutChunk):
            if chunk.bbox:
                # Group elements in horizontal bands (tolerance of 20 pixels)
                y_band = int(chunk.bbox.y1 / 20)
                return (y_band, chunk.bbox.x1)
            return (0, 0)
        
        sorted_chunks = sorted(chunks, key=sort_key)
        
        # Assign reading order
        for i, chunk in enumerate(sorted_chunks):
            chunk.reading_order = i
        
        return sorted_chunks
    
    def _establish_relationships(self, chunks: List[LayoutChunk]) -> List[LayoutChunk]:
        """
        Establish spatial relationships between layout chunks.
        Determines which chunks are above, below, left, or right of each other.
        """
        if len(chunks) < 2:
            return chunks
        
        for i, chunk1 in enumerate(chunks):
            if not chunk1.bbox:
                continue
            
            for j, chunk2 in enumerate(chunks):
                if i == j or not chunk2.bbox:
                    continue
                
                # Calculate relative positions
                center1 = chunk1.bbox.center
                center2 = chunk2.bbox.center
                
                # Vertical relationships
                if chunk2.bbox.y2 < chunk1.bbox.y1:  # chunk2 is above chunk1
                    chunk1.above_ids.append(chunk2.layout_id)
                elif chunk2.bbox.y1 > chunk1.bbox.y2:  # chunk2 is below chunk1
                    chunk1.below_ids.append(chunk2.layout_id)
                
                # Horizontal relationships
                if chunk2.bbox.x2 < chunk1.bbox.x1:  # chunk2 is left of chunk1
                    chunk1.left_ids.append(chunk2.layout_id)
                elif chunk2.bbox.x1 > chunk1.bbox.x2:  # chunk2 is right of chunk1
                    chunk1.right_ids.append(chunk2.layout_id)
                
                # Parent-child relationships (e.g., caption below figure)
                if chunk1.layout_type == LayoutType.FIGURE and chunk2.layout_type == LayoutType.CAPTION:
                    # If caption is directly below figure and horizontally aligned
                    if (chunk2.bbox.y1 > chunk1.bbox.y2 and 
                        abs(center1[0] - center2[0]) < chunk1.bbox.width * 0.5):
                        chunk2.parent_id = chunk1.layout_id
                        chunk1.children_ids.append(chunk2.layout_id)
        
        return chunks
    
    def visualize_layout(self, image_path: str, chunks: List[LayoutChunk], 
                        output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detected layouts on the image.
        
        Args:
            image_path: Path to original image
            chunks: List of layout chunks to visualize
            output_path: Optional path to save the visualization
            
        Returns:
            Image with drawn bounding boxes
        """
        image = cv2.imread(image_path)
        
        # Color map for different layout types
        colors = {
            LayoutType.TITLE: (255, 0, 0),      # Blue
            LayoutType.TEXT: (0, 255, 0),        # Green
            LayoutType.TABLE: (0, 0, 255),       # Red
            LayoutType.FIGURE: (255, 255, 0),    # Cyan
            LayoutType.CAPTION: (255, 0, 255),   # Magenta
            LayoutType.HEADER: (0, 255, 255),    # Yellow
            LayoutType.FOOTER: (128, 128, 128),  # Gray
            LayoutType.LIST: (255, 128, 0),      # Orange
            LayoutType.EQUATION: (128, 0, 255),  # Purple
        }
        
        for chunk in chunks:
            if not chunk.bbox:
                continue
            
            color = colors.get(chunk.layout_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(
                image,
                (int(chunk.bbox.x1), int(chunk.bbox.y1)),
                (int(chunk.bbox.x2), int(chunk.bbox.y2)),
                color,
                2
            )
            
            # Add label
            label = f"{chunk.layout_type.value} ({chunk.reading_order})"
            cv2.putText(
                image,
                label,
                (int(chunk.bbox.x1), int(chunk.bbox.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
