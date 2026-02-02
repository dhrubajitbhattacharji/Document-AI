from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import uuid


class LayoutType(Enum):
    TITLE = "title"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    EQUATION = "equation"
    HANDWRITTEN = "handwritten"
    OTHER = "other"


@dataclass
class BoundingBox:
    x1: float  # Top-left x
    y1: float  # Top-left y
    x2: float  # Bottom-right x
    y2: float  # Bottom-right y
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_dict(self) -> Dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        }


@dataclass
class LayoutChunk:
    layout_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    layout_type: LayoutType = LayoutType.TEXT
    bbox: Optional[BoundingBox] = None
    page_number: int = 1
    confidence: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    reading_order: int = 0
    text: str = ""
    ocr_confidence: float = 0.0
    is_handwritten: bool = False
    above_ids: List[str] = field(default_factory=list)
    below_ids: List[str] = field(default_factory=list)
    left_ids: List[str] = field(default_factory=list)
    right_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "layout_id": self.layout_id,
            "layout_type": self.layout_type.value,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "page_number": self.page_number,
            "confidence": self.confidence,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "reading_order": self.reading_order,
            "text": self.text,
            "ocr_confidence": self.ocr_confidence,
            "is_handwritten": self.is_handwritten,
            "above_ids": self.above_ids,
            "below_ids": self.below_ids,
            "left_ids": self.left_ids,
            "right_ids": self.right_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LayoutChunk':
        bbox = BoundingBox(**data["bbox"]) if data.get("bbox") else None
        layout_type = LayoutType(data["layout_type"])
        
        return cls(
            layout_id=data["layout_id"],
            layout_type=layout_type,
            bbox=bbox,
            page_number=data["page_number"],
            confidence=data["confidence"],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            reading_order=data.get("reading_order", 0),
            text=data.get("text", ""),
            ocr_confidence=data.get("ocr_confidence", 0.0),
            is_handwritten=data.get("is_handwritten", False),
            above_ids=data.get("above_ids", []),
            below_ids=data.get("below_ids", []),
            left_ids=data.get("left_ids", []),
            right_ids=data.get("right_ids", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class Document:
    """Represents a complete document with all its layout chunks"""
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    num_pages: int = 0
    chunks: List[LayoutChunk] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def get_chunks_by_page(self, page_number: int) -> List[LayoutChunk]:
        """Get all chunks for a specific page"""
        return [chunk for chunk in self.chunks if chunk.page_number == page_number]
    
    def get_chunks_by_type(self, layout_type: LayoutType) -> List[LayoutChunk]:
        """Get all chunks of a specific type"""
        return [chunk for chunk in self.chunks if chunk.layout_type == layout_type]
    
    def get_chunk_by_id(self, layout_id: str) -> Optional[LayoutChunk]:
        """Get a specific chunk by ID"""
        for chunk in self.chunks:
            if chunk.layout_id == layout_id:
                return chunk
        return None
    
    def get_reading_order(self) -> List[LayoutChunk]:
        """Get chunks sorted by reading order"""
        return sorted(self.chunks, key=lambda x: (x.page_number, x.reading_order))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "num_pages": self.num_pages,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """Create from dictionary"""
        chunks = [LayoutChunk.from_dict(chunk_data) for chunk_data in data.get("chunks", [])]
        
        return cls(
            document_id=data["document_id"],
            file_path=data["file_path"],
            num_pages=data["num_pages"],
            chunks=chunks,
            metadata=data.get("metadata", {})
        )


@dataclass
class QueryResult:
    """Result from querying the vector database"""
    chunk: LayoutChunk
    similarity_score: float
    document_id: str
    context_chunks: List[LayoutChunk] = field(default_factory=list)  # Surrounding chunks for context
    
    def to_dict(self) -> Dict:
        return {
            "chunk": self.chunk.to_dict(),
            "similarity_score": self.similarity_score,
            "document_id": self.document_id,
            "context_chunks": [c.to_dict() for c in self.context_chunks]
        }
