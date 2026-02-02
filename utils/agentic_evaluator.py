from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional

from .models import Document
from .vlm_processor import VLMProcessor


@dataclass
class ExtractionCandidate:
    source: str
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata,
        }


class AgenticExtractionEvaluator:
    def __init__(
        self,
        vlm_processor: Optional[VLMProcessor],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vlm = vlm_processor
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)
        self.max_chars = int(self.config.get("max_candidate_chars", 6000))
        self.tie_breaker = (self.config.get("tie_breaker") or "ocr").lower()
        self.min_ocr_confidence = float(self.config.get("min_ocr_confidence", 0.55))
        self.run_vlm_after_layout = self.config.get("run_vlm_after_layout", True)
        self.evaluation_prompt = self.config.get(
            "evaluation_prompt",
            "You are an extraction arbiter. Compare two candidates and select the best one.",
        )

    def orchestrate(self, document: Document, page_image_paths: List[str]) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        vlm_payload = self._run_vlm_extraction(page_image_paths) if self.run_vlm_after_layout else []
        ocr_candidate = self._build_ocr_candidate(document)
        vlm_candidate = self._build_vlm_candidate(vlm_payload)
        decision = self._decide(document, ocr_candidate, vlm_candidate)

        return {
            "vlm_payload": vlm_payload,
            "ocr_candidate": ocr_candidate.to_dict(),
            "vlm_candidate": vlm_candidate.to_dict(),
            "decision": decision,
        }
    
    def _build_ocr_candidate(self, document: Document) -> ExtractionCandidate:
        ordered_chunks = document.get_reading_order()
        buffer: List[str] = []
        confidences: List[float] = []
        chunks_with_text = 0

        for chunk in ordered_chunks:
            text = (chunk.text or "").strip()
            if text:
                label = chunk.layout_type.value
                buffer.append(f"[{chunk.page_number}:{label}] {text}")
                chunks_with_text += 1
            confidences.append(chunk.ocr_confidence)

        combined = "\n".join(buffer)
        truncated = self._truncate(combined)

        metadata = {
            "chunk_count": len(ordered_chunks),
            "chunks_with_text": chunks_with_text,
            "avg_confidence": mean(confidences) if confidences else 0.0,
            "text_length": len(combined),
        }

        return ExtractionCandidate(source="ocr", content=truncated, metadata=metadata)

    def _build_vlm_candidate(self, vlm_payload: List[Dict[str, Any]]) -> ExtractionCandidate:
        if not vlm_payload:
            return ExtractionCandidate("vlm", "", {"page_count": 0, "element_count": 0})

        serialisable = []
        element_counter = 0
        for entry in vlm_payload:
            elements = entry.get("elements") or []
            element_counter += len(elements)
            serialisable.append(
                {
                    "page": entry.get("page"),
                    "image_path": entry.get("image_path"),
                    "elements": elements,
                }
            )

        serialized = json.dumps(serialisable, indent=2)
        truncated = self._truncate(serialized)

        metadata = {
            "page_count": len(vlm_payload),
            "element_count": element_counter,
            "text_length": len(serialized),
        }
        return ExtractionCandidate(source="vlm", content=truncated, metadata=metadata)

    def _decide(
        self,
        document: Document,
        ocr_candidate: ExtractionCandidate,
        vlm_candidate: ExtractionCandidate,
    ) -> Dict[str, Any]:
        if not vlm_candidate.content.strip():
            return self._package_decision("ocr", 0.75, "No VLM extraction available", ocr_candidate, vlm_candidate)

        if not ocr_candidate.content.strip():
            return self._package_decision("vlm", 0.75, "OCR produced no text", ocr_candidate, vlm_candidate)

        if (ocr_candidate.metadata.get("avg_confidence", 0.0) < self.min_ocr_confidence
                and vlm_candidate.metadata.get("element_count", 0) > 0):
            return self._package_decision(
                "vlm",
                0.65,
                "OCR confidence below threshold",
                ocr_candidate,
                vlm_candidate,
            )

        if not self.vlm:
            winner = self.tie_breaker if self.tie_breaker in {"ocr", "vlm"} else "ocr"
            return self._package_decision(
                winner,
                0.5,
                "VLM judge unavailable; applied tie breaker",
                ocr_candidate,
                vlm_candidate,
            )

        prompt = self._compose_prompt(document, ocr_candidate, vlm_candidate)
        winner = self.tie_breaker
        confidence = 0.5
        reason = "Tie breaker engaged"
        notes = ""

        try:
            response = self.vlm.client.chat.completions.create(
                model=self.vlm.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            raw_content = response.choices[0].message.content.strip()
            parsed = self._extract_json(raw_content)
            parsed_winner = (parsed.get("winner") or "").lower()
            if parsed_winner in {"ocr", "vlm"}:
                winner = parsed_winner
                confidence = float(parsed.get("confidence", confidence))
                reason = parsed.get("reason", reason)
                notes = parsed.get("notes", "")
            else:
                reason = f"Invalid winner in response: {parsed_winner}"
            debug_payload = {"raw": raw_content, "parsed": parsed}
        except Exception as exc:
            debug_payload = {"error": str(exc)}
            reason = f"Evaluation agent failed: {exc}"

        decision = self._package_decision(winner, confidence, reason, ocr_candidate, vlm_candidate)
        if notes:
            decision["notes"] = notes
        decision["debug"] = debug_payload
        return decision

    def _package_decision(
        self,
        winner: str,
        confidence: float,
        reason: str,
        ocr_candidate: ExtractionCandidate,
        vlm_candidate: ExtractionCandidate,
    ) -> Dict[str, Any]:
        winner = winner if winner in {"ocr", "vlm"} else "ocr"
        best_content = ocr_candidate.content if winner == "ocr" else vlm_candidate.content
        best_metadata = (
            ocr_candidate.metadata if winner == "ocr" else vlm_candidate.metadata
        )
        return {
            "winner": winner,
            "confidence": max(0.0, min(confidence, 1.0)),
            "reason": reason,
            "best_content": best_content,
            "best_metadata": best_metadata,
            "sources": {
                "ocr": ocr_candidate.metadata,
                "vlm": vlm_candidate.metadata,
            },
        }

    # ------------------------------------------------------------------
    # Supporting helpers
    # ------------------------------------------------------------------
    def _run_vlm_extraction(self, page_image_paths: List[str]) -> List[Dict[str, Any]]:
        if not self.vlm or not page_image_paths:
            return []

        payload = []
        for idx, image_path in enumerate(page_image_paths, start=1):
            try:
                elements = self.vlm.analyze_full_document(image_path)
            except Exception as exc:
                print(f"VLM full-document extraction failed for {image_path}: {exc}")
                elements = []
            payload.append({
                "page": idx,
                "image_path": image_path,
                "elements": elements,
            })
        return payload

    def _compose_prompt(
        self,
        document: Document,
        ocr_candidate: ExtractionCandidate,
        vlm_candidate: ExtractionCandidate,
    ) -> str:
        head = self.evaluation_prompt.strip()
        stats = (
            f"Document ID: {document.document_id}\n"
            f"OCR avg confidence: {ocr_candidate.metadata.get('avg_confidence', 0):.2f}\n"
            f"OCR length: {ocr_candidate.metadata.get('text_length', 0)} chars\n"
            f"VLM elements: {vlm_candidate.metadata.get('element_count', 0)}\n"
            f"VLM length: {vlm_candidate.metadata.get('text_length', 0)} chars\n"
        )
        instructions = (
            "Respond strictly with JSON: "
            "{\"winner\": \"ocr|vlm\", \"confidence\": <0-1>, \"reason\": <string>, \"notes\": <string optional>}"
        )
        return (
            f"{head}\n\n{stats}\nCandidate ocr:\n{ocr_candidate.content}\n\n"
            f"Candidate vlm:\n{vlm_candidate.content}\n\n{instructions}"
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars - 20] + "\n...[truncated]"

    @staticmethod
    def _extract_json(payload: str) -> Dict[str, Any]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            start = payload.find("{")
            end = payload.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = payload[start : end + 1]
                return json.loads(snippet)
            raise


__all__ = ["AgenticExtractionEvaluator", "ExtractionCandidate"]
