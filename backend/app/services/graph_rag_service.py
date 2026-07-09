import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain
from functools import cached_property
from app.vector.embedding_service import EmbeddingService
from app.vector.qdrant_store import QdrantSectionStore
from app.core.config import Settings
from app.db.neo4j_graph import Neo4jGraphFactory
from app.llm.factory import LLMFactory
from app.prompts.answer_prompt import ANSWER_TEMPLATE
from app.prompts.cypher_prompt import CYPHER_GENERATION_TEMPLATE
from app.services.cypher_utils import assert_readonly, extract_cypher, normalize_rows, repair_invalid_kind

logger = logging.getLogger(__name__)

QDRANT_ENRICH_CYPHER = """
MATCH (s:Section)
WHERE s.id IN $section_ids
OPTIONAL MATCH (c:Concept)-[r]->(s)
WHERE r IS NULL OR type(r) ENDS WITH '_SECTION'
OPTIONAL MATCH (a:Article)-[:HAS_SECTION]->(s)
RETURN
  s.id AS section_id,
  s.heading AS heading,
  left(s.text, $text_chars) AS text,
  collect(DISTINCT c.displayName) AS concepts,
  collect(DISTINCT c.kind) AS concept_kinds,
  collect(DISTINCT type(r)) AS relations,
  collect(DISTINCT a.title) AS articles
"""

class YouMedGraphRAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMFactory(settings).create_chat_model()
        self.graph = Neo4jGraphFactory(settings).create_graph()
        self.cypher_chain = self._build_cypher_chain()

    @cached_property
    def embedding_service(self) -> EmbeddingService:
        return EmbeddingService(self.settings)

    @cached_property
    def qdrant_store(self) -> QdrantSectionStore:
        return QdrantSectionStore(self.settings)

    def _build_cypher_chain(self) -> GraphCypherQAChain:
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_TEMPLATE,
        )
        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=cypher_prompt,
            verbose=self.settings.debug,
            validate_cypher=False,
            top_k=self.settings.graph_top_k,
            return_intermediate_steps=True,
            return_direct=True,
            allow_dangerous_requests=True,
        )

    def refresh_schema(self) -> None:
        self.graph.refresh_schema()

    def run_cypher_readonly(self, cypher: str) -> List[Dict[str, Any]]:
        if not cypher:
            return []
        assert_readonly(cypher)
        return normalize_rows(self.graph.query(cypher))

    def ask_graph(self, question: str) -> Dict[str, Any]:
        try:
            chain_result = self.cypher_chain.invoke({"query": question})
            cypher = extract_cypher(chain_result)

            if not cypher:
                return self._empty_result(question, error="empty_cypher_from_chain")

            repaired_cypher = repair_invalid_kind(cypher, question)
            if repaired_cypher != cypher:
                logger.info("Repaired invalid kind in Cypher")
                cypher = repaired_cypher

            # Không lấy rows từ chain_result nữa.
            # Chạy lại Cypher trực tiếp để đảm bảo lấy raw rows từ Neo4j.
            rows = self.run_cypher_readonly(cypher)

            return {
                "question": question,
                "cypher": cypher,
                "rows": rows,
                "row_count": len(rows),
                "answer": "",
                "error": None,
            }

        except Exception as exc:
            logger.exception("Graph query failed")
            return self._empty_result(question, error=repr(exc))

    def ask_graph_with_answer(self, question: str) -> Dict[str, Any]:
        graph_result = self.ask_graph(question)
        if graph_result.get("error"):
            graph_result["answer"] = f"Chưa thể truy vấn graph do lỗi: {graph_result.get('error')}"
            return graph_result

        try:
            rows = self.compact_rows_for_answer(graph_result["rows"])
            prompt = ANSWER_TEMPLATE.format(
                question=graph_result["question"],
                cypher=graph_result["cypher"],
                rows=json.dumps(rows, ensure_ascii=False, indent=2),
            )
            answer = self.llm.invoke([HumanMessage(content=prompt)])
            graph_result["answer"] = getattr(answer, "content", str(answer))
            graph_result["answer_error"] = None
            return graph_result
        except Exception as exc:
            logger.exception("Answer generation failed")
            graph_result["answer"] = ""
            graph_result["answer_error"] = repr(exc)
            return graph_result

    def ask_hybrid(self, question: str) -> Dict[str, Any]:
        """Semantic Section retrieval via Qdrant, then Neo4j enrichment by section_id."""
        try:
            query_vector = self.embedding_service.embed_query(question)
            qdrant_hits = self.qdrant_store.search_sections(
                query_vector,
                top_k=self.settings.qdrant_top_k,
                score_threshold=self.settings.qdrant_score_threshold,
            )
            section_ids = [str(hit.get("section_id")) for hit in qdrant_hits if hit.get("section_id")]

            rows = self.enrich_sections_from_neo4j(section_ids, qdrant_hits)
            cypher = self._hybrid_debug_cypher(section_ids)

            graph_result = {
                "question": question,
                "answer": "",
                "cypher": cypher,
                "rows": rows,
                "row_count": len(rows),
                "error": None,
                "retrieval_mode": "qdrant_neo4j",
                "qdrant_hits": qdrant_hits,
            }
            return self._answer_from_graph_result(graph_result)
        except Exception as exc:
            logger.exception("Hybrid Qdrant retrieval failed")
            result = self._empty_result(question, error=repr(exc), retrieval_mode="qdrant_neo4j")
            result["answer"] = f"Chưa thể truy vấn Qdrant/Neo4j do lỗi: {repr(exc)}"
            return result

    def enrich_sections_from_neo4j(self, section_ids: List[str], qdrant_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not section_ids:
            return []

        rows = normalize_rows(
            self.graph.query(
                QDRANT_ENRICH_CYPHER,
                params={"section_ids": section_ids, "text_chars": self.settings.answer_max_text_chars},
            )
        )
        row_by_id = {str(row.get("section_id")): row for row in rows if row.get("section_id")}

        output: List[Dict[str, Any]] = []
        for rank, hit in enumerate(qdrant_hits, start=1):
            section_id = str(hit.get("section_id") or "")
            row = dict(row_by_id.get(section_id) or {})
            if not row:
                continue
            row["qdrant_rank"] = rank
            row["qdrant_score"] = hit.get("score")
            row["qdrant_payload_heading"] = hit.get("heading")
            output.append(row)
        return output
    
    def _answer_from_graph_result(self, graph_result: Dict[str, Any]) -> Dict[str, Any]:
        if graph_result.get("error"):
            graph_result["answer"] = f"Chưa thể truy vấn graph do lỗi: {graph_result.get('error')}"
            return graph_result

        try:
            rows = self.compact_rows_for_answer(graph_result["rows"])
            prompt = ANSWER_TEMPLATE.format(
                question=graph_result["question"],
                cypher=graph_result.get("cypher", ""),
                rows=json.dumps(rows, ensure_ascii=False, indent=2),
            )
            answer = self.llm.invoke([HumanMessage(content=prompt)])
            graph_result["answer"] = getattr(answer, "content", str(answer))
            graph_result["answer_error"] = None
            return graph_result
        except Exception as exc:
            logger.exception("Answer generation failed")
            graph_result["answer"] = ""
            graph_result["answer_error"] = repr(exc)
            return graph_result
        
    def compact_rows_for_answer(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for row in (rows or [])[: self.settings.answer_max_rows]:
            item: Dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, str) and key in {"text", "evidence_text"}:
                    item[key] = value[: self.settings.answer_max_text_chars]
                else:
                    item[key] = value
            output.append(item)
        return output

    @staticmethod
    def _hybrid_debug_cypher(section_ids: List[str]) -> str:
        return (
            "// Qdrant returned section_id values. Neo4j enrich query:\n"
            "MATCH (s:Section)\n"
            "WHERE s.id IN $section_ids\n"
            "OPTIONAL MATCH (c:Concept)-[r]->(s)\n"
            "WHERE r IS NULL OR type(r) ENDS WITH '_SECTION'\n"
            "OPTIONAL MATCH (a:Article)-[:HAS_SECTION]->(s)\n"
            "RETURN s.id AS section_id, s.heading AS heading, left(s.text, $text_chars) AS text,\n"
            "       collect(DISTINCT c.displayName) AS concepts,\n"
            "       collect(DISTINCT c.kind) AS concept_kinds,\n"
            "       collect(DISTINCT type(r)) AS relations,\n"
            "       collect(DISTINCT a.title) AS articles\n"
            f"// section_ids={json.dumps(section_ids, ensure_ascii=False)}"
        )


    @staticmethod
    def _empty_result(question: str, error: str | None = None, retrieval_mode: str = "cypher") -> Dict[str, Any]:
        return {
            "question": question,
            "answer": "",
            "cypher": "",
            "rows": [],
            "row_count": 0,
            "error": error,
            "retrieval_mode": retrieval_mode,
            "qdrant_hits": [],
        }
