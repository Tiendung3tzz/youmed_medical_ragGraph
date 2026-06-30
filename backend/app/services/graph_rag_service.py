import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain

from app.core.config import Settings
from app.db.neo4j_graph import Neo4jGraphFactory
from app.llm.factory import LLMFactory
from app.prompts.answer_prompt import ANSWER_TEMPLATE
from app.prompts.cypher_prompt import CYPHER_GENERATION_TEMPLATE
from app.services.cypher_utils import assert_readonly, extract_cypher, normalize_rows, repair_invalid_kind

logger = logging.getLogger(__name__)


class YouMedGraphRAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMFactory(settings).create_chat_model()
        self.graph = Neo4jGraphFactory(settings).create_graph()
        self.cypher_chain = self._build_cypher_chain()

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
    def _empty_result(question: str, error: str | None = None) -> Dict[str, Any]:
        return {
            "question": question,
            "cypher": "",
            "rows": [],
            "row_count": 0,
            "answer": "",
            "error": error,
        }
