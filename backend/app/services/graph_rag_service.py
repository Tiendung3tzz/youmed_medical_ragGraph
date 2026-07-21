from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain

from app.core.config import Settings
from app.db.neo4j_graph import Neo4jGraphFactory
from app.llm.factory import LLMFactory
from app.prompts.answer_prompt import ANSWER_TEMPLATE
from app.prompts.cypher_prompt import CYPHER_GENERATION_TEMPLATE
from app.services.cypher_utils import assert_readonly, extract_cypher, normalize_rows, repair_invalid_kind
from app.services.reranker_service import RerankerService
from app.vector.embedding_service import EmbeddingService
from app.vector.qdrant_store import QdrantSectionStore

logger = logging.getLogger(__name__)

ALLOWED_ROUTES = {"heading_lookup", "semantic_search", "multi_entity_search", "exact_entity_lookup"}
ALLOWED_KINDS = {"Drug", "Disease", "BodyPart", "TraditionalMedicine"}
ALLOWED_RELATIONS = {
    "HAS_OVERVIEW_SECTION",
    "HAS_DOSAGE_SECTION",
    "HAS_SIDE_EFFECT_SECTION",
    "HAS_INTERACTION_SECTION",
    "HAS_CAUTION_SECTION",
    "HAS_COMPOSITION_SECTION",
    "HAS_DIAGNOSIS_SECTION",
    "HAS_TREATMENT_SECTION",
    "HAS_SYMPTOM_SECTION",
    "HAS_CAUSE_OR_RISK_SECTION",
    "HAS_COMPLICATION_SECTION",
    "HAS_BOTANICAL_DESCRIPTION_SECTION",
    "HAS_TRADITIONAL_USE_SECTION",
    "HAS_TRADITIONAL_USE_OR_PHARMACOLOGY_SECTION",
    "HAS_PREPARATION_SECTION",
    "HAS_ANATOMY_SECTION",
    "HAS_FUNCTION_SECTION",
    "HAS_PHYSIOLOGY_SECTION",
}

HEADING_TYPE_TO_RELATION = {
    "OVERVIEW": "HAS_OVERVIEW_SECTION",
    "GENERAL": "HAS_OVERVIEW_SECTION",
    "SYMPTOM": "HAS_SYMPTOM_SECTION",
    "DIAGNOSIS": "HAS_DIAGNOSIS_SECTION",
    "TREATMENT": "HAS_TREATMENT_SECTION",
    "CAUSE_OR_RISK": "HAS_CAUSE_OR_RISK_SECTION",
    "COMPLICATION": "HAS_COMPLICATION_SECTION",
    "DOSAGE": "HAS_DOSAGE_SECTION",
    "SIDE_EFFECT": "HAS_SIDE_EFFECT_SECTION",
    "INTERACTION": "HAS_INTERACTION_SECTION",
    "CAUTION": "HAS_CAUTION_SECTION",
    "STORAGE": "HAS_STORAGE_SECTION",
    "COMPOSITION": "HAS_COMPOSITION_SECTION",
    "CONTRAINDICATION": "HAS_CONTRAINDICATION_SECTION",
    "SPECIAL_POPULATION": "HAS_SPECIAL_POPULATION_SECTION",
    "MISSED_OR_OVERDOSE": "HAS_MISSED_OR_OVERDOSE_SECTION",
    "BOTANICAL_DESCRIPTION": "HAS_BOTANICAL_DESCRIPTION_SECTION",
    "TRADITIONAL_USE": "HAS_TRADITIONAL_USE_SECTION",
    "PREPARATION": "HAS_PREPARATION_SECTION",
    "ANATOMY": "HAS_ANATOMY_SECTION",
    "FUNCTION": "HAS_FUNCTION_SECTION",
    "PHYSIOLOGY_OR_PROCESS": "HAS_PHYSIOLOGY_SECTION",
    "PHYSIOLOGY": "HAS_PHYSIOLOGY_SECTION",
}

HEADING_LOOKUP_PATTERNS = [
    r'tiêu đề\s+[“"]([^”"]+)[”"]',
    r'tieu de\s+[“"]([^”"]+)[”"]',
    r'phần tiêu đề\s+[“"]([^”"]+)[”"]',
    r'phan tieu de\s+[“"]([^”"]+)[”"]',
    r'phần\s+[“"]([^”"]+)[”"]',
    r'phan\s+[“"]([^”"]+)[”"]',
    r'mục\s+[“"]([^”"]+)[”"]',
    r'muc\s+[“"]([^”"]+)[”"]',
    r'heading\s+[“"]([^”"]+)[”"]',
]

INTENT_EXTRACTION_PROMPT = """
Bạn là bộ phân tích intent cho hệ thống GraphRAG y khoa.

Nhiệm vụ:
- Không trả lời câu hỏi.
- Không sinh Cypher.
- Chỉ trả JSON hợp lệ.

Schema JSON:
{
  "route": "heading_lookup | semantic_search | multi_entity_search | exact_entity_lookup",
  "heading": string | null,
  "entity_kind": "Drug | Disease | BodyPart | TraditionalMedicine | null",
  "entities": [string],
  "relations": [string],
  "confidence": number
}

Quy tắc:
- Nếu câu hỏi hỏi theo tiêu đề/mục/phần/heading, route = "heading_lookup".
- Nếu câu hỏi có nhiều thực thể cần so sánh/tổng hợp, route = "multi_entity_search".
- Nếu câu hỏi hỏi đúng một thực thể cụ thể, route = "exact_entity_lookup".
- Nếu câu hỏi hỏi ngữ nghĩa chung, route = "semantic_search".
- Không sinh Cypher.
- Chỉ dùng relation trong danh sách cho phép:
  HAS_OVERVIEW_SECTION,
  HAS_DOSAGE_SECTION,
  HAS_SIDE_EFFECT_SECTION,
  HAS_INTERACTION_SECTION,
  HAS_CAUTION_SECTION,
  HAS_COMPOSITION_SECTION,
  HAS_DIAGNOSIS_SECTION,
  HAS_TREATMENT_SECTION,
  HAS_SYMPTOM_SECTION,
  HAS_CAUSE_OR_RISK_SECTION,
  HAS_COMPLICATION_SECTION,
  HAS_BOTANICAL_DESCRIPTION_SECTION,
  HAS_TRADITIONAL_USE_SECTION,
  HAS_TRADITIONAL_USE_OR_PHARMACOLOGY_SECTION,
  HAS_PREPARATION_SECTION,
  HAS_ANATOMY_SECTION,
  HAS_FUNCTION_SECTION,
  HAS_PHYSIOLOGY_SECTION

Câu hỏi:
{question}
"""

QDRANT_ENRICH_CYPHER = """
MATCH (s:Section)
WHERE s.id IN $section_ids
OPTIONAL MATCH (a:Article)-[:HAS_SECTION]->(s)
OPTIONAL MATCH (a)-[:IN_CATEGORY]->(cat:Category)
OPTIONAL MATCH (c:Concept)-[section_rel]->(s)
WHERE section_rel IS NULL
   OR (type(section_rel) STARTS WITH 'HAS_' AND type(section_rel) ENDS WITH '_SECTION')
OPTIONAL MATCH (s)-[:HAS_EXTRACTED_TERM]->(term:ClinicalTerm)
WITH
    s,
    a,
    cat,
    collect(DISTINCT CASE WHEN c IS NULL THEN NULL ELSE {
        name: c.name,
        displayName: c.displayName,
        kind: c.kind,
        relation: type(section_rel)
    } END) AS raw_concepts,
    collect(DISTINCT CASE WHEN term IS NULL THEN NULL ELSE {
        id: term.id,
        name: term.name,
        displayName: term.displayName,
        kind: term.kind
    } END) AS raw_clinical_terms
RETURN
    s.id AS section_id,
    s.heading AS heading,
    left(s.text, $text_chars) AS text,
    s.order AS section_order,
    s.heading_type AS heading_type,
    a.id AS article_id,
    a.title AS article,
    cat.name AS category,
    [x IN raw_concepts WHERE x IS NOT NULL] AS concepts,
    [x IN raw_clinical_terms WHERE x IS NOT NULL] AS clinical_terms
"""

HEADING_LOOKUP_CYPHER = """
MATCH (s:Section)
WHERE toLower(trim(s.heading)) = toLower(trim($heading))
OPTIONAL MATCH (a:Article)-[:HAS_SECTION]->(s)
OPTIONAL MATCH (a)-[:IN_CATEGORY]->(cat:Category)
OPTIONAL MATCH (c:Concept)-[section_rel]->(s)
WHERE section_rel IS NULL
   OR (type(section_rel) STARTS WITH 'HAS_' AND type(section_rel) ENDS WITH '_SECTION')
OPTIONAL MATCH (s)-[:HAS_EXTRACTED_TERM]->(term:ClinicalTerm)
WITH
    s,
    a,
    cat,
    collect(DISTINCT CASE WHEN c IS NULL THEN NULL ELSE {
        name: c.name,
        displayName: c.displayName,
        kind: c.kind,
        relation: type(section_rel)
    } END) AS raw_concepts,
    collect(DISTINCT CASE WHEN term IS NULL THEN NULL ELSE {
        id: term.id,
        name: term.name,
        displayName: term.displayName,
        kind: term.kind
    } END) AS raw_clinical_terms
WITH
    s,
    a,
    cat,
    [x IN raw_concepts WHERE x IS NOT NULL] AS concepts,
    [x IN raw_clinical_terms WHERE x IS NOT NULL] AS clinical_terms
WHERE $entity_kind IS NULL
   OR any(x IN concepts WHERE x.kind = $entity_kind)
RETURN
    s.id AS section_id,
    s.heading AS heading,
    left(s.text, $text_chars) AS text,
    s.order AS section_order,
    s.heading_type AS heading_type,
    a.id AS article_id,
    a.title AS article,
    cat.name AS category,
    concepts AS concepts,
    clinical_terms AS clinical_terms
ORDER BY coalesce(a.title, ""), coalesce(s.order, 999999)
LIMIT $limit
"""

EXACT_ENTITY_LOOKUP_CYPHER = """
MATCH (c:Concept)
WHERE ($entity_kind IS NULL OR c.kind = $entity_kind)
  AND (
        toLower(trim(c.name)) = $entity_norm
     OR toLower(trim(c.displayName)) = toLower(trim($entity_raw))
  )
MATCH (c)-[section_rel]->(s:Section)
WHERE type(section_rel) STARTS WITH 'HAS_'
  AND type(section_rel) ENDS WITH '_SECTION'
OPTIONAL MATCH (a:Article)-[:HAS_SECTION]->(s)
OPTIONAL MATCH (a)-[:IN_CATEGORY]->(cat:Category)
OPTIONAL MATCH (s)-[:HAS_EXTRACTED_TERM]->(term:ClinicalTerm)
WITH
    s,
    a,
    cat,
    collect(DISTINCT {
        name: c.name,
        displayName: c.displayName,
        kind: c.kind,
        relation: type(section_rel)
    }) AS concepts,
    collect(DISTINCT CASE WHEN term IS NULL THEN NULL ELSE {
        id: term.id,
        name: term.name,
        displayName: term.displayName,
        kind: term.kind
    } END) AS raw_clinical_terms
RETURN
    s.id AS section_id,
    s.heading AS heading,
    left(s.text, $text_chars) AS text,
    s.order AS section_order,
    s.heading_type AS heading_type,
    a.id AS article_id,
    a.title AS article,
    cat.name AS category,
    concepts AS concepts,
    [x IN raw_clinical_terms WHERE x IS NOT NULL] AS clinical_terms
ORDER BY coalesce(a.title, ""), coalesce(s.order, 999999)
LIMIT $limit
"""


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", text)


def norm_entity_key(value: Any) -> str:
    text = norm_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unique_keep_order(items: Sequence[Any]) -> List[Any]:
    seen = set()
    output: List[Any] = []
    for item in items or []:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def minmax_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if abs(mx - mn) < 1e-9:
        return [0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


class YouMedGraphRAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMFactory(settings).create_chat_model()
        self.graph = Neo4jGraphFactory(settings).create_graph()
        self.cypher_chain = self._build_cypher_chain()
        self._intent_cache: Dict[str, Dict[str, Any]] = {}

    @cached_property
    def embedding_service(self) -> EmbeddingService:
        return EmbeddingService(self.settings)

    @cached_property
    def qdrant_store(self) -> QdrantSectionStore:
        return QdrantSectionStore(self.settings)

    @cached_property
    def reranker_service(self) -> RerankerService:
        return RerankerService(self.settings)

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
        return self._answer_from_graph_result(graph_result)

    def ask_hybrid(self, question: str) -> Dict[str, Any]:
        """Routed hybrid retrieval.

        Flow:
        - heading_lookup      -> Neo4j exact Section.heading
        - exact_entity_lookup -> Neo4j exact Concept, fallback Qdrant
        - multi_entity_search -> Qdrant sub-query per entity + Neo4j enrich + rerank
        - semantic_search     -> Qdrant top N + Neo4j enrich + rerank
        """
        try:
            graph_result = self.ask_routed_retrieval(
                question=question,
                top_k=self.settings.qdrant_top_k,
                search_top_k=self.settings.qdrant_search_top_k,
            )
            return self._answer_from_graph_result(graph_result)
        except Exception as exc:
            logger.exception("Hybrid routed retrieval failed")
            result = self._empty_result(question, error=repr(exc), retrieval_mode="routed_retrieval")
            result["answer"] = f"Chưa thể truy vấn Qdrant/Neo4j do lỗi: {repr(exc)}"
            return result

    def ask_routed_retrieval(self, question: str, top_k: int = 10, search_top_k: int = 50) -> Dict[str, Any]:
        slots = self.detect_query_route(question)
        route = slots.get("route")

        if route == "heading_lookup":
            result = self.neo4j_heading_lookup_by_slots(question, slots, top_k=top_k)
            if result.get("row_count", 0) > 0:
                return result
            fallback = self.ask_qdrant_neo4j_rerank(question, top_k=top_k, search_top_k=search_top_k)
            fallback.setdefault("debug", {})["fallback_from"] = "heading_lookup"
            fallback["debug"]["slots"] = slots
            return fallback

        if route == "exact_entity_lookup":
            result = self.neo4j_exact_entity_lookup_by_slots(question, slots, top_k=top_k)
            if result.get("row_count", 0) > 0:
                return result
            fallback = self.ask_qdrant_neo4j_rerank(question, top_k=top_k, search_top_k=search_top_k)
            fallback.setdefault("debug", {})["fallback_from"] = "exact_entity_lookup"
            fallback["debug"]["slots"] = slots
            return fallback

        if route == "multi_entity_search":
            result = self.ask_multi_entity_qdrant_rerank(question, slots, top_k=top_k, search_top_k=search_top_k)
            if result.get("row_count", 0) > 0:
                return result
            fallback = self.ask_qdrant_neo4j_rerank(question, top_k=top_k, search_top_k=search_top_k)
            fallback.setdefault("debug", {})["fallback_from"] = "multi_entity_search"
            fallback["debug"]["slots"] = slots
            return fallback

        return self.ask_qdrant_neo4j_rerank(question, top_k=top_k, search_top_k=search_top_k)

    # ------------------------------------------------------------------
    # Route detection / LLM intent extraction
    # ------------------------------------------------------------------
    def detect_query_route(self, question: str) -> Dict[str, Any]:
        heading = self.extract_heading_lookup_text(question)
        if heading:
            kinds = self.detect_expected_kinds(question)
            return {
                "route": "heading_lookup",
                "heading": heading,
                "entity_kind": kinds[0] if kinds else None,
                "entities": [],
                "relations": self.detect_expected_relations(question),
                "confidence": 1.0,
                "source": "rule_heading",
            }

        if self.has_structured_lookup_signal(question):
            slots = self.extract_query_intent_with_llm(question)
            slots = self.normalize_intent_slots(slots)
            slots["source"] = slots.get("source", "llm_intent")

            if slots["route"] == "heading_lookup" and not slots.get("heading"):
                slots["route"] = "semantic_search"
            if slots["route"] in {"exact_entity_lookup", "multi_entity_search"} and not slots.get("entities"):
                slots["route"] = "semantic_search"
            if float(slots.get("confidence") or 0.0) < self.settings.intent_confidence_threshold:
                slots["route"] = "semantic_search"
            return slots

        return {
            "route": "semantic_search",
            "heading": None,
            "entity_kind": None,
            "entities": [],
            "relations": self.detect_expected_relations(question),
            "confidence": 1.0,
            "source": "rule_default_semantic",
        }

    @staticmethod
    def extract_heading_lookup_text(question: str) -> Optional[str]:
        raw_q = question or ""
        for pattern in HEADING_LOOKUP_PATTERNS:
            match = re.search(pattern, raw_q, flags=re.IGNORECASE)
            if match:
                heading = match.group(1).strip()
                if heading:
                    return heading
        return None

    @staticmethod
    def has_structured_lookup_signal(question: str) -> bool:
        q = norm_text(question)
        signals = [
            "tieu de",
            "heading",
            "phan",
            "muc",
            "co phan",
            "co muc",
            "co tieu de",
            "liet ke",
            "nhung",
            "cac",
            "so sanh",
            "tong hop",
            "doi chieu",
            "khac nhau",
            "giong nhau",
        ]
        return any(signal in q for signal in signals)

    def extract_query_intent_with_llm(self, question: str) -> Dict[str, Any]:
        if not self.settings.use_llm_intent_router:
            return {"route": "semantic_search", "heading": None, "entity_kind": None, "entities": [], "relations": [], "confidence": 0.0, "source": "llm_disabled"}

        cache_key = norm_entity_key(question)
        if cache_key in self._intent_cache:
            return dict(self._intent_cache[cache_key])

        prompt = INTENT_EXTRACTION_PROMPT.replace("{question}", question or "")
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = getattr(response, "content", str(response))
            slots = self.safe_json_loads(content)
            slots = self.normalize_intent_slots(slots)
            slots["source"] = "llm_intent"
            slots["raw"] = content
        except Exception as exc:
            logger.exception("LLM intent extraction failed")
            slots = {
                "route": "semantic_search",
                "heading": None,
                "entity_kind": None,
                "entities": [],
                "relations": [],
                "confidence": 0.0,
                "source": "llm_error",
                "error": repr(exc),
            }
        self._intent_cache[cache_key] = dict(slots)
        return slots

    @staticmethod
    def safe_json_loads(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
        return json.loads(text)

    @staticmethod
    def normalize_intent_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
        route = slots.get("route")
        if route not in ALLOWED_ROUTES:
            route = "semantic_search"

        entity_kind = slots.get("entity_kind")
        if entity_kind not in ALLOWED_KINDS:
            entity_kind = None

        relations = [rel for rel in (slots.get("relations") or []) if rel in ALLOWED_RELATIONS]
        entities = [ent.strip() for ent in (slots.get("entities") or []) if isinstance(ent, str) and ent.strip()]

        heading = slots.get("heading")
        heading = heading.strip() if isinstance(heading, str) and heading.strip() else None

        try:
            confidence = float(slots.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0

        return {
            "route": route,
            "heading": heading,
            "entity_kind": entity_kind,
            "entities": unique_keep_order(entities),
            "relations": unique_keep_order(relations),
            "confidence": confidence,
            "source": slots.get("source"),
        }

    # ------------------------------------------------------------------
    # Relation/kind detection and rerank bonuses
    # ------------------------------------------------------------------
    @staticmethod
    def detect_expected_relations(question: str) -> List[str]:
        q = norm_text(question)
        rules = [
            (["tac dung phu", "khong mong muon", "phan ung phu"], ["HAS_SIDE_EFFECT_SECTION"]),
            (["lieu dung", "cach dung", "su dung the nao"], ["HAS_DOSAGE_SECTION"]),
            (["tuong tac"], ["HAS_INTERACTION_SECTION"]),
            (["bao quan", "cat giu"], ["HAS_STORAGE_SECTION"]),
            (["chong chi dinh", "khong duoc dung", "khong nen dung"], ["HAS_CONTRAINDICATION_SECTION"]),
            (["luu y", "than trong", "can chu y"], ["HAS_CAUTION_SECTION"]),
            (["thanh phan", "hoat chat", "cong thuc"], ["HAS_COMPOSITION_SECTION"]),
            (["trieu chung", "bieu hien", "dau hieu"], ["HAS_SYMPTOM_SECTION"]),
            (["chan doan", "xet nghiem", "kiem tra"], ["HAS_DIAGNOSIS_SECTION"]),
            (["dieu tri", "huong xu ly", "xu tri"], ["HAS_TREATMENT_SECTION"]),
            (["nguyen nhan", "yeu to nguy co"], ["HAS_CAUSE_OR_RISK_SECTION"]),
            (["bien chung", "hau qua"], ["HAS_COMPLICATION_SECTION"]),
            (["tong quan", "la gi", "khai niem", "gioi thieu chung"], ["HAS_OVERVIEW_SECTION"]),
            (["mo ta duoc lieu", "mo ta toan cay", "dac diem thuc vat"], ["HAS_BOTANICAL_DESCRIPTION_SECTION"]),
            (["cong dung duoc lieu", "duoc ly", "y hoc co truyen", "y hoc hien dai"], ["HAS_TRADITIONAL_USE_SECTION", "HAS_TRADITIONAL_USE_OR_PHARMACOLOGY_SECTION", "HAS_USE_SECTION"]),
            (["bao che", "che bien", "thu hai", "phan bo"], ["HAS_PREPARATION_SECTION"]),
            (["cau tao", "giai phau", "vi tri"], ["HAS_ANATOMY_SECTION"]),
            (["chuc nang", "vai tro"], ["HAS_FUNCTION_SECTION"]),
            (["sinh ly", "co che", "qua trinh"], ["HAS_PHYSIOLOGY_SECTION"]),
        ]
        found: List[str] = []
        for keywords, relations in rules:
            if any(keyword in q for keyword in keywords):
                found.extend(relations)
        return unique_keep_order(found)

    @staticmethod
    def detect_expected_kinds(question: str) -> List[str]:
        q = norm_text(question)
        kinds: List[str] = []
        if any(x in q for x in ["thuoc", "san pham", "vien uong", "vien sui", "hoat chat"]):
            kinds.append("Drug")
        if any(x in q for x in ["benh", "hoi chung", "roi loan"]):
            kinds.append("Disease")
        if any(x in q for x in ["co quan", "bo phan", "cau truc co the", "day than kinh", "giac mac", "gan", "phoi", "mat"]):
            kinds.append("BodyPart")
        if any(x in q for x in ["duoc lieu", "cay thuoc", "vi thuoc", "y hoc co truyen"]):
            kinds.append("TraditionalMedicine")
        return unique_keep_order(kinds)

    @staticmethod
    def row_relations(row: Dict[str, Any]) -> set[str]:
        rels: set[str] = set()
        for concept in row.get("concepts") or []:
            if isinstance(concept, dict) and concept.get("relation"):
                rels.add(str(concept["relation"]))
        for rel in row.get("relations") or row.get("relation_types") or []:
            if rel:
                rels.add(str(rel))
        heading_type = row.get("heading_type")
        if heading_type:
            mapped = HEADING_TYPE_TO_RELATION.get(str(heading_type).upper())
            if mapped:
                rels.add(mapped)
        return rels

    @staticmethod
    def row_kinds(row: Dict[str, Any]) -> set[str]:
        kinds: set[str] = set()
        for concept in row.get("concepts") or []:
            if isinstance(concept, dict) and concept.get("kind"):
                kinds.add(str(concept["kind"]))
        for kind in row.get("concept_kinds") or []:
            if kind:
                kinds.add(str(kind))
        category = norm_text(row.get("category"))
        if category == "drug":
            kinds.add("Drug")
        elif category == "disease":
            kinds.add("Disease")
        elif category in {"body-part", "bodypart"}:
            kinds.add("BodyPart")
        elif category in {"medicine", "traditional-medicine", "traditionalmedicine"}:
            kinds.add("TraditionalMedicine")
        return kinds

    def relation_bonus(self, question: str, row: Dict[str, Any]) -> float:
        expected = set(self.detect_expected_relations(question))
        return 0.30 if expected and expected & self.row_relations(row) else 0.0

    def kind_bonus(self, question: str, row: Dict[str, Any]) -> float:
        expected = set(self.detect_expected_kinds(question))
        return 0.06 if expected and expected & self.row_kinds(row) else 0.0

    def entity_bonus(self, question: str, row: Dict[str, Any]) -> float:
        q = norm_text(question)
        bonus = 0.0
        for concept in row.get("concepts") or []:
            if not isinstance(concept, dict):
                continue
            display = norm_text(concept.get("displayName"))
            name = norm_text(concept.get("name"))
            if display and len(display) >= 3 and display in q:
                bonus += 0.18
            elif name and len(name) >= 3 and name in q:
                bonus += 0.12
        article = norm_text(row.get("article"))
        heading = norm_text(row.get("heading"))
        if article and len(article) >= 5 and article in q:
            bonus += 0.10
        if heading and len(heading) >= 5 and heading in q:
            bonus += 0.08
        return min(bonus, 0.25)

    @staticmethod
    def heading_bonus(question: str, row: Dict[str, Any]) -> float:
        q = norm_text(question)
        heading = norm_text(row.get("heading"))
        if not heading:
            return 0.0
        bonus = 0.0
        quoted = re.findall(r'"([^"]+)"|“([^”]+)”', question or "")
        quoted_norm = [norm_text(a or b) for a, b in quoted]
        if quoted_norm and any(heading == x for x in quoted_norm):
            bonus += 0.40
        intent_words = ["tac dung phu", "lieu dung", "tuong tac", "bao quan", "chong chi dinh", "luu y", "than trong", "chan doan", "trieu chung", "dieu tri", "nguyen nhan", "bien chung", "cau tao", "chuc nang", "sinh ly", "gioi thieu", "tong quan"]
        if any(word in q and word in heading for word in intent_words):
            bonus += 0.10
        return min(bonus, 0.40)

    @staticmethod
    def clinical_term_bonus(question: str, row: Dict[str, Any]) -> float:
        q = norm_text(question)
        bonus = 0.0
        for term in row.get("clinical_terms") or []:
            if not isinstance(term, dict):
                continue
            name = norm_text(term.get("displayName") or term.get("name"))
            if name and len(name) >= 4 and name in q:
                bonus += 0.05
        return min(bonus, 0.15)

    @staticmethod
    def text_bonus(question: str, row: Dict[str, Any]) -> float:
        q = norm_text(question)
        text = norm_text((row.get("text") or "")[:1000])
        bonus = 0.0
        for keyword in ["tac dung phu", "lieu dung", "tuong tac", "chan doan", "trieu chung", "dieu tri", "sinh ly", "chuc nang", "cau tao"]:
            if keyword in q and keyword in text:
                bonus += 0.03
        return min(bonus, 0.12)

    # ------------------------------------------------------------------
    # Qdrant, rerank and enrichment
    # ------------------------------------------------------------------
    def ask_qdrant_neo4j_rerank(self, question: str, top_k: int = 10, search_top_k: int = 50) -> Dict[str, Any]:
        query_vector = self.embedding_service.embed_query(question)
        qdrant_hits = self.qdrant_store.search_sections(
            query_vector,
            top_k=search_top_k,
            score_threshold=self.settings.qdrant_score_threshold,
        )
        candidate_section_ids = unique_keep_order([str(hit.get("section_id")) for hit in qdrant_hits if hit.get("section_id")])
        rows = self.enrich_sections_from_neo4j(candidate_section_ids, qdrant_hits)
        reranked_rows = self.rerank_rows(question, rows, qdrant_hits)
        final_rows = reranked_rows[:top_k]
        return {
            "question": question,
            "retrieval_mode": "qdrant_neo4j_st_bge_rerank",
            "cypher": "",
            "qdrant_hits": qdrant_hits,
            "candidate_section_ids": candidate_section_ids,
            "section_ids": [row.get("section_id") for row in final_rows if row.get("section_id")],
            "rows": final_rows,
            "row_count": len(final_rows),
            "answer": "",
            "error": None,
            "debug": {
                "route": "semantic_search",
                "search_top_k": search_top_k,
                "final_top_k": top_k,
                "candidate_count": len(candidate_section_ids),
                "qdrant_hit_count": len(qdrant_hits),
                "enriched_row_count": len(rows),
                "expected_relations": self.detect_expected_relations(question),
                "expected_kinds": self.detect_expected_kinds(question),
                "reranker": "sentence-transformers CrossEncoder" if self.settings.reranker_enabled else "disabled",
                "reranker_model": self.settings.reranker_model_name,
            },
        }

    def enrich_sections_from_neo4j(self, section_ids: List[str], qdrant_hits: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        if not section_ids:
            return []
        rows = normalize_rows(
            self.graph.query(
                QDRANT_ENRICH_CYPHER,
                params={"section_ids": section_ids, "text_chars": self.settings.answer_max_text_chars},
            )
        )
        row_by_id = {str(row.get("section_id")): row for row in rows if row.get("section_id")}
        if qdrant_hits is None:
            return [row_by_id[sid] for sid in section_ids if sid in row_by_id]

        output: List[Dict[str, Any]] = []
        for rank, hit in enumerate(qdrant_hits, start=1):
            section_id = str(hit.get("section_id") or "")
            row = dict(row_by_id.get(section_id) or {})
            if not row:
                continue
            row["_qdrant_rank"] = rank
            row["_qdrant_score"] = hit.get("score")
            row["qdrant_rank"] = rank
            row["qdrant_score"] = hit.get("score")
            row["qdrant_payload_heading"] = hit.get("heading")
            output.append(row)
        return output

    def row_to_reranker_text(self, row: Dict[str, Any]) -> str:
        concept_texts = []
        for concept in row.get("concepts") or []:
            if isinstance(concept, dict):
                concept_texts.append(" | ".join(str(x) for x in [concept.get("displayName"), concept.get("kind"), concept.get("relation")] if x))
        term_texts = []
        for term in row.get("clinical_terms") or []:
            if isinstance(term, dict):
                term_texts.append(" | ".join(str(x) for x in [term.get("displayName"), term.get("kind")] if x))
        parts = [
            f"Bài viết: {row.get('article') or ''}",
            f"Danh mục: {row.get('category') or ''}",
            f"Tiêu đề: {row.get('heading') or ''}",
            f"Loại heading: {row.get('heading_type') or ''}",
            "Khái niệm: " + "; ".join(concept_texts),
            "Thuật ngữ: " + "; ".join(term_texts[:10]),
            str(row.get("text") or ""),
        ]
        text = "\n".join(part for part in parts if part).strip()
        return text[: self.settings.reranker_max_chars]

    def attach_bge_reranker_score(self, question: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [self.row_to_reranker_text(row) for row in rows]
        raw_scores = self.reranker_service.predict(question, texts)
        norm_scores = minmax_normalize(raw_scores)
        output = []
        for row, raw, normed in zip(rows, raw_scores, norm_scores):
            item = dict(row)
            item["_bge_reranker_score_raw"] = float(raw)
            item["_bge_reranker_score"] = float(normed)
            output.append(item)
        return output

    def rerank_rows(self, question: str, rows: List[Dict[str, Any]], qdrant_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        score_by_sid = {str(hit.get("section_id")): float(hit.get("score") or 0.0) for hit in qdrant_hits or [] if hit.get("section_id")}
        rank_by_sid = {str(hit.get("section_id")): idx for idx, hit in enumerate(qdrant_hits or [], start=1) if hit.get("section_id")}
        prepared = []
        for row in rows or []:
            item = dict(row)
            sid = str(item.get("section_id") or "")
            item["_qdrant_score"] = float(item.get("_qdrant_score") or score_by_sid.get(sid, 0.0))
            item["_qdrant_rank"] = int(item.get("_qdrant_rank") or rank_by_sid.get(sid, 999999))
            prepared.append(item)

        qdrant_norm = minmax_normalize([float(row.get("_qdrant_score") or 0.0) for row in prepared])
        for row, normed in zip(prepared, qdrant_norm):
            row["_qdrant_score_norm"] = float(normed)

        prepared = self.attach_bge_reranker_score(question, prepared)
        output = []
        for row in prepared:
            item = dict(row)
            score_parts = {
                "qdrant": 0.30 * float(item.get("_qdrant_score_norm") or 0.0),
                "bge_reranker": 0.45 * float(item.get("_bge_reranker_score") or 0.0),
                "relation": self.relation_bonus(question, item),
                "entity": self.entity_bonus(question, item),
                "heading": self.heading_bonus(question, item),
                "kind": self.kind_bonus(question, item),
                "clinical_term": self.clinical_term_bonus(question, item),
                "text": self.text_bonus(question, item),
            }
            item["_rerank_score_parts"] = score_parts
            item["_rerank_score"] = sum(score_parts.values())
            item["_matched_relations"] = sorted(self.row_relations(item))
            item["_matched_kinds"] = sorted(self.row_kinds(item))
            output.append(item)

        output.sort(
            key=lambda x: (
                x.get("_rerank_score", 0.0),
                x.get("_bge_reranker_score", 0.0),
                x.get("_qdrant_score_norm", 0.0),
                -int(x.get("_qdrant_rank", 999999)),
            ),
            reverse=True,
        )
        return output

    # ------------------------------------------------------------------
    # Neo4j exact routes
    # ------------------------------------------------------------------
    def attach_route_scores(self, rows: List[Dict[str, Any]], slots: Dict[str, Any]) -> List[Dict[str, Any]]:
        expected_relations = set(slots.get("relations") or [])
        expected_kind = slots.get("entity_kind")
        expected_kinds = {expected_kind} if expected_kind else set()
        output = []
        for row in rows or []:
            item = dict(row)
            row_rels = self.row_relations(item)
            row_kinds = self.row_kinds(item)
            relation_match = bool(expected_relations & row_rels) if expected_relations else True
            kind_match = bool(expected_kinds & row_kinds) if expected_kinds else True
            score_parts = {
                "route_base": 1.0,
                "relation": 0.4 if relation_match else 0.0,
                "kind": 0.3 if kind_match else 0.0,
            }
            item["_route_score_parts"] = score_parts
            item["_route_score"] = sum(score_parts.values())
            item["_rerank_score_parts"] = score_parts
            item["_rerank_score"] = item["_route_score"]
            item["_matched_relations"] = sorted(row_rels)
            item["_matched_kinds"] = sorted(row_kinds)
            item["_qdrant_score"] = 0.0
            item["_qdrant_rank"] = 999999
            item["_qdrant_score_norm"] = 0.0
            item["_bge_reranker_score_raw"] = 0.0
            item["_bge_reranker_score"] = 0.0
            output.append(item)
        output.sort(key=lambda x: (x.get("_route_score", 0.0), x.get("article") or "", -int(x.get("section_order") or 999999)), reverse=True)
        return output

    def neo4j_heading_lookup_by_slots(self, question: str, slots: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        heading = slots.get("heading")
        entity_kind = slots.get("entity_kind")
        if not heading:
            return self._empty_result(question, error="Missing heading for heading_lookup.", retrieval_mode="neo4j_heading_lookup")
        rows = normalize_rows(
            self.graph.query(
                HEADING_LOOKUP_CYPHER,
                params={
                    "heading": heading,
                    "entity_kind": entity_kind,
                    "limit": max(top_k * 5, 50),
                    "text_chars": self.settings.answer_max_text_chars,
                },
            )
        )
        scored_rows = self.attach_route_scores(rows, slots)
        final_rows = scored_rows[:top_k]
        return {
            "question": question,
            "retrieval_mode": "neo4j_heading_lookup",
            "cypher": HEADING_LOOKUP_CYPHER,
            "qdrant_hits": [],
            "candidate_section_ids": [row.get("section_id") for row in scored_rows if row.get("section_id")],
            "section_ids": [row.get("section_id") for row in final_rows if row.get("section_id")],
            "rows": final_rows,
            "row_count": len(final_rows),
            "answer": "",
            "error": None,
            "debug": {"route": "heading_lookup", "slots": slots, "heading": heading, "entity_kind": entity_kind, "candidate_count": len(scored_rows), "final_top_k": top_k},
        }

    def neo4j_exact_entity_lookup_by_slots(self, question: str, slots: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        entities = slots.get("entities") or []
        entity_kind = slots.get("entity_kind")
        relations = slots.get("relations") or []
        if not entities:
            return self._empty_result(question, error="Missing entities for exact_entity_lookup.", retrieval_mode="neo4j_exact_entity_lookup")
        entity_raw = entities[0]
        rows = normalize_rows(
            self.graph.query(
                EXACT_ENTITY_LOOKUP_CYPHER,
                params={
                    "entity_raw": entity_raw,
                    "entity_norm": norm_entity_key(entity_raw),
                    "entity_kind": entity_kind,
                    "limit": max(top_k * 5, 50),
                    "text_chars": self.settings.answer_max_text_chars,
                },
            )
        )
        scored_rows = self.attach_route_scores(rows, slots)
        if relations:
            expected = set(relations)
            scored_rows.sort(key=lambda row: (1 if expected & self.row_relations(row) else 0, row.get("_route_score", 0.0), -int(row.get("section_order") or 999999)), reverse=True)
        final_rows = scored_rows[:top_k]
        return {
            "question": question,
            "retrieval_mode": "neo4j_exact_entity_lookup",
            "cypher": EXACT_ENTITY_LOOKUP_CYPHER,
            "qdrant_hits": [],
            "candidate_section_ids": [row.get("section_id") for row in scored_rows if row.get("section_id")],
            "section_ids": [row.get("section_id") for row in final_rows if row.get("section_id")],
            "rows": final_rows,
            "row_count": len(final_rows),
            "answer": "",
            "error": None,
            "debug": {"route": "exact_entity_lookup", "slots": slots, "entity": entity_raw, "entity_kind": entity_kind, "relations": relations, "candidate_count": len(scored_rows), "final_top_k": top_k},
        }

    def ask_multi_entity_qdrant_rerank(self, question: str, slots: Dict[str, Any], top_k: int = 10, search_top_k: int = 50) -> Dict[str, Any]:
        entities = slots.get("entities") or []
        if len(entities) < 2:
            return self.ask_qdrant_neo4j_rerank(question, top_k=top_k, search_top_k=search_top_k)

        per_entity_k = max(10, min(25, search_top_k // max(len(entities), 1)))
        merged_by_sid: Dict[str, Dict[str, Any]] = {}
        for entity in entities[:6]:
            sub_question = f"{entity}. {question}"
            query_vector = self.embedding_service.embed_query(sub_question)
            hits = self.qdrant_store.search_sections(query_vector, top_k=per_entity_k, score_threshold=self.settings.qdrant_score_threshold)
            for hit in hits or []:
                sid = str(hit.get("section_id") or "")
                if not sid:
                    continue
                hit_score = float(hit.get("score") or 0.0)
                if sid not in merged_by_sid:
                    item = dict(hit)
                    item["_sub_entities"] = [entity]
                    item["_sub_query"] = sub_question
                    merged_by_sid[sid] = item
                else:
                    old_score = float(merged_by_sid[sid].get("score") or 0.0)
                    if hit_score > old_score:
                        merged_by_sid[sid].update(dict(hit))
                        merged_by_sid[sid]["_sub_query"] = sub_question
                    merged_by_sid[sid].setdefault("_sub_entities", [])
                    if entity not in merged_by_sid[sid]["_sub_entities"]:
                        merged_by_sid[sid]["_sub_entities"].append(entity)

        qdrant_hits = list(merged_by_sid.values())
        qdrant_hits.sort(key=lambda hit: float(hit.get("score") or 0.0), reverse=True)
        qdrant_hits = qdrant_hits[:search_top_k]
        candidate_section_ids = unique_keep_order([str(hit.get("section_id")) for hit in qdrant_hits if hit.get("section_id")])
        rows = self.enrich_sections_from_neo4j(candidate_section_ids, qdrant_hits)
        reranked_rows = self.rerank_rows(question, rows, qdrant_hits)
        final_rows = reranked_rows[:top_k]
        return {
            "question": question,
            "retrieval_mode": "qdrant_neo4j_multi_entity_rerank",
            "cypher": "",
            "qdrant_hits": qdrant_hits,
            "candidate_section_ids": candidate_section_ids,
            "section_ids": [row.get("section_id") for row in final_rows if row.get("section_id")],
            "rows": final_rows,
            "row_count": len(final_rows),
            "answer": "",
            "error": None,
            "debug": {"route": "multi_entity_search", "slots": slots, "entities": entities, "per_entity_k": per_entity_k, "candidate_count": len(candidate_section_ids), "final_top_k": top_k},
        }

    # ------------------------------------------------------------------
    # Answer and output helpers
    # ------------------------------------------------------------------
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
    def _empty_result(question: str, error: Optional[str] = None, retrieval_mode: str = "cypher") -> Dict[str, Any]:
        return {
            "question": question,
            "answer": "",
            "cypher": "",
            "rows": [],
            "row_count": 0,
            "error": error,
            "retrieval_mode": retrieval_mode,
            "qdrant_hits": [],
            "candidate_section_ids": [],
            "section_ids": [],
            "debug": {},
        }
