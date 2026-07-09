from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.core.config import get_settings
from app.db.neo4j_graph import Neo4jGraphFactory
from app.vector.embedding_service import EmbeddingService
from app.vector.qdrant_store import QdrantSectionStore

SECTION_QUERY = """
MATCH (c:Concept)-[r]->(s:Section)
WHERE s.id IS NOT NULL
  AND s.text IS NOT NULL
  AND type(r) ENDS WITH '_SECTION'
OPTIONAL MATCH (a:Article)-[:HAS_SECTION]->(s)
WITH
  s,
  collect(DISTINCT {
    displayName: c.displayName,
    name: c.name,
    kind: c.kind,
    relation: type(r)
  }) AS concepts,
  collect(DISTINCT a.title) AS articles,
  collect(DISTINCT type(r)) AS relation_types
RETURN
  s.id AS section_id,
  s.heading AS heading,
  s.text AS text,
  concepts AS concepts,
  articles AS articles,
  relation_types AS relation_types
ORDER BY s.id
"""


def chunked(values: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def build_document(row: Dict[str, Any]) -> str:
    concepts = row.get("concepts") or []
    concept_names = []
    for item in concepts:
        if not isinstance(item, dict):
            continue
        name = item.get("displayName") or item.get("name")
        kind = item.get("kind")
        relation = item.get("relation")
        concept_names.append(" | ".join(str(x) for x in [name, kind, relation] if x))
    parts = [
        str(row.get("heading") or ""),
        "\n".join(concept_names),
        str(row.get("text") or ""),
    ]
    return "\n".join(part for part in parts if part).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant Section index from Neo4j sections.")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the Qdrant collection")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", default="qdrant_section_index_report.json")
    args = parser.parse_args()

    settings = get_settings()
    graph = Neo4jGraphFactory(settings).create_graph()
    embedding_service = EmbeddingService(settings)
    qdrant_store = QdrantSectionStore(settings)

    rows = graph.query(SECTION_QUERY)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        report = {"indexed": 0, "collection": settings.qdrant_collection, "message": "No Section rows found from Neo4j."}
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    indexed = 0
    collection_ready = False
    for batch in chunked(rows, args.batch_size):
        docs = [build_document(row) for row in batch]
        vectors = embedding_service.embed_texts(docs)
        if not collection_ready:
            vector_size = len(vectors[0]) if vectors else 0
            qdrant_store.ensure_collection(vector_size=vector_size, reset=args.reset)
            collection_ready = True
        indexed += qdrant_store.upsert_sections(batch, vectors)
        print(f"Indexed {indexed}/{len(rows)} sections")

    report = {
        "indexed": indexed,
        "neo4j_rows": len(rows),
        "collection": settings.qdrant_collection,
        "qdrant_count": qdrant_store.count(),
        "embedding_model_name": settings.embedding_model_name,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
