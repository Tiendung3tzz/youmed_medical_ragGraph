from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from neo4j import GraphDatabase

from tools.db_builder.text_utils import (
    category_to_kind,
    heading_type_id,
    infer_section_edge,
    norm_text,
    split_markdown_sections,
    stable_id,
)


@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str = "neo4j"


class YouMedNeo4jImporter:
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))

    def close(self) -> None:
        self.driver.close()

    def apply_schema(self, schema_path: str | Path) -> None:
        text = Path(schema_path).read_text(encoding="utf-8")
        statements = [s.strip() for s in text.split(";") if s.strip()]
        with self.driver.session(database=self.config.database) as session:
            for statement in statements:
                session.run(statement)

    def clear_database(self) -> None:
        with self.driver.session(database=self.config.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def import_jsonl(self, path: str | Path, batch_size: int = 200) -> dict:
        path = Path(path)
        total_articles = 0
        total_sections = 0
        batch: list[dict] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                record = self._build_record(item)
                batch.append(record)
                total_articles += 1
                total_sections += len(record["sections"])
                if len(batch) >= batch_size:
                    self._write_batch(batch)
                    batch = []

        if batch:
            self._write_batch(batch)

        return {"articles": total_articles, "sections": total_sections, "source": str(path)}

    def check_counts(self) -> dict:
        query = """
        CALL {
          MATCH (n:Article) RETURN 'Article' AS label, count(n) AS count
          UNION ALL MATCH (n:Concept) RETURN 'Concept' AS label, count(n) AS count
          UNION ALL MATCH (n:Section) RETURN 'Section' AS label, count(n) AS count
          UNION ALL MATCH (n:Category) RETURN 'Category' AS label, count(n) AS count
          UNION ALL MATCH (n:HeadingType) RETURN 'HeadingType' AS label, count(n) AS count
          UNION ALL MATCH (n:ClinicalTerm) RETURN 'ClinicalTerm' AS label, count(n) AS count
        }
        RETURN label, count
        """
        with self.driver.session(database=self.config.database) as session:
            return {r["label"]: r["count"] for r in session.run(query)}

    def _build_record(self, item: dict) -> dict:
        metadata = item.get("metadata") or {}
        content = item.get("content") or ""
        title = metadata.get("title") or item.get("title") or metadata.get("keyword") or "Untitled"
        url = metadata.get("url") or ""
        category = metadata.get("category") or item.get("category") or ""
        keyword = metadata.get("keyword") or title
        kind = category_to_kind(category)

        article_id = stable_id("article", url or title)
        concept_id = stable_id("concept", kind, keyword)
        category_id = norm_text(category or "unknown") or "unknown"

        raw_sections = split_markdown_sections(content)
        sections = []
        for idx, section in enumerate(raw_sections):
            section_id = stable_id("section", article_id, idx, section["heading"], section["text"][:120])
            rel = infer_section_edge(section["heading"])
            htype = heading_type_id(section["level"], section["heading"])
            sections.append({
                "id": section_id,
                "order": idx,
                "heading": section["heading"],
                "heading_norm": norm_text(section["heading"]),
                "text": section["text"],
                "rel": rel,
                "heading_type_id": htype,
                "heading_type_name": htype.upper(),
            })

        return {
            "article": {
                "id": article_id,
                "title": title,
                "url": url,
                "author": metadata.get("author"),
                "publish_date": metadata.get("publish_date"),
                "category": category,
            },
            "category": {"id": category_id, "name": category},
            "concept": {
                "id": concept_id,
                "name": norm_text(keyword),
                "displayName": keyword,
                "kind": kind,
            },
            "sections": sections,
        }

    def _write_batch(self, batch: list[dict]) -> None:
        with self.driver.session(database=self.config.database) as session:
            session.execute_write(self._tx_write_batch, batch)

    @staticmethod
    def _tx_write_batch(tx, batch: list[dict]) -> None:
        for record in batch:
            tx.run(
                """
                MERGE (a:Article {id: $article.id})
                SET a += $article
                MERGE (c:Category {id: $category.id})
                SET c += $category
                MERGE (k:Concept {id: $concept.id})
                SET k += $concept
                MERGE (a)-[:IN_CATEGORY]->(c)
                MERGE (a)-[:HAS_TOPIC]->(k)
                MERGE (k)-[:HAS_ARTICLE]->(a)
                """,
                article=record["article"], category=record["category"], concept=record["concept"]
            )

            prev_section_id = None
            for section in record["sections"]:
                rel = section["rel"]
                if not rel.isidentifier() or not rel.isupper():
                    rel = "HAS_OVERVIEW_SECTION"
                tx.run(
                    f"""
                    MATCH (a:Article {{id: $article_id}})
                    MATCH (k:Concept {{id: $concept_id}})
                    MERGE (s:Section {{id: $section.id}})
                    SET s += $section
                    MERGE (a)-[:HAS_SECTION]->(s)
                    MERGE (k)-[:{rel}]->(s)
                    MERGE (s)-[:MENTIONS]->(k)
                    MERGE (a)-[:MENTIONS]->(k)
                    MERGE (h:HeadingType {{id: $section.heading_type_id}})
                    SET h.name = $section.heading_type_name
                    MERGE (s)-[:HAS_HEADING_TYPE]->(h)
                    """,
                    article_id=record["article"]["id"], concept_id=record["concept"]["id"], section=section
                )
                if prev_section_id:
                    tx.run(
                        """
                        MATCH (p:Section {id: $prev_id})
                        MATCH (s:Section {id: $section_id})
                        MERGE (p)-[:NEXT_SECTION]->(s)
                        """,
                        prev_id=prev_section_id, section_id=section["id"]
                    )
                prev_section_id = section["id"]
