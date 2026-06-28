import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from tools.db_builder.importer import Neo4jConfig, YouMedNeo4jImporter


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Import YouMed JSONL into Neo4j graph.")
    parser.add_argument("--data", required=True, help="Path to YouMed JSONL file")
    parser.add_argument("--schema", default="tools/db_builder/schema.cypher")
    parser.add_argument("--reset", action="store_true", help="Delete all nodes before import")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--out", default="import_report.json")
    args = parser.parse_args()

    config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    importer = YouMedNeo4jImporter(config)
    try:
        importer.apply_schema(args.schema)
        if args.reset:
            importer.clear_database()
            importer.apply_schema(args.schema)
        report = importer.import_jsonl(args.data, batch_size=args.batch_size)
        report["counts"] = importer.check_counts()
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        importer.close()


if __name__ == "__main__":
    main()
