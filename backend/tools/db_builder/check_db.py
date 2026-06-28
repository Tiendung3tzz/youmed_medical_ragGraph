import json
import os

from dotenv import load_dotenv

from tools.db_builder.importer import Neo4jConfig, YouMedNeo4jImporter


def main() -> None:
    load_dotenv()
    importer = YouMedNeo4jImporter(Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    ))
    try:
        print(json.dumps(importer.check_counts(), ensure_ascii=False, indent=2))
    finally:
        importer.close()


if __name__ == "__main__":
    main()
