from langchain_neo4j import Neo4jGraph

from app.core.config import Settings


class Neo4jGraphFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create_graph(self) -> Neo4jGraph:
        graph = Neo4jGraph(
            url=self.settings.neo4j_uri,
            username=self.settings.neo4j_username,
            password=self.settings.neo4j_password,
            database=self.settings.neo4j_database,
            enhanced_schema=self.settings.enhanced_schema,
        )
        graph.refresh_schema()
        return graph
