CREATE CONSTRAINT article_id IF NOT EXISTS FOR (n:Article) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT category_id IF NOT EXISTS FOR (n:Category) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT heading_type_id IF NOT EXISTS FOR (n:HeadingType) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT clinical_term_id IF NOT EXISTS FOR (n:ClinicalTerm) REQUIRE n.id IS UNIQUE;

CREATE INDEX concept_name IF NOT EXISTS FOR (n:Concept) ON (n.name);
CREATE INDEX concept_display_name IF NOT EXISTS FOR (n:Concept) ON (n.displayName);
CREATE INDEX concept_kind IF NOT EXISTS FOR (n:Concept) ON (n.kind);
CREATE INDEX section_heading IF NOT EXISTS FOR (n:Section) ON (n.heading);
CREATE INDEX section_heading_norm IF NOT EXISTS FOR (n:Section) ON (n.heading_norm);
CREATE INDEX clinical_term_name IF NOT EXISTS FOR (n:ClinicalTerm) ON (n.name);
