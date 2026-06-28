import re
import unicodedata
from typing import Any, Dict, List

VALID_CONCEPT_KIND_VALUES = {"Drug", "Disease", "BodyPart", "TraditionalMedicine"}
INVALID_CONCEPT_KIND_VALUES = {"Concept", "Section", "Article", "ClinicalTerm"}
FORBIDDEN_WRITE_PATTERN = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP)\b|CALL\s+(dbms|apoc)", re.IGNORECASE)


def clean_cypher(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:cypher)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\b(MATCH|WITH|RETURN)\b", text, flags=re.IGNORECASE)
    return text[match.start():].strip() if match else text


def extract_cypher(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    for step in result.get("intermediate_steps") or []:
        if isinstance(step, dict):
            if step.get("query"):
                return clean_cypher(step["query"])
            if step.get("cypher"):
                return clean_cypher(step["cypher"])
    for key in ["cypher", "query"]:
        if result.get(key):
            return clean_cypher(result[key])
    return ""


def normalize_rows(rows: Any) -> List[dict]:
    if rows is None:
        return []
    if isinstance(rows, list):
        return [r if isinstance(r, dict) else {"value": r} for r in rows]
    if isinstance(rows, dict):
        return [rows]
    return [{"value": rows}]


def norm_for_rule(text: str) -> str:
    text = str(text or "")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_kind_from_question(question: str) -> str | None:
    q = norm_for_rule(question)
    traditional_terms = ["duoc lieu", "cay thuoc", "vi thuoc", "thuoc nam", "thuoc dong y", "bao che", "che bien", "doc tinh duoc lieu"]
    body_part_terms = ["bo phan", "co quan", "mo", "te bao", "cau truc", "giai phau", "chuc nang", "sinh ly", "vi tri", "cau tao"]
    drug_terms = ["thuoc", "vien uong", "san pham", "hoat chat", "lieu dung", "cach dung", "tac dung phu", "tuong tac", "bao quan", "chong chi dinh", "quen lieu", "qua lieu"]
    disease_terms = ["benh", "hoi chung", "trieu chung", "bieu hien", "nguyen nhan", "yeu to nguy co", "chan doan", "xet nghiem", "dieu tri", "bien chung", "phong ngua"]
    if any(t in q for t in traditional_terms):
        return "TraditionalMedicine"
    if any(t in q for t in body_part_terms):
        return "BodyPart"
    if any(t in q for t in drug_terms):
        return "Drug"
    if any(t in q for t in disease_terms):
        return "Disease"
    return None


def has_invalid_kind(cypher: str) -> bool:
    return bool(re.search(r"\b\w+\.kind\s*=\s*['\"](?:Concept|Section|Article|ClinicalTerm)['\"]", cypher or "", flags=re.IGNORECASE))


def repair_invalid_kind(cypher: str, question: str) -> str:
    if not has_invalid_kind(cypher):
        return cypher
    inferred_kind = infer_kind_from_question(question)
    if not inferred_kind:
        return cypher

    def repl(match: re.Match) -> str:
        alias = match.group(1)
        return f"{alias}.kind = '{inferred_kind}'"

    return re.sub(r"\b(\w+)\.kind\s*=\s*['\"](?:Concept|Section|Article|ClinicalTerm)['\"]", repl, cypher, flags=re.IGNORECASE)


def assert_readonly(cypher: str) -> None:
    if FORBIDDEN_WRITE_PATTERN.search(cypher or ""):
        raise ValueError("Blocked non-read-only Cypher.")
