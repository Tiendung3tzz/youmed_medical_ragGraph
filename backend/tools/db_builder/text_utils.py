import hashlib
import re
import unicodedata


def stable_id(*parts: object) -> str:
    raw = "|".join(str(p or "") for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def norm_text(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9\s_:/.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_markdown_sections(content: str) -> list[dict]:
    sections: list[dict] = []
    current_heading = "Tổng quan"
    current_level = 2
    buffer: list[str] = []

    def flush():
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if text:
            sections.append({"heading": current_heading, "level": current_level, "text": text})
        buffer = []

    for line in str(content or "").splitlines():
        m = re.match(r"^(#{2,6})\s+(.+?)\s*$", line)
        if m:
            flush()
            current_level = len(m.group(1))
            current_heading = m.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return sections


def category_to_kind(category: str) -> str:
    c = norm_text(category)
    if "drug" in c or "thuoc" in c or "product" in c:
        return "Drug"
    if "disease" in c or "benh" in c:
        return "Disease"
    if "traditional" in c or "duoc" in c or "medicine" in c or "cay" in c:
        return "TraditionalMedicine"
    if "body" in c or "part" in c or "co the" in c:
        return "BodyPart"
    return "Disease"


def infer_section_edge(heading: str) -> str:
    h = norm_text(heading)
    rules = [
        (["tac dung phu", "khong mong muon"], "HAS_SIDE_EFFECT_SECTION"),
        (["lieu", "cach dung", "su dung"], "HAS_DOSAGE_SECTION"),
        (["chong chi dinh", "khong nen dung"], "HAS_CONTRAINDICATION_SECTION"),
        (["phu nu", "tre em", "mang thai", "cho con bu", "doi tuong"], "HAS_SPECIAL_POPULATION_SECTION"),
        (["tuong tac"], "HAS_INTERACTION_SECTION"),
        (["cong dung", "duoc dung", "chi dinh"], "HAS_USE_SECTION"),
        (["luu y", "than trong", "canh bao", "kieng ky"], "HAS_CAUTION_SECTION"),
        (["bao quan"], "HAS_STORAGE_SECTION"),
        (["quen lieu", "qua lieu"], "HAS_MISSED_OR_OVERDOSE_SECTION"),
        (["thanh phan", "hoat chat"], "HAS_COMPOSITION_SECTION"),
        (["trieu chung", "bieu hien", "dau hieu"], "HAS_SYMPTOM_SECTION"),
        (["nguyen nhan", "yeu to nguy co", "nguy co"], "HAS_CAUSE_OR_RISK_SECTION"),
        (["chan doan", "xet nghiem", "kiem tra"], "HAS_DIAGNOSIS_SECTION"),
        (["dieu tri", "tri benh", "chua"], "HAS_TREATMENT_SECTION"),
        (["bien chung", "hau qua"], "HAS_COMPLICATION_SECTION"),
        (["phong ngua", "du phong"], "HAS_PREVENTION_SECTION"),
        (["sinh ly", "co che", "qua trinh"], "HAS_PHYSIOLOGY_SECTION"),
        (["cau tao", "giai phau", "vi tri", "cau truc"], "HAS_ANATOMY_SECTION"),
        (["chuc nang", "vai tro"], "HAS_FUNCTION_SECTION"),
        (["mo ta", "dac diem tu nhien"], "HAS_BOTANICAL_DESCRIPTION_SECTION"),
        (["bao che", "che bien", "bo phan dung"], "HAS_PREPARATION_SECTION"),
        (["doc tinh"], "HAS_TOXICITY_SECTION"),
        (["ten goi", "ten khac"], "HAS_NAME_SECTION"),
    ]
    for keywords, rel in rules:
        if any(k in h for k in keywords):
            return rel
    return "HAS_OVERVIEW_SECTION"


def heading_type_id(level: int, heading: str) -> str:
    if level <= 2:
        return "h2"
    if level == 3:
        return "h3"
    return f"h{level}"
