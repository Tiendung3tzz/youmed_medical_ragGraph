# YouMed RAGGraph Fullstack

Project chatbot y tế tiếng Việt dùng **React + FastAPI + Neo4j + Qdrant + LLM**.

Hệ thống có 2 luồng hỏi đáp:

1. **Cypher mode**: LLM sinh Cypher, truy vấn Neo4j, sau đó LLM viết câu trả lời.
2. **Hybrid mode**: Qdrant semantic search lấy `section_id`, Neo4j enrich node liên quan, sau đó LLM viết câu trả lời.

Hybrid mode là luồng khuyến nghị dùng mặc định vì giảm lỗi LLM sinh sai Cypher.

---

## 1. Kiến trúc tổng quan

```text
User
  ↓
React Chat UI
  ↓
FastAPI Backend
  ↓
Qdrant semantic search
  ↓
section_id
  ↓
Neo4j enrich Section / Concept / Article / Relation
  ↓
LLM answer from evidence rows
```

Luồng hybrid:

```text
Question
  -> Embed question
  -> Search Qdrant collection youmed_sections
  -> Get section_id list
  -> Query Neo4j by section_id
  -> Return evidence rows
  -> LLM generates final Vietnamese answer
```

Luồng Cypher cũ vẫn được giữ lại:

```text
Question
  -> GraphCypherQAChain
  -> LLM-generated Cypher
  -> Neo4j rows
  -> LLM answer
```

---

## 2. Cấu trúc thư mục

```text
youmed_raggraph_fullstack/
  docker-compose.yml
  README.md
  README_QDRANT.md

  backend/
    .env.example
    Dockerfile
    requirements.txt
    data/
      youmed_articles.jsonl
      youmed_graph_test_cases_100.json

    app/
      main.py
      api/
        chat.py
        health.py
      core/
        config.py
        logging.py
      db/
        neo4j_graph.py
      llm/
        factory.py
      prompts/
        cypher_prompt.py
        answer_prompt.py
      schemas/
        chat.py
      services/
        graph_rag_service.py
        cypher_utils.py
        dependencies.py
      vector/
        embedding_service.py
        qdrant_store.py

    tools/
      db_builder/
        run_import.py
        importer.py
        check_db.py
        clear_db.py
        schema.cypher
      qdrant/
        build_section_index.py
        check_qdrant.py
      eval/
        run_graph_eval.py
        youmed_graphrag_evaluator.py

  frontend/
    Dockerfile
    package.json
    nginx.conf
    src/
      App.tsx
      components/
      lib/api.ts
      types/chat.ts
      styles.css
```

---

## 3. Thành phần chính

### 3.1 Frontend

Frontend là React + Vite, giao diện chat đơn giản.

File quan trọng:

```text
frontend/src/App.tsx
frontend/src/lib/api.ts
frontend/src/components/ChatMessage.tsx
frontend/src/types/chat.ts
```

Mặc định UI gọi:

```text
POST /api/chat/hybrid
```

---

### 3.2 Backend

Backend dùng FastAPI.

File chính:

```text
backend/app/services/graph_rag_service.py
```

Service này xử lý:

```text
- ask_graph()              : graph-only Cypher mode
- ask_graph_with_answer()  : Cypher mode + final answer
- ask_hybrid()             : Qdrant -> Neo4j -> LLM answer
```

API chính:

```text
GET  /health
GET  /api/health
POST /api/chat
POST /api/chat/graph
POST /api/chat/hybrid
```

---

### 3.3 Neo4j

Neo4j lưu knowledge graph y tế.

Các node chính:

```text
Article
Section
Concept
Category
HeadingType
ClinicalTerm
```

Các relationship chính:

```text
Article -[:HAS_SECTION]-> Section
Article -[:IN_CATEGORY]-> Category
Article -[:HAS_TOPIC]-> Concept
Concept -[:HAS_OVERVIEW_SECTION]-> Section
Concept -[:HAS_DOSAGE_SECTION]-> Section
Concept -[:HAS_SIDE_EFFECT_SECTION]-> Section
Concept -[:HAS_FUNCTION_SECTION]-> Section
Section -[:HAS_EXTRACTED_TERM]-> ClinicalTerm
Section -[:NEXT_SECTION]-> Section
```

---

### 3.4 Qdrant

Qdrant lưu vector của từng `Section`.

Mỗi point trong Qdrant tương ứng với một `Section` trong Neo4j.

Payload mẫu:

```json
{
  "section_id": "8809366ef038e68e886eedfd759e397284e20dcf",
  "heading": "DNA là gì?",
  "text": "...",
  "article_id": "...",
  "article": "...",
  "category": "...",
  "concepts": [
    {
      "name": "dna",
      "displayName": "DNA",
      "kind": "BodyPart",
      "relation": "HAS_OVERVIEW_SECTION"
    }
  ]
}
```

Qdrant không thay thế Neo4j. Qdrant chỉ tìm semantic section tốt hơn, còn Neo4j vẫn dùng để enrich graph context.

---

## 4. Cấu hình môi trường

Tạo file backend env:

```bash
cd backend
cp .env.example .env
```

Ví dụ `.env` dùng Neo4j Aura + Qdrant local Docker:

```env
APP_ENV=dev
DEBUG=false
FRONTEND_ORIGINS=http://localhost:5173,http://localhost:3000,http://localhost

# Neo4j Aura
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
ENHANCED_SCHEMA=true

# LLM
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GOOGLE_API_KEY=
GOOGLE_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1024

# GraphRAG
GRAPH_TOP_K=10
ANSWER_MAX_ROWS=5
ANSWER_MAX_TEXT_CHARS=900

# Qdrant local Docker
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=youmed_sections
QDRANT_TOP_K=10
QDRANT_SCORE_THRESHOLD=0
QDRANT_TIMEOUT=30
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
```

Nếu chạy backend local ngoài Docker, đổi:

```env
QDRANT_URL=http://localhost:6333
```

Nếu dùng Qdrant Cloud:

```env
QDRANT_URL=https://xxxxx.region.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

Không commit `.env` thật lên Git.

---

## 5. Chạy bằng Docker

Từ root project:

```bash
docker compose down --remove-orphans
docker compose build --no-cache backend frontend
docker compose up backend frontend qdrant
```

Sau khi chạy:

```text
Frontend: http://localhost:3000
Backend:  http://localhost:8000
Qdrant:   http://localhost:6333
```

Kiểm tra backend:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/health
```

Kiểm tra backend đang đọc Neo4j URI nào:

```bash
docker compose exec backend python -c "from app.core.config import get_settings; s=get_settings(); print(s.neo4j_uri); print(s.neo4j_database)"
```

Nếu dùng Neo4j Aura, kết quả phải là dạng:

```text
neo4j+s://xxxxx.databases.neo4j.io
neo4j
```

---

## 6. Chạy local không dùng Docker

### 6.1 Chạy Qdrant local

Dùng Docker chỉ cho Qdrant:

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.6
```

Hoặc dùng Qdrant Cloud và bỏ bước này.

---

### 6.2 Chạy backend

```bash
cd backend
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Cài package:

```bash
pip install -r requirements.txt
```

Chạy API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

### 6.3 Chạy frontend

```bash
cd frontend
npm install
npm run dev
```

Mở:

```text
http://localhost:5173
```

---

## 7. Import dữ liệu vào Neo4j

Nếu Neo4j đã có graph sẵn thì bỏ qua bước này.

### 7.1 Import data mẫu

```bash
cd backend
python -m tools.db_builder.run_import --data data/youmed_articles.jsonl --reset --batch-size 200 --out import_report.json
```

Cảnh báo: `--reset` sẽ xóa dữ liệu graph hiện tại trước khi import.

---

### 7.2 Check graph

```bash
cd backend
python -m tools.db_builder.check_db
```

---

### 7.3 Clear graph

```bash
cd backend
python -m tools.db_builder.clear_db
```

Chỉ chạy khi chắc chắn muốn xóa graph.

---

## 8. Build Qdrant index từ Neo4j

Sau khi Neo4j đã có dữ liệu, build Qdrant collection:

```bash
docker compose exec backend python -m tools.qdrant.build_section_index --reset --report qdrant_section_index_report.json
```

Nếu chạy backend local:

```bash
cd backend
python -m tools.qdrant.build_section_index --reset --report qdrant_section_index_report.json
```

Script này sẽ:

```text
1. Đọc Section từ Neo4j.
2. Lấy thêm Article / Concept / relation.
3. Tạo embedding cho từng Section.
4. Upsert vector + payload vào Qdrant collection.
```

Check Qdrant:

```bash
docker compose exec backend python -m tools.qdrant.check_qdrant
```

Kết quả mong muốn:

```text
Collection: youmed_sections
Points count: > 0
```

---

## 9. Test API

### 9.1 Test hybrid mode

```bash
curl -X POST "http://localhost:8000/api/chat/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"message":"DNA là gì và DNA có chức năng gì?","include_debug":true}'
```

PowerShell:

```powershell
@"
{
  "message": "DNA là gì và DNA có chức năng gì?",
  "include_debug": true
}
"@ | Set-Content -Encoding utf8 .\body.json

curl.exe -X POST "http://localhost:8000/api/chat/hybrid" -H "Content-Type: application/json" --data-binary "@body.json"
```

Response có các field chính:

```json
{
  "answer": "...",
  "retrieval_mode": "qdrant_neo4j",
  "qdrant_hits": [],
  "rows": [],
  "row_count": 0,
  "error": null
}
```

---

### 9.2 Test Cypher mode cũ

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"DNA là gì?","include_debug":true}'
```

Graph-only:

```bash
curl -X POST "http://localhost:8000/api/chat/graph" \
  -H "Content-Type: application/json" \
  -d '{"message":"DNA là gì?","include_debug":true}'
```

---

## 10. Notebook Qdrant hybrid

Notebook `raggraph_qdrant_hybrid.ipynb` dùng để test nhanh pipeline hybrid.

Nếu Neo4j đã có graph sẵn, chỉ cần chạy:

```text
1. Cell install packages.
2. Cell config Neo4j / tạo run_cypher().
3. Cell init LLM nếu muốn sinh answer.
4. Chạy toàn bộ phần 15: Qdrant Hybrid Retrieval.
```

Phần 15 gồm:

```text
15.1 Qdrant config
15.2 Init Qdrant + embedding model
15.3 Lấy Section từ Neo4j
15.4 Build / rebuild Qdrant collection
15.5 Search Qdrant lấy section_id
15.6 Neo4j enrich theo section_id
15.7 ask_hybrid()
15.8 Test nhiều câu hỏi
```

Lần đầu nên test ít dữ liệu:

```python
qdrant_report = build_qdrant_index_from_neo4j(reset=True, limit=100)
```

Sau khi ổn mới build full:

```python
qdrant_report = build_qdrant_index_from_neo4j(reset=True, limit=None)
```

Với version `qdrant-client` mới, nếu lỗi:

```text
AttributeError: 'QdrantClient' object has no attribute 'search'
```

thì đổi `qdrant.search(...)` sang `qdrant.query_points(...)`.

---

## 11. Evaluation

Eval cũ được thiết kế cho Cypher mode.

Chạy graph eval cũ:

```bash
cd backend
python -m tools.eval.run_graph_eval --cases data/youmed_graph_test_cases_100.json --limit 20 --results graph_results.json --report graph_eval_report.json
```

Với hybrid mode, vẫn dùng được các metric retrieval vì result vẫn trả `rows` và `section_id`. Tuy nhiên không nên bắt buộc `pass_cypher`, vì hybrid không dùng LLM-generated Cypher.

Policy chấm nên là:

```text
Cypher mode:
  pass_overall = pass_cypher + pass_retrieval + pass_answer

Hybrid mode:
  pass_overall = pass_retrieval + pass_answer
```

Khi dùng hybrid, cần đảm bảo `rows` có field:

```text
section_id
heading
text
concepts
relations hoặc concepts[].relation
```

---

## 12. Troubleshooting

### 12.1 Backend connect nhầm Neo4j local

Kiểm tra:

```bash
docker compose exec backend python -c "from app.core.config import get_settings; s=get_settings(); print(s.neo4j_uri); print(s.neo4j_database)"
```

Nếu ra:

```text
bolt://neo4j:7687
```

thì backend đang dùng Neo4j Docker local, không phải Aura.

---

### 12.2 Neo4j Aura DNS lỗi

Lỗi:

```text
Failed to DNS resolve address xxxxx.databases.neo4j.io:7687
```

Kiểm tra trên Windows:

```powershell
nslookup xxxxx.databases.neo4j.io
Test-NetConnection xxxxx.databases.neo4j.io -Port 7687
```

Nếu dùng Docker và chỉ container lỗi DNS, thêm vào service backend:

```yaml
dns:
  - 8.8.8.8
  - 1.1.1.1
```

---

### 12.3 Qdrant đã upsert xong nhưng lỗi `vectors_count`

Nếu thấy:

```text
Upserted 5174/5174
AttributeError: 'CollectionInfo' object has no attribute 'vectors_count'
```

Dữ liệu đã ghi xong. Chỉ sửa report:

```python
info = qdrant.get_collection(QDRANT_COLLECTION)
report["points_count"] = getattr(info, "points_count", None)
report["vectors_count"] = report["points_count"]
```

---

### 12.4 Qdrant search lỗi `client.search`

Với `qdrant-client` mới, thay:

```python
qdrant.search(...)
```

bằng:

```python
qdrant.query_points(...)
```

---

### 12.5 API key bị lộ

Nếu đã paste API key hoặc password vào chat/log, cần rotate lại:

```text
- Groq API key
- Qdrant API key
- Neo4j password
```

---

## 13. Quy trình chạy đầy đủ từ đầu

```text
1. Tạo backend/.env.
2. Cấu hình Neo4j Aura hoặc Neo4j local.
3. Cấu hình Groq/Gemini LLM.
4. Cấu hình Qdrant local hoặc Qdrant Cloud.
5. docker compose up backend frontend qdrant.
6. Import data vào Neo4j nếu DB chưa có graph.
7. Check graph bằng tools.db_builder.check_db.
8. Build Qdrant index bằng tools.qdrant.build_section_index.
9. Check Qdrant bằng tools.qdrant.check_qdrant.
10. Test /api/chat/hybrid.
11. Mở frontend.
12. Chạy eval nếu cần.
```

---

## 14. Lệnh nhanh

Chạy Docker:

```bash
docker compose down --remove-orphans
docker compose build --no-cache backend frontend
docker compose up backend frontend qdrant
```

Import Neo4j:

```bash
cd backend
python -m tools.db_builder.run_import --data data/youmed_articles.jsonl --reset --batch-size 200 --out import_report.json
```

Build Qdrant:

```bash
docker compose exec backend python -m tools.qdrant.build_section_index --reset --report qdrant_section_index_report.json
```

Check Qdrant:

```bash
docker compose exec backend python -m tools.qdrant.check_qdrant
```

Test hybrid API:

```bash
curl -X POST "http://localhost:8000/api/chat/hybrid" -H "Content-Type: application/json" -d '{"message":"DNA là gì?","include_debug":true}'
```

Mở UI:

```text
http://localhost:3000
```
