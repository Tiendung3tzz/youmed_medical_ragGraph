# YouMed RAGGraph Fullstack

Ứng dụng chatbot GraphRAG y tế tiếng Việt sử dụng **React + FastAPI + Neo4j + LLM**. Giao diện frontend mô phỏng cách dùng của ChatGPT, backend nhận câu hỏi tiếng Việt, sinh Cypher, truy vấn Neo4j và tạo câu trả lời dựa trên dữ liệu graph.

Project này được tách thành ba phần rõ ràng:

1. **Runtime chat app**: frontend React và backend FastAPI để người dùng hỏi đáp.
2. **DB builder**: code tạo schema và import dữ liệu vào Neo4j, chạy riêng khi cần build lại graph.
3. **Evaluation**: code đánh giá Cypher, retrieval, ranking và answer, chạy riêng trong VS Code hoặc terminal.

---

## 1. Kiến trúc tổng quan

```text
User
  |
  v
React Chat UI
  |
  | POST /api/chat
  v
FastAPI Backend
  |
  | 1. LLM sinh Cypher từ câu hỏi
  | 2. Validate Cypher read-only
  | 3. Query Neo4j
  | 4. LLM tổng hợp answer từ rows
  v
Neo4j / Neo4j Aura
```

Luồng chính:

```text
Question -> GraphCypherQAChain -> Cypher -> Neo4j rows -> Answer prompt -> Final answer
```

Backend không hardcode API key. Tất cả key và connection string lấy từ `.env`.

---

## 2. Cấu trúc thư mục

```text
youmed_raggraph_fullstack/
  docker-compose.yml
  README.md
  README_HUONG_DAN.md

  frontend/
    src/
      App.tsx
      components/
      lib/api.ts
      types/chat.ts
      styles.css
    package.json
    Dockerfile
    nginx.conf
    .env.example

  backend/
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

    tools/
      db_builder/
        importer.py
        run_import.py
        check_db.py
        clear_db.py
        schema.cypher
        text_utils.py
      eval/
        youmed_graphrag_evaluator.py
        run_graph_eval.py

    data/
      sample_youmed_articles.jsonl

    requirements.txt
    Dockerfile
    .env.example
```

---

## 3. Thành phần chính

### 3.1 Frontend React

Frontend là giao diện chat:

- Sidebar danh sách hội thoại.
- Khung chat user / assistant.
- Ô nhập câu hỏi ở dưới.
- Loading typing dots.
- Hiển thị answer.
- Có thể mở rộng để xem Cypher và evidence rows.

File chính:

```text
frontend/src/App.tsx
frontend/src/components/ChatMessage.tsx
frontend/src/components/ChatInput.tsx
frontend/src/lib/api.ts
```

Frontend gọi backend qua biến môi trường:

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

### 3.2 Backend FastAPI

Backend cung cấp API chat:

```http
POST /api/chat
POST /api/chat/graph
GET  /health
```

Class chính:

```python
YouMedGraphRAGService
```

File:

```text
backend/app/services/graph_rag_service.py
```

Nhiệm vụ:

1. Nhận câu hỏi.
2. Sinh Cypher bằng LLM.
3. Làm sạch và validate Cypher.
4. Chạy query Neo4j.
5. Tạo answer cuối từ rows.
6. Trả về answer, cypher, rows, row_count, error.

---

### 3.3 Neo4j / Neo4j Aura

Project chạy được với cả Neo4j local và Neo4j Aura.

Với Neo4j Aura, dùng URI dạng:

```env
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_aura_password
NEO4J_DATABASE=neo4j
```

Với Neo4j local trong Docker:

```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

---

### 3.4 DB Builder

Phần tạo DB được tách riêng trong:

```text
backend/tools/db_builder/
```

Các file chính:

```text
importer.py       # class import dữ liệu JSONL vào Neo4j
run_import.py     # script chạy import
check_db.py       # script kiểm tra số lượng node/relationship
clear_db.py       # script xóa dữ liệu graph
schema.cypher     # constraint/index/schema gợi ý
```

Phần này không bắt buộc chạy khi start app. Chỉ dùng khi muốn build lại graph hoặc nạp thêm dữ liệu.

---

### 3.5 Evaluation

Phần eval được tách riêng trong:

```text
backend/tools/eval/
```

Các file chính:

```text
youmed_graphrag_evaluator.py
run_graph_eval.py
```

Evaluator dùng để chấm:

- Cypher có sinh đúng không.
- Query có read-only không.
- Có dùng đúng relationship không.
- Rows có trả đúng `section_id` gold không.
- Có hit@k, recall@k, MRR, MAP, NDCG.
- Có chấm answer bằng string check, ROUGE, token F1 hoặc LLM judge.

Phần này chỉ phục vụ kiểm thử, không cần đưa vào luồng runtime chat.

---

## 4. Cấu hình môi trường backend

Tạo file `.env` từ mẫu:

```bash
cd backend
cp .env.example .env
```

Ví dụ cấu hình dùng Groq + Neo4j Aura:

```env
APP_NAME=YouMed RAGGraph API
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

FRONTEND_ORIGINS=http://localhost:5173,http://localhost:3000

LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
LLM_TEMPERATURE=0

NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j

GRAPH_TOP_K=10
RETURN_DIRECT=true
```

Không commit `.env` lên Git.

---

## 5. Chạy local không dùng Docker

### 5.1 Chạy backend

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

Kiểm tra:

```bash
curl http://localhost:8000/health
```

---

### 5.2 Chạy frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Mở trình duyệt:

```text
http://localhost:5173
```

---

## 6. Chạy bằng Docker

Tạo `.env` cho backend trước:

```bash
cd backend
cp .env.example .env
```

Điền key và Neo4j connection vào `.env`.

Quay lại root project:

```bash
cd ..
docker compose up --build
```

Sau khi chạy:

```text
Frontend: http://localhost:5173
Backend:  http://localhost:8000
Neo4j:    http://localhost:7474 nếu dùng Neo4j local container
```

Nếu dùng Neo4j Aura, không cần chạy service Neo4j local. Có thể chạy riêng:

```bash
docker compose up --build backend frontend
```

---

## 7. Tạo hoặc import dữ liệu Neo4j

### 7.1 Import sample data

```bash
cd backend
python -m tools.db_builder.run_import \
  --data data/sample_youmed_articles.jsonl \
  --reset \
  --out import_report.json
```

`--reset` sẽ xóa dữ liệu graph hiện tại trước khi import. Không dùng trên database thật nếu chưa backup.

---

### 7.2 Import file JSONL thật

```bash
cd backend
python -m tools.db_builder.run_import \
  --data /path/to/youmed_articles.jsonl \
  --batch-size 500 \
  --out import_report.json
```

Nếu muốn xóa dữ liệu cũ trước khi import:

```bash
python -m tools.db_builder.run_import \
  --data /path/to/youmed_articles.jsonl \
  --reset \
  --batch-size 500 \
  --out import_report.json
```

---

### 7.3 Check dữ liệu

```bash
cd backend
python -m tools.db_builder.check_db
```

Script sẽ in số lượng node/relationship để kiểm tra import có thành công không.

---

### 7.4 Clear dữ liệu

```bash
cd backend
python -m tools.db_builder.clear_db
```

Chỉ chạy khi chắc chắn muốn xóa graph.

---

## 8. Chạy eval

Chuẩn bị file testcase, ví dụ:

```text
youmed_graph_eval_cases_100.json
```

Chạy graph eval:

```bash
cd backend
python -m tools.eval.run_graph_eval \
  --cases /path/to/youmed_graph_eval_cases_100.json \
  --limit 20 \
  --results graph_results_20.json \
  --report graph_eval_report_20.json
```

Kết quả gồm:

```text
pass_overall_rate
pass_cypher_rate
pass_retrieval_rate
pass_answer_rate
hit_at_1 / hit_at_3 / hit_at_5 / hit_at_10
recall_at_1 / recall_at_3 / recall_at_5 / recall_at_10
```

Lưu ý:

- `run_graph_eval` chỉ đánh giá sinh Cypher và retrieval graph.
- Nếu muốn đánh giá answer cuối, cần chạy luồng answer riêng và dùng quota LLM đủ lớn.
- Nếu dùng Groq free/on-demand, dễ gặp `RateLimitError 429` khi chạy nhiều case vì prompt Cypher dài.

---

## 9. API contract

### 9.1 Chat API

Request:

```http
POST /api/chat
Content-Type: application/json
```

Body:

```json
{
  "message": "Unasyn có tác dụng phụ nào?",
  "session_id": "optional-session-id"
}
```

Response:

```json
{
  "answer": "...",
  "cypher": "MATCH ...",
  "rows": [],
  "row_count": 0,
  "error": null
}
```

---

### 9.2 Graph-only API

Request:

```http
POST /api/chat/graph
Content-Type: application/json
```

Body:

```json
{
  "message": "DNA được mô tả tổng quan như thế nào?"
}
```

Response tương tự `/api/chat`, nhưng có thể không sinh answer cuối tùy cấu hình backend.

---

## 10. Quy tắc an toàn Cypher

Backend có lớp kiểm tra Cypher trước khi chạy:

Không cho phép:

```text
CREATE
MERGE
SET
DELETE
REMOVE
DROP
CALL dbms
CALL apoc
```

Mục tiêu là chỉ cho phép truy vấn read-only.

Nếu cần mở rộng sau này, nên dùng whitelist query template thay vì để LLM sinh Cypher tự do.

---

## 11. Lưu ý khi dùng Neo4j Aura

Dùng URI dạng:

```env
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
```

Các lỗi thường gặp:

```text
1. Sai password Aura.
2. Aura instance đang paused.
3. Dùng nhầm bolt://localhost:7687 trong Docker.
4. Backend container không đọc đúng backend/.env.
5. Chạy --reset nhầm vào Aura database thật.
```

Test kết nối nhanh:

```python
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))
driver.verify_connectivity()
print("Connected to Neo4j")
driver.close()
```

---

## 12. Lưu ý khi dùng Groq

Nếu chạy nhiều eval case hoặc prompt quá dài, có thể gặp:

```text
RateLimitError 429
TPD Limit exceeded
```

Cách xử lý:

```text
1. Chạy eval theo batch nhỏ.
2. Tăng sleep_seconds giữa các case.
3. Giảm độ dài prompt Cypher.
4. Đổi model có quota cao hơn.
5. Chờ quota reset.
```

---

## 13. Quy trình đề xuất khi phát triển

```text
1. Import dữ liệu vào Neo4j.
2. Check DB bằng tools.db_builder.check_db.
3. Chạy backend local.
4. Test /api/chat/graph với vài câu đơn giản.
5. Chạy frontend.
6. Chạy eval 20 case easy.
7. Sửa prompt / evaluator nếu fail.
8. Chạy tiếp medium / hard / very_hard.
9. Build Docker.
10. Deploy dùng Neo4j Aura hoặc Neo4j managed riêng.
```

---

## 14. Ghi chú triển khai thực tế

Đây là project mẫu theo chuẩn class/service để dễ tách phần runtime, tạo DB và eval. Khi đưa vào production nên bổ sung:

```text
- Authentication cho API.
- Logging request/response có masking dữ liệu nhạy cảm.
- Rate limit theo user/session.
- Cache schema Neo4j.
- Streaming answer qua SSE/WebSocket.
- Lưu lịch sử hội thoại.
- Giới hạn số row và số ký tự gửi vào answer prompt.
- Monitoring lỗi LLM và lỗi Neo4j.
```

---

## 15. Tóm tắt nhanh lệnh cần nhớ

Chạy backend:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Chạy frontend:

```bash
cd frontend
npm run dev
```

Chạy Docker:

```bash
docker compose up --build
```

Import DB:

```bash
cd backend
python -m tools.db_builder.run_import --data data/sample_youmed_articles.jsonl --reset
```

Check DB:

```bash
cd backend
python -m tools.db_builder.check_db
```

Chạy eval:

```bash
cd backend
python -m tools.eval.run_graph_eval --cases /path/to/cases.json --limit 20
```
