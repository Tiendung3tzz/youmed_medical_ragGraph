# YouMed Medical RAG Graph

Ứng dụng chatbot y khoa tiếng Việt dùng kiến trúc GraphRAG kết hợp:

- Frontend: React + Vite
- Backend: FastAPI
- Vector Database: Qdrant Cloud
- Graph Database: Neo4j Aura
- Retrieval: Qdrant + Neo4j enrichment + rerank
- LLM: Groq hoặc Google Gemini
- Deploy:
  - Frontend: Vercel
  - Backend: Railway

---

## 1. Kiến trúc tổng quan

Luồng production hiện tại:

```text
User
→ Frontend Vercel
→ Backend FastAPI Railway
→ Router retrieval
→ Qdrant vector search
→ Neo4j enrich section/article/concept
→ Rerank / scoring
→ LLM generate answer
→ Frontend hiển thị kết quả
```

Các service bên ngoài:

```text
Neo4j Aura
→ lưu graph Article, Section, Concept, Category, ClinicalTerm

Qdrant Cloud
→ lưu vector của Section

Groq / Google Gemini
→ sinh câu trả lời cuối
```

---

## 2. Luồng GraphRAG mới

Backend không còn chỉ chạy một kiểu retrieval duy nhất. Luồng mới dùng routed retrieval.

```text
Question
→ Rule detector
→ Optional LLM intent extraction
→ Route dispatcher
→ Retrieval theo route
→ Neo4j enrichment
→ Rerank
→ LLM answer
```

### 2.1 Các route retrieval

#### `heading_lookup`

Dùng khi câu hỏi hỏi theo tiêu đề/mục/phần.

Ví dụ:

```text
Những bệnh nào có phần tiêu đề "Chẩn đoán"?
Những dược liệu nào có mục "Y học hiện đại"?
```

Luồng xử lý:

```text
Question
→ detect heading
→ Neo4j exact lookup Section.heading
→ trả về các Section khớp heading
```

Không cần gọi Qdrant nếu lookup ra kết quả.

---

#### `exact_entity_lookup`

Dùng khi câu hỏi hỏi đúng một thực thể cụ thể.

Ví dụ:

```text
Vastarel có tác dụng phụ gì?
Giác mạc có chức năng gì?
```

Luồng xử lý:

```text
Question
→ extract entity
→ Neo4j lookup Concept.name / Concept.displayName
→ lấy các Section liên quan
→ nếu không có kết quả thì fallback sang Qdrant semantic search
```

---

#### `multi_entity_search`

Dùng khi câu hỏi có nhiều thực thể hoặc yêu cầu so sánh/tổng hợp.

Ví dụ:

```text
So sánh tác dụng phụ của Vastarel, Viartril-S và Varogel.
```

Luồng xử lý:

```text
Question
→ extract multiple entities
→ tạo sub-query theo từng entity
→ Qdrant search từng sub-query
→ merge section_id
→ Neo4j enrich
→ rerank
```

---

#### `semantic_search`

Dùng cho câu hỏi ngữ nghĩa thông thường.

Ví dụ:

```text
Thuốc điều trị đau thắt ngực có lưu ý gì?
```

Luồng xử lý:

```text
Question
→ embedding model
→ Qdrant top N sections
→ lấy section_id
→ Neo4j enrich Article/Concept/Category/ClinicalTerm
→ rerank
→ LLM answer
```

---

## 3. Embedding và Qdrant

Qdrant collection hiện tại:

```env
QDRANT_COLLECTION=youmed_sections
```

Collection này đang dùng vector dimension:

```text
1024
```

Do đó backend phải dùng embedding model trả vector 1024 dim.

Model đang dùng:

```env
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

## 4. evaluate
```json
    "total": 100,
    "pass_overall_rate": 0.99,
    "pass_cypher_rate": 1.0,
    "pass_retrieval_rate": 0.99,
    "pass_answer_rate": 1.0,
    "avg_hit_at_10": 0.99,
    "avg_recall_at_10": 0.8613333333333333,
    "avg_precision_at_10": 0.40452380952380956,
    "avg_mrr": 0.8536666666666666,
    "avg_map": 0.7427501322751323,
    "avg_rank_score_at_10": 0.9570000000000001,
    "avg_first_relevant_rank": 1.3333333333333333,
    "avg_qdrant_mrr": 0.6010785510785511,
    "avg_qdrant_rank_score_at_10": 0.7826666666666667,
```
## 5. Backend local setup

Đi vào thư mục backend:

```bash
cd backend
```

Tạo virtual environment:

```bash
python -m venv .venv
```

Kích hoạt môi trường:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Cài package:

```bash
pip install -r requirements.txt
```

Chạy backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test health:

```bash
curl http://localhost:8000/health
```

Hoặc:

```bash
curl http://localhost:8000/api/health
```

---

## 6. Frontend local setup

Đi vào thư mục frontend:

```bash
cd frontend
```

Cài package:

```bash
npm install
```

Tạo file `.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Chạy frontend:

```bash
npm run dev
```

Frontend local mặc định chạy ở:

```text
http://localhost:5173
```

---

## 7. Environment Variables backend

Backend dùng các biến môi trường sau.

### App

```env
APP_ENV=dev
DEBUG=false
FRONTEND_ORIGINS=http://localhost:5173,http://localhost:3000
```

Production trên Railway:

```env
APP_ENV=prod
DEBUG=false
FRONTEND_ORIGINS=https://your-frontend.vercel.app
```

Nếu muốn cho cả local và Vercel:

```env
FRONTEND_ORIGINS=http://localhost:5173,http://localhost:3000,https://your-frontend.vercel.app
```

Không thêm dấu `/` cuối domain.

Sai:

```env
FRONTEND_ORIGINS=https://your-frontend.vercel.app/
```

Đúng:

```env
FRONTEND_ORIGINS=https://your-frontend.vercel.app
```

---

### Neo4j Aura

```env
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=xxxxx
NEO4J_DATABASE=neo4j
```

Nếu dùng database name riêng trên Aura thì set đúng database name tương ứng.

---

### LLM

Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=xxxxx
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1024
```

Google Gemini:

```env
LLM_PROVIDER=google
GOOGLE_API_KEY=xxxxx
GOOGLE_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1024
```

---

### Qdrant

```env
QDRANT_URL=https://xxxxx.cloud.qdrant.io
QDRANT_API_KEY=xxxxx
QDRANT_COLLECTION=youmed_sections
QDRANT_TOP_K=10
QDRANT_SEARCH_TOP_K=50
QDRANT_SCORE_THRESHOLD=0
QDRANT_TIMEOUT=30
```

---

### Embedding

Nếu dùng collection hiện tại:

```env
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

Nếu đổi model nhỏ hơn:

```env
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
QDRANT_COLLECTION=youmed_sections_e5_small
```

Lưu ý: đổi embedding model thì phải rebuild Qdrant collection.

---

### GraphRAG

```env
GRAPH_TOP_K=10
ANSWER_MAX_ROWS=5
ANSWER_MAX_TEXT_CHARS=900
ENHANCED_SCHEMA=false
```

Khuyến nghị production:

```env
ENHANCED_SCHEMA=false
```

Vì `ENHANCED_SCHEMA=true` có thể làm LangChain Neo4j scan schema nhiều và sinh log rất dài.

---

### Router / reranker

```env
USE_LLM_INTENT_ROUTER=true
INTENT_CONFIDENCE_THRESHOLD=0.5

RERANKER_ENABLED=false
RERANKER_MODEL_NAME=BAAI/bge-reranker-base
RERANKER_DEVICE=
RERANKER_MAX_LENGTH=512
RERANKER_MAX_CHARS=900
RERANKER_BATCH_SIZE=4
```

Khuyến nghị deploy Railway lần đầu:

```env
RERANKER_ENABLED=false
```

Sau khi backend chạy ổn mới bật reranker.

---

## 8. Deploy backend lên Railway

### 8.1 Dockerfile backend

Backend cần chạy theo biến `PORT` của Railway.

File:

```text
backend/Dockerfile
```

Nội dung khuyến nghị:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tools ./tools

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

### 8.2 Deploy Railway

```text
Railway
→ New Project
→ Deploy from GitHub repo
→ chọn repo
→ Root Directory: backend
→ Railway detect Dockerfile
→ Add Variables
→ Deploy
```

Sau deploy:

```text
Settings / Networking
→ Generate Domain
```

Backend URL sẽ có dạng:

```text
https://your-backend.up.railway.app
```

Test:

```bash
curl https://your-backend.up.railway.app/health
```

Hoặc:

```bash
curl https://your-backend.up.railway.app/api/health
```

---

## 9. Deploy frontend lên Vercel

Frontend nằm trong thư mục:

```text
frontend
```

Cấu hình Vercel:

```text
Root Directory: frontend
Framework Preset: Vite
Install Command: npm install
Build Command: npm run build
Output Directory: dist
```

Vercel Environment Variable:

```env
VITE_API_BASE_URL=https://your-backend.up.railway.app
```

Sau khi set env, cần redeploy frontend.

Frontend sẽ gọi backend qua:

```text
VITE_API_BASE_URL + /api/chat/hybrid
```

Ví dụ:

```text
https://your-backend.up.railway.app/api/chat/hybrid
```

---

## 10. CORS

Backend CORS nằm trong:

```text
backend/app/main.py
```

Cấu hình:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins_list,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Railway cần set:

```env
FRONTEND_ORIGINS=https://your-frontend.vercel.app
```

Nếu muốn thêm local:

```env
FRONTEND_ORIGINS=http://localhost:5173,http://localhost:3000,https://your-frontend.vercel.app
```

Test preflight:

```bash
curl -i -X OPTIONS "https://your-backend.up.railway.app/api/chat/hybrid" \
  -H "Origin: https://your-frontend.vercel.app" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type"
```

Kết quả đúng phải có:

```text
access-control-allow-origin: https://your-frontend.vercel.app
```

---

## 11. API endpoints

### Health

```http
GET /health
GET /api/health
```

### Hybrid chat

```http
POST /api/chat/hybrid
```

Request body:

```json
{
  "message": "Vastarel có tác dụng phụ gì?",
  "include_debug": true
}
```

Response mẫu:

```json
{
  "question": "Vastarel có tác dụng phụ gì?",
  "answer": "...",
  "cypher": "",
  "rows": [],
  "row_count": 0,
  "error": null,
  "retrieval_mode": "qdrant_neo4j_st_bge_rerank",
  "qdrant_hits": [],
  "candidate_section_ids": [],
  "section_ids": [],
  "debug": {}
}
```

Nếu `include_debug=false`, backend có thể ẩn bớt:

```text
qdrant_hits
candidate_section_ids
section_ids
debug
```

---

## 12. Test API bằng PowerShell

Tạo file body:

```powershell
New-Item -ItemType Directory -Force C:\Temp

@'
{
  "message": "Vastarel có tác dụng phụ gì?",
  "include_debug": true
}
'@ | Set-Content -Encoding UTF8 C:\Temp\body.json
```

Gọi API:

```powershell
curl.exe -i -X POST "https://your-backend.up.railway.app/api/chat/hybrid" `
  -H "Origin: https://your-frontend.vercel.app" `
  -H "Content-Type: application/json" `
  --data-binary "@C:\Temp\body.json"
```

---

## 13. Các lỗi thường gặp

### 13.1 CORS error

Lỗi:

```text
No Access-Control-Allow-Origin header is present
```

Kiểm tra Railway env:

```env
FRONTEND_ORIGINS=https://your-frontend.vercel.app
```

Không được nhập sai kiểu:

```env
https://https://your-frontend.vercel.app
```

Test:

```bash
curl -i -X OPTIONS "https://your-backend.up.railway.app/api/chat/hybrid" \
  -H "Origin: https://your-frontend.vercel.app" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type"
```

---

### 13.2 Frontend vẫn gọi localhost

Kiểm tra Vercel env:

```env
VITE_API_BASE_URL=https://your-backend.up.railway.app
```

Sau khi sửa env phải redeploy frontend.

Trong DevTools Network, request đúng phải là:

```text
https://your-backend.up.railway.app/api/chat/hybrid
```

Không được còn:

```text
http://localhost:8000/api/chat/hybrid
```

---

### 13.3 Neo4j DNS error

Lỗi:

```text
Cannot resolve address xxxxx.databases.neo4j.io:7687
```

Kiểm tra:

```env
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_DATABASE=neo4j
```

Copy connection URI từ Neo4j Aura Console, không tự gõ.

---

### 13.4 Qdrant vector dimension mismatch

Lỗi:

```text
Vector dimension error: expected dim: 1024, got 384
```

Nghĩa là Qdrant collection đang 1024 dim nhưng backend tạo vector 384 dim.

Nếu dùng collection hiện tại:

```env
QDRANT_COLLECTION=youmed_sections
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

Nếu muốn dùng model 384 dim thì phải rebuild collection mới.

---

### 13.5 FastEmbed không hỗ trợ bge-m3

Lỗi:

```text
Model BAAI/bge-m3 is not supported in TextEmbedding
```

Nguyên nhân: code dùng `fastembed.TextEmbedding` để load `BAAI/bge-m3`.

Cách xử lý: dùng `sentence-transformers` cho `BAAI/bge-m3`.

---

### 13.6 Railway Out of Memory

Lỗi:

```text
Deployment failed
Out of Memory (OOM)
Killed
```

Thường xảy ra khi load:

```env
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

Vì model nặng.

Cách xử lý:

```text
1. Nâng RAM Railway.
2. Hoặc rebuild Qdrant bằng model nhỏ hơn.
3. Hoặc tách embedding thành service riêng.
```

Tạm thời nên set:

```env
DEBUG=false
ENHANCED_SCHEMA=false
RERANKER_ENABLED=false
USE_LLM_INTENT_ROUTER=false
```

Nhưng nếu vẫn load `bge-m3`, OOM vẫn có thể xảy ra.

---

## 14. Ghi chú bảo mật

Không commit các file sau:

```text
.env
.env.local
.env.production
```

Không đưa các key này lên Git:

```text
GROQ_API_KEY
GOOGLE_API_KEY
QDRANT_API_KEY
NEO4J_PASSWORD
```

Nếu từng paste key lên chat/log public, nên rotate lại:

```text
Groq API key
Google API key
Qdrant API key
Neo4j Aura password
```

---

## 15. Checklist production

Backend Railway:

```text
- Dockerfile dùng ${PORT:-8000}
- DEBUG=false
- ENHANCED_SCHEMA=false
- FRONTEND_ORIGINS đúng domain Vercel
- QDRANT_URL đúng
- QDRANT_COLLECTION đúng
- EMBEDDING_MODEL_NAME khớp collection
- RERANKER_ENABLED=false khi deploy lần đầu
- /health OK
- /api/chat/hybrid trả 200
```

Frontend Vercel:

```text
- Root Directory = frontend
- Build Command = npm run build
- Output Directory = dist
- VITE_API_BASE_URL trỏ về Railway backend
- Redeploy sau khi sửa env
- Network không còn gọi localhost
```

GraphRAG:

```text
- Qdrant search ra section_id
- Neo4j enrich được section/article/concept
- retrieval_mode trả đúng
- answer không rỗng
```
