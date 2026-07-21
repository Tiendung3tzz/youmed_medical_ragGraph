Đã cập nhật logic GraphRAG trong project fullstack theo phiên bản Routed Retrieval V2 đã kiểm thử trên notebook.

1. Cập nhật backend/app/services/graph_rag_service.py

- Bổ sung cơ chế routed retrieval thay cho chỉ chạy retrieval tuyến tính.
- Thêm hàm ask_routed_retrieval() làm dispatcher chính cho retrieval.
- Bổ sung 4 route xử lý:
  + heading_lookup:
    * Dùng Neo4j exact lookup theo Section.heading.
    * Phù hợp với các câu hỏi kiểu: "Những bệnh nào có phần tiêu đề Chẩn đoán?"
  + exact_entity_lookup:
    * Dùng Neo4j exact lookup theo Concept.name / Concept.displayName.
    * Nếu không có dữ liệu thì fallback sang semantic search.
  + multi_entity_search:
    * Tách nhiều entity trong câu hỏi.
    * Gọi Qdrant theo từng entity/sub-query.
    * Merge candidate section_id.
    * Enrich dữ liệu từ Neo4j.
    * Rerank lại bằng BGE reranker + graph-aware score.
  + semantic_search:
    * Qdrant lấy top N candidates.
    * Neo4j enrich theo section_id.
    * Rerank lại bằng BGE reranker + graph-aware score.

- Bổ sung rule detector để nhận diện nhanh heading lookup.
- Bổ sung LLM intent extraction nhưng chỉ gọi khi câu hỏi có dấu hiệu structured lookup.
- Không cho LLM sinh Cypher trực tiếp cho retrieval route.
- LLM chỉ extract JSON intent gồm route, heading, entities, relations, entity_kind, confidence.
- Bổ sung normalize text / normalize entity key để so khớp tiếng Việt không dấu và entity có ký tự đặc biệt.
- Bổ sung detect_expected_relations() để nhận diện intent như tác dụng phụ, liều dùng, chống chỉ định, tương tác, chẩn đoán, điều trị, tổng quan...
- Bổ sung detect_expected_kinds() để nhận diện nhóm Drug, Disease, BodyPart, TraditionalMedicine.
- Bổ sung các hàm scoring:
  + relation_bonus()
  + kind_bonus()
  + entity_bonus()
  + heading_bonus()
  + clinical_term_bonus()
  + text_bonus()
- Bổ sung hàm enrich_sections_from_neo4j() để lấy dữ liệu Section, Article, Category, Concept, ClinicalTerm, previous_sections, next_sections.
- Bổ sung hàm rerank_rows() để tính điểm cuối cùng từ:
  + qdrant score
  + BGE reranker score
  + relation bonus
  + entity bonus
  + heading bonus
  + kind bonus
  + clinical term bonus
  + text bonus
- Bổ sung debug output gồm:
  + retrieval_mode
  + qdrant_hits
  + candidate_section_ids
  + section_ids
  + debug
- Cập nhật ask_hybrid() để gọi routed retrieval thay vì chỉ gọi Qdrant/Neo4j retrieval cũ.
- Giữ fallback an toàn: nếu heading/entity route không có kết quả thì tự fallback sang semantic search.

2. Thêm file backend/app/services/reranker_service.py

- Tạo service riêng để load và gọi SentenceTransformers CrossEncoder reranker.
- Hỗ trợ cấu hình model reranker bằng HuggingFace model id hoặc local path.
- Có fallback an toàn:
  + Nếu chưa cài sentence-transformers/torch thì reranker trả score 0.
  + Nếu load model lỗi thì retrieval vẫn chạy bằng Qdrant + graph-aware score.
  + Nếu predict lỗi thì trả score 0 để không làm chết API.
- Model mặc định:
  + BAAI/bge-reranker-base
- Có thể đổi sang local path, ví dụ:
  + /models/bge-reranker-base
  + /kaggle/input/.../bge-reranker-v2-m3

3. Cập nhật backend/app/core/config.py

- Bổ sung cấu hình Qdrant/rerank/router:
  + qdrant_search_top_k
  + use_llm_intent_router
  + intent_confidence_threshold
  + reranker_enabled
  + reranker_model_name
  + reranker_device
  + reranker_max_length
  + reranker_max_chars
  + reranker_batch_size
- Phân biệt:
  + qdrant_search_top_k: số candidate lấy từ Qdrant trước khi rerank, mặc định 50.
  + qdrant_top_k: số kết quả cuối trả về, mặc định 10.
- Giữ embedding_model_name là model embedding dùng cho Qdrant, không trộn với reranker model.
- Reranker model được cấu hình riêng qua reranker_model_name.

4. Cập nhật backend/app/schemas/chat.py

- Bổ sung field vào ChatResponse:
  + retrieval_mode
  + qdrant_hits
  + candidate_section_ids
  + section_ids
  + debug
- Bổ sung field tương tự vào GraphOnlyResponse.
- Mục đích: frontend/API caller có thể kiểm tra route, candidate, kết quả rerank và debug pipeline.

5. Cập nhật backend/app/api/chat.py

- Điều chỉnh response của hybrid chat.
- Khi include_debug=false:
  + Ẩn qdrant_hits
  + Ẩn candidate_section_ids
  + Ẩn section_ids
  + Ẩn debug
- Khi include_debug=true:
  + Trả đầy đủ thông tin debug để kiểm tra retrieval/rerank.

6. Cập nhật backend/requirements.txt

- Bổ sung dependency phục vụ reranker:
  + sentence-transformers
  + torch
- Các dependency này dùng cho CrossEncoder reranker.
- Nếu môi trường không cài hoặc model không load được thì code vẫn fallback, không làm hỏng retrieval chính.

7. Cập nhật backend/.env.example

- Bổ sung cấu hình mẫu cho routed retrieval:
  + QDRANT_TOP_K=10
  + QDRANT_SEARCH_TOP_K=50
  + USE_LLM_INTENT_ROUTER=true
  + INTENT_CONFIDENCE_THRESHOLD=0.5
  + RERANKER_ENABLED=true
  + RERANKER_MODEL_NAME=BAAI/bge-reranker-base
  + RERANKER_DEVICE=
  + RERANKER_MAX_LENGTH=512
  + RERANKER_MAX_CHARS=900
  + RERANKER_BATCH_SIZE=4
- Không đưa file .env thật vào zip để tránh lộ secret.

8. Thêm file ROUTED_RETRIEVAL_UPDATE.md

- Ghi chú lại flow mới:
  + question
  + rule detector
  + LLM intent extraction nếu cần
  + route dispatcher
  + Qdrant/Neo4j/rerank
- Mô tả 4 route:
  + heading_lookup
  + exact_entity_lookup
  + multi_entity_search
  + semantic_search
- Ghi chú API test:
  + POST /api/chat/hybrid
  + include_debug=true để xem retrieval_mode và debug.

9. Kiểm tra kỹ thuật

- Đã chạy compile check backend bằng:
  + python -m compileall app
- Kết quả: không có lỗi cú pháp.
- Chưa chạy runtime API thực tế vì cần môi trường thật gồm:
  + Neo4j
  + Qdrant
  + LLM API key
  + model embedding/reranker


  