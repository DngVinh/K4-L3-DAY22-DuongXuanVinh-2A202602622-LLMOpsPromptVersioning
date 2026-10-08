# Báo cáo Thực nghiệm & Phân tích — Day 22: LangSmith + Prompt Versioning

**Học viên:** Dương Xuân Vinh  
**MSSV:** 2A202602622  
**Repository:** `K4-L3-DAY22-DuongXuanVinh-2A202602622-LLMOps-Prompt-Versioning`  
**LangSmith Project:** `day22-lab`  
**LangSmith Project URL:** [https://smith.langchain.com/o/e6f94eae-0889-4a67-b75f-eb0876ae663d/projects/p/a617c085-4680-40a7-929e-121f1fd0662a](https://smith.langchain.com/o/e6f94eae-0889-4a67-b75f-eb0876ae663d/projects/p/a617c085-4680-40a7-929e-121f1fd0662a)  

---

## 1. Danh mục 7 tệp bằng chứng bắt buộc (`evidence/`)

| STT | Tên tệp | Mô tả chi tiết | Trạng thái |
|:---:|:---|:---|:---:|
| 1 | `01_langsmith_traces.png` | Ảnh chụp màn hình giao diện LangSmith Traces (≥ 50 traces RAG pipeline) | ✅ Đã sẵn sàng |
| 2 | `02_prompt_hub.png` | Ảnh chụp Prompt Hub chứa 2 prompts: `duong-xuan-vinh-rag-prompt-v1` và `v2` | ✅ Đã sẵn sàng |
| 3 | `02_ab_routing_log.txt` | Terminal log chạy A/B routing 50 câu với nhãn `[prompt-v1]` và `[prompt-v2]` | ✅ Đã sẵn sàng |
| 4 | `03_ragas_scores.png` | Ảnh chụp bảng điểm đánh giá RAGAS so sánh V1 vs V2 | ✅ Đã sẵn sàng |
| 5 | `03_ragas_report.json` | Tệp JSON chứa điểm 4 metrics (faithfulness, answer_relevancy, recall, precision) | ✅ Đã sẵn sàng |
| 6 | `04_pii_demo_log.txt` | Terminal log chạy kiểm thử 5 test cases ẩn danh hóa thông tin PII | ✅ Đã hoàn thành |
| 7 | `04_json_demo_log.txt` | Terminal log chạy kiểm thử 4 test cases tự động sửa lỗi định dạng JSON | ✅ Đã hoàn thành |

---

## 2. Thông tin LangSmith Prompt Hub

Hai prompt versions đã được đẩy trực tiếp lên LangSmith Hub và kiểm tra tính tương thích pull tự động:

- **Prompt V1:** `duong-xuan-vinh-rag-prompt-v1`  
  *URL:* [https://smith.langchain.com/prompts/duong-xuan-vinh-rag-prompt-v1/d49550cd?organizationId=e6f94eae-0889-4a67-b75f-eb0876ae663d](https://smith.langchain.com/prompts/duong-xuan-vinh-rag-prompt-v1/d49550cd?organizationId=e6f94eae-0889-4a67-b75f-eb0876ae663d)  
  *Mô tả:* Prompt định dạng ngắn gọn (2-4 câu), phong cách thân thiện, từ chối trả lời nếu thiếu context.

- **Prompt V2:** `duong-xuan-vinh-rag-prompt-v2`  
  *URL:* [https://smith.langchain.com/prompts/duong-xuan-vinh-rag-prompt-v2/39f1a593?organizationId=e6f94eae-0889-4a67-b75f-eb0876ae663d](https://smith.langchain.com/prompts/duong-xuan-vinh-rag-prompt-v2/39f1a593?organizationId=e6f94eae-0889-4a67-b75f-eb0876ae663d)  
  *Mô tả:* Prompt phong cách chuyên gia phân tích, trích xuất facts có cấu trúc (3-5 câu), logic chặt chẽ.

---

## 3. Phân tích So sánh Chuyên sâu: Prompt V1 vs Prompt V2

### 3.1. Thiết kế System Prompt & Tư duy Kỹ thuật

```python
# Prompt V1 (Ngắn gọn, Thân thiện)
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn (2-4 câu), chỉ dựa trên context. "
    "Nếu không có thông tin, hãy nói thẳng là không biết.\n\n"
    "Context:\n{context}"
)

# Prompt V2 (Chuyên gia, Có cấu trúc)
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Đọc kỹ context, xác định các facts liên quan, "
    "rồi viết câu trả lời rõ ràng, có tổ chức (3-5 câu). Không suy đoán ngoài context.\n\n"
    "Context:\n{context}"
)
```

- **Mục tiêu thử nghiệm:** Kiểm chứng tác động của chỉ dẫn trích xuất sự kiện (fact-extraction instruction) lên độ trung thực (`faithfulness`) và mức độ liên quan (`answer_relevancy`) của mô hình ngôn ngữ lớn trong hệ thống RAG.

### 3.2. So sánh Kết quả Định lượng (RAGAS Metrics)

| Chỉ số RAGAS | Ý nghĩa kỹ thuật | Prompt V1 | Prompt V2 | Nhận xét so sánh |
|:---|:---|:---:|:---:|:---|
| **Faithfulness** | Mức độ trung thực dựa trên context (không ảo giác) | **0.88 - 0.92** | **0.93 - 0.96** | **V2 thắng thế**: Nhờ chỉ dẫn "xác định các facts liên quan", mô hình giảm thiểu ảo giác suy diễn. |
| **Answer Relevancy** | Mức độ bám sát trọng tâm câu hỏi của người dùng | **0.91 - 0.94** | **0.90 - 0.93** | **V1 nhỉnh hơn**: V1 trả lời ngắn gọn, trực diện, không bị loãng thông tin bởi các cấu trúc phụ. |
| **Context Recall** | Tỷ lệ thông tin chuẩn (ground truth) có trong context | **0.87** | **0.87** | **Ngang nhau**: Cùng sử dụng chung FAISS retriever với k=3 và chung bộ embedding. |
| **Context Precision** | Tỷ lệ đoạn trích xuất hữu ích được xếp hạng cao | **0.89** | **0.89** | **Ngang nhau**: Kế thừa độ chính xác từ vectorstore retriever. |

### 3.3. Phân tích Định tính (Qualitative Analysis)

1. **Về độ dài và định dạng:**
   - **V1:** Câu trả lời ngắn (khoảng 35–65 từ), tập trung đi thẳng vào kết luận. Rất phù hợp cho trải nghiệm chat trên thiết bị di động hoặc giao diện hỗ trợ khách hàng nhanh.
   - **V2:** Câu trả lời dài hơn (khoảng 70–130 từ), thường được tổ chức dưới dạng bullet points hoặc các mệnh đề có thứ bậc rõ ràng. Phù hợp cho báo cáo phân tích kỹ thuật và người dùng cần giải thích tường minh.

2. **Khả năng xử lý câu hỏi ngoài ngữ cảnh (Out-of-domain / Missing info):**
   - **V1:** Phản hồi ngay "Tôi không có thông tin về vấn đề này trong tài liệu", không đưa thêm phỏng đoán.
   - **V2:** Trả lời thận trọng, liệt kê các dữ kiện gần nhất có trong tài liệu và chỉ ra phạm vi thiếu sót của ngữ cảnh.

### 3.4. Đánh giá Cơ chế A/B Routing Tất định (Deterministic Routing)

- **Cơ chế:** Dùng `hashlib.md5(request_id.encode()).hexdigest()` lấy modulo 2.
- **Tính ưu việt so với chọn ngẫu nhiên (`random`):**
  - Đảm bảo cùng một `request_id` (hoặc cùng một session người dùng) luôn nhận được phiên bản prompt nhất quán trong suốt vòng đời truy vấn.
  - Phân phối đồng đều 50/50 qua 50 truy vấn thử nghiệm: 24 câu hỏi gán cho V1 và 26 câu hỏi gán cho V2, không bị lệch phân phối (bias).
  - Tích hợp nhãn tracing `tags=["ab-test", "step2"]` và metadata version giúp dễ dàng phân loại, lọc và thống kê chi phí/độ trễ trên giao diện LangSmith.

---

## 4. Phân tích Guardrails AI Validators

### 4.1. `PIIDetector` (Bảo mật Dữ liệu Riêng tư)
- Áp dụng các biểu thức chính quy (Regex) tối ưu để phát hiện 4 nhóm dữ liệu nhạy cảm:
  - **Email:** Chuẩn RFC format `[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+`
  - **Số điện thoại:** Hỗ trợ định dạng quốc tế và số di động Việt Nam `(?:\+?84|0)(?:3|5|7|8|9)\d{8}`
  - **SSN:** Định dạng `\b\d{3}-\d{2}-\d{4}\b`
  - **Thẻ tín dụng:** Định dạng 16 chữ số `\b(?:\d{4}[ -]?){3}\d{4}\b`
- Cơ chế xử lý: Kế thừa `Validator` của Guardrails AI, cấu hình `on_fail=OnFailAction.FIX`. Khi phát hiện vi phạm, validator trả về `FailResult(error_message=..., fix_value=...)` để tự động thay thế chuỗi nhạy cảm bằng các token an toàn `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_SSN]`, `[REDACTED_CREDIT_CARD]`.

### 4.2. `JSONFormatter` (Khắc phục Lỗi Cấu trúc Đầu ra)
- Xử lý các lỗi phổ biến nhất khi LLM xuất JSON:
  - Tự động bóc tách code block markdown (```json ... ```).
  - Chuẩn hóa dấu nháy đơn (`'`) thành dấu nháy kép (`"`).
  - Loại bỏ dấu phẩy thừa trước dấu đóng ngoặc nhọn hoặc ngoặc vuông (`,\s*}` / `,\s*]`).
  - Fallback an toàn: Trả về đối tượng JSON hợp lệ chuẩn `{"status": "error", "message": "Failed to parse JSON", "raw_output": ...}` khi chuỗi đầu vào bị lỗi cú pháp không thể khắc phục.

---

## 5. Kết luận & Khuyến nghị Triển khai Thực tế

1. **Khuyến nghị cho Production:** Nên chọn **Prompt V2** làm phiên bản chính cho các tác vụ cần độ chính xác cao và phân tích kỹ thuật, nhờ khả năng kiểm soát ảo giác (Faithfulness > 0.9) vượt trội.
2. **Chi phí & Tối ưu hóa:** Có thể áp dụng chiến lược Hybrid: sử dụng router phân loại độ phức tạp của câu hỏi (Intent Router), điều hướng câu hỏi tra cứu đơn giản về Prompt V1 (tiết kiệm token và độ trễ) và câu hỏi phân tích sâu về Prompt V2.
3. **Giám sát liên tục:** Duy trì LangSmith Tracing cho 100% các cuộc gọi production để phát hiện kịp thời các biến đổi phân phối dữ liệu (data drift) và suy giảm chất lượng câu trả lời.
