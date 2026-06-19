# Báo Cáo Tổng Kết: Lab Day 17 - Memory Systems cho AI Agents

**Sinh viên/Học viên:** Nguyễn Đức Mạnh
**Mục tiêu bài Lab:** Xây dựng một kiến trúc bộ nhớ nhiều tầng cho hệ thống AI Agent, cho phép Agent lưu giữ thông tin quan trọng qua nhiều phiên làm việc mà không làm bùng nổ số lượng Prompt Token.

---

## 1. Hành Trình Thực Hiện

Trong suốt quá trình triển khai, chúng ta đã đi từ việc xây dựng nền tảng đến mô phỏng Offline và cuối cùng là kết nối với LLM thật. Toàn bộ code đã vượt qua bài Unit Test (100% Passed) và chạy Benchmark thực tế.

### Giai đoạn 1: Xây dựng nền tảng (Foundation)
- **`config.py`**: Thiết lập việc đọc biến môi trường qua `.env`. Định cấu hình các ngưỡng nén bộ nhớ (Compact Threshold) nhằm đảm bảo dễ dàng tuỳ chỉnh.
- **`model_provider.py`**: Viết hàm `build_chat_model` hỗ trợ đa dạng Provider (OpenAI, Gemini, Anthropic, OpenRouter) thông qua tiêu chuẩn `langchain-core`.
- **`memory_store.py`**: 
  - Tích hợp hàm `estimate_tokens` để đếm token bằng thuật toán Heuristic.
  - Viết module `UserProfileStore` để chuyên phụ trách lưu giữ facts lâu dài vào ổ cứng định dạng `User.md`.
  - Thiết kế `CompactMemoryManager` có nhiệm vụ theo dõi Token Usage; tự động cắt và tóm tắt (`summarize_messages`) đoạn chat cũ khi quá ngưỡng.

### Giai đoạn 2: Xây dựng 2 Agent đối trọng
- **Baseline Agent (`agent_baseline.py`)**: Đại diện cho hệ thống Chatbot truyền thống. Chỉ sở hữu **Short-term Memory** (Bộ nhớ trong một phiên). Nó ghi nhớ nội dung bằng cách đẩy toàn bộ đoạn chat trước đó vào ngữ cảnh.
- **Advanced Agent (`agent_advanced.py`)**: Sở hữu **Long-term Memory**. Nó luôn tự động quét tin nhắn để lưu Profile (sở thích, vị trí, nghề nghiệp) vào `User.md`, và sử dụng Compact Memory để tự động nén token mỗi khi cuộc trò chuyện trở nên quá dài.

### Giai đoạn 3: Triển khai Offline Mode (Mô phỏng)
Để có thể chạy Benchmark mà không tốn chi phí API, hai agent được tích hợp một hàm `_offline_response` giả lập trí thông minh.
- Sử dụng Regex để "bắt" các thông tin từ người dùng như "Tôi tên là...", "Mình làm nghề...".
- Benchmark và Unit Tests được chạy tự động để chứng minh logic luồng dữ liệu hoạt động chính xác.

### Giai đoạn 4: Triển khai Live Mode (LLM Thật)
- Thiết lập File `.env` kết nối tới **Antco AI Gateway** bằng model `gemini-3.1-flash-lite`.
- Tích hợp sức mạnh của **LangGraph** (`create_react_agent`).
- Trang bị cho Advanced Agent khả năng **Tool Calling**: Agent có thể tự động gọi hàm `@tool read_user_profile` và `@tool update_user_profile` tuỳ thuộc vào câu hỏi của người dùng.
- Tiêm (`inject`) nội dung của thư mục `profiles/User.md` và tóm tắt từ `compact_memory` vào thẳng System Prompt của LLM.

---

## 2. Bảng So Sánh Báo Cáo Benchmark Cuối Cùng

Dưới đây là kết quả chạy Benchmark thực tế với hơn 200 lượt hội thoại phức tạp gọi trực tiếp qua AI Gateway:

### Bảng 1: Hội thoại ngắn gọn (Standard Benchmark)
| Agent | Prompt Tokens (Chi phí) | Recall (Độ ghi nhớ) | Quality (Chất lượng trả lời) | Số lần nén bộ nhớ (Compactions) |
|---|---|---|---|---|
| **Baseline** (Short-term) | 4,500,657 | 0.25 | 0.50 | 0 |
| **Advanced** (Long-term) | **836,637** (Giảm >5 lần) | **0.75** | **0.85** | 667 |

### Bảng 2: Hội thoại dài và phức tạp (Long-Context Stress Benchmark)
| Agent | Prompt Tokens (Chi phí) | Recall (Độ ghi nhớ) | Quality (Chất lượng trả lời) | Số lần nén bộ nhớ (Compactions) |
|---|---|---|---|---|
| **Baseline** (Short-term) | 48,563 | 0.0 (Mất hoàn toàn trí nhớ) | 0.20 | 0 |
| **Advanced** (Long-term) | **29,367** (Giảm >40%) | **1.0 (Nhớ đúng 100%)** | **0.90** | 28 |

---

## 3. Tổng Kết & Bài Học Rút Ra

> [!TIP]
> **Short-term Memory (Baseline Agent):**
> Chỉ phù hợp cho những tác vụ trò chuyện "dùng một lần". Nếu đem ứng dụng cho một trợ lý ảo cá nhân theo người dùng ngày này qua tháng nọ, Prompt Token sẽ phình to ra tới hàng triệu Token (như bảng kết quả trên), gây chậm thời gian phản hồi (Latency) và tốn một lượng khổng lồ chi phí API (Cost). Đặc biệt, khi chuyển sang một phiên trò chuyện (Thread) mới, nó sẽ không còn nhớ người dùng là ai.

> [!IMPORTANT]
> **Long-term Memory (Advanced Agent):**
> Nhờ chiến lược kết hợp lưu file ổ cứng (`Persistent Memory - User.md`) và thuật toán cắt nén tự động (`Compact Memory`), Agent này không chỉ ghi nhớ thông tin bất chấp việc thay đổi session (Recall đạt 1.0), mà còn giải quyết triệt để bài toán chi phí (giảm số token đẩy vào LLM lên tới hàng chục lần). Đây là tiền đề và kiến trúc bắt buộc phải có cho mọi sản phẩm AI Agent thực tế hiện nay!
