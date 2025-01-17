# Tóm tắt Đa văn bản Tiếng Việt

Đây là một ứng dụng tóm tắt Đa văn bản Tiếng Việt viết trên Python và deploy với Streamlit cloud.

Link sản phẩm:

https://vietnamesetextsummarization-pbuilwergn5mbkcmf9ybyg.streamlit.app/

## Bộ dữ liệu Abmusu

Abmusu gồm 600 cụm văn bản (1839 văn bản) cho tóm tắt đa văn bản tiếng Việt, trong đó

Tập train: 200 cụm - 621 văn bản

Tập validation: 100 cụm - 304 văn bản

Tập test: 300 cụm - 914 văn bản

Mỗi data example gồm title, anchor text và body text của toàn bộ văn bản trong cụm văn bản. Mỗi cụm văn bản có một chủ đề và một tóm tắt mẫu.

## Phương pháp tóm tắt

Bài toán tóm tắt văn bản có 2 hướng tiếp cận:

- Extractive summarization: Chọn những câu mang nhiều ý nghĩa quan trọng nhất, có thể thể hiện nội dung toàn bộ văn bản. Lấy nguyên văn những câu đó để tạo bản tóm tắt.

- Abstractive summarization: Dùng mô hình sinh để sinh ra bản tóm tắt. Bản tóm tắt có thể chứa các cụm từ và các câu không xuất hiện trong văn bản gốc.

Ứng dụng này cài đặt cả 2 hướng tiếp cận với 2 phương pháp cụ thể sau:

- Hướng extractive: Contrastive Hierarchical Discourse Graph
- Hướng abstractive: Finetune ViT5
