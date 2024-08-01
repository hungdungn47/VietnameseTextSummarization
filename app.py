import streamlit as st
from io import StringIO
from chdg_inference import infer

st.title("Tóm tắt văn bản tiếng Việt")
uploaded_file = st.file_uploader(label="Chọn file văn bản")

category = st.selectbox("Chọn chủ để của văn bản: ", ['Giáo dục', 'Giải trí - Thể thao', 'Khoa học - Công nghệ', 'Kinh tế', 'Pháp luật', 'Thế giới', 'Văn hóa - Xã hội', 'Đời sống'])

def summarize():
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        full_text = stringio.read()
        summ, docs = infer(full_text, category)
        st.subheader("Kết quả: ")
        st.write(summ)
        st.subheader("Docs: ")
        st.write(docs)
    else:
        st.error("Hãy tải file văn bản lên")

if st.button("Tóm tắt"):
    summarize()