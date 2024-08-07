import streamlit as st
from io import StringIO
from chdg_inference import infer
from infer_concat import vit5_infer

st.set_page_config(layout="wide")
st.title("Tóm tắt Đa văn bản Tiếng Việt")

col1, col2 = st.columns([1, 1])
col2_title, = col2.columns(1)
col2_chdg, col2_vit5 = col2.columns(2)

# Initialize session state
if 'num_docs' not in st.session_state:
    st.session_state.num_docs = 0
if 'docs' not in st.session_state:
    st.session_state.docs = []

# Function to add a new text area
def add_text_area():
    st.session_state.num_docs += 1

# Button to add a new text area
col1.button("Thêm văn bản", on_click=add_text_area)

# Display text areas for document input
for i in range(st.session_state.num_docs):
    doc = col1.text_area(f"Văn bản {i+1}", key=f"doc_{i}", height=150)
    doc.replace('\r', '\n')
    doc.replace('\"', "'")
    if len(st.session_state.docs) <= i:
        st.session_state.docs.append(doc)
    else:
        st.session_state.docs[i] = doc

category = col1.selectbox("Chọn chủ để của văn bản: ", ['Giáo dục', 'Giải trí - Thể thao', 'Khoa học - Công nghệ', 'Kinh tế', 'Pháp luật', 'Thế giới', 'Văn hóa - Xã hội', 'Đời sống'])

def summarize():
    summ, _ = infer(st.session_state.docs, category)
    with col2.container():
        col2_title.subheader("Kết quả: ")

    with col2.container():
        col2_chdg.write(summ)
        summ_vit5 = vit5_infer(st.session_state.docs)
        col2_vit5.write(summ_vit5)
    
if col1.button("Tóm tắt"):
    summarize()