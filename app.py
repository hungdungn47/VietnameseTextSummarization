import streamlit as st
from io import StringIO
from chdg_inference import infer

st.set_page_config(layout="wide")
st.title("Tóm tắt Đa văn bản Tiếng Việt")

col1, col2 = st.columns([2.5, 1])

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
    col2.subheader("Kết quả")
    col2.write(summ)
    
if col1.button("Tóm tắt"):
    summarize()