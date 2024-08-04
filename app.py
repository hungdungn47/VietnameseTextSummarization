import streamlit as st
from io import StringIO
from chdg_inference import infer

st.title("Tóm tắt đa văn bản tiếng Việt")

# Initialize session state
if 'num_docs' not in st.session_state:
    st.session_state.num_docs = 0
if 'docs' not in st.session_state:
    st.session_state.docs = []

# Function to add a new text area
def add_text_area():
    st.session_state.num_docs += 1


# Button to add a new text area
st.button("Thêm văn bản", on_click=add_text_area)

# Display text areas for document input
for i in range(st.session_state.num_docs):
    doc = st.text_area(f"Văn bản {i+1}", key=f"doc_{i}", height=200)
    doc.replace('\r', '\n')
    doc.replace('\"', "'")
    if len(st.session_state.docs) <= i:
        st.session_state.docs.append(doc)
    else:
        st.session_state.docs[i] = doc

category = st.selectbox("Chọn chủ để của văn bản: ", ['Giáo dục', 'Giải trí - Thể thao', 'Khoa học - Công nghệ', 'Kinh tế', 'Pháp luật', 'Thế giới', 'Văn hóa - Xã hội', 'Đời sống'])

def summarize():
    summ, _ = infer(st.session_state.docs, category)
    st.subheader("Kết quả")
    st.write(summ)
    
if st.button("Tóm tắt"):
    summarize()