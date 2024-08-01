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
    if len(st.session_state.docs) <= i:
        st.session_state.docs.append(doc)
    else:
        st.session_state.docs[i] = doc
# Display the documents for verification
# st.write("**Entered Documents:**")
# st.write(st.session_state.docs)
# uploaded_file = st.file_uploader(label="Chọn file văn bản")

category = st.selectbox("Chọn chủ để của văn bản: ", ['Giáo dục', 'Giải trí - Thể thao', 'Khoa học - Công nghệ', 'Kinh tế', 'Pháp luật', 'Thế giới', 'Văn hóa - Xã hội', 'Đời sống'])

def summarize():
    # if uploaded_file is not None:
    #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #     full_text = stringio.read()
    #     summ, docs = infer(full_text, category)
    #     st.subheader("Kết quả: ")
    #     st.write(summ)
    #     st.subheader("Docs: ")
    #     st.write(docs)
    # else:
    #     st.error("Hãy tải file văn bản lên")
    summ, docs = infer(st.session_state.docs, category)
    st.subheader("Kết quả")
    st.write(summ)

if st.button("Tóm tắt"):
    summarize()