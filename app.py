import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
import os
import base64
from fpdf import FPDF
import zipfile
import io
import PyPDF2
import google.generativeai as genai
import tempfile
from docx2pdf import convert


# Configure the Gemini API with the key (replace with your actual API key)
GEMINI_API_KEY = "AIzaSyAh1CxfDouDYBIz5l6Ia0MSt-jeM5HtJmI"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Initialize session state for PDF upload and conversion status
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False
if 'conversion_done' not in st.session_state:
    st.session_state.conversion_done = False

# Initialize session state for to-do list
if 'todo_list' not in st.session_state:
    st.session_state.todo_list = []

# Custom CSS for animations and layout
st.markdown("""
<style>
@keyframes fadeIn {
    0% {opacity: 0;}
    100% {opacity: 1;}
}
@keyframes slideIn {
    0% {transform: translateX(-100%);}
    100% {transform: translateX(0);}
}
@keyframes pulse {
    0% {transform: scale(1);}
    50% {transform: scale(1.05);}
    100% {transform: scale(1);}
}
.fade-in {
    animation: fadeIn 1s ease-out;
}
.slide-in {
    animation: slideIn 0.5s ease-out;
}
.pulse {
    animation: pulse 2s infinite;
}
</style>
""", unsafe_allow_html=True)

# Function to convert PDF to images and create a zip file
def convert_pdf_to_images_zip(pdf_path):
    images = convert_from_path(pdf_path)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for i, img in enumerate(images):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr(f'page_{i + 1}.png', img_buffer.getvalue())
    return zip_buffer.getvalue()

# Function to create a download link for a file
def create_download_link(file_content, file_name):
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="{file_name}" class="pulse">Download {file_name}</a>'
    return href

# Function to convert images to PDF
def convert_images_to_pdf(image_paths, output_pdf_path):
    pdf = FPDF()
    for image_path in image_paths:
        pdf.add_page()
        pdf.image(image_path, x=10, y=10, w=190)  # Adjust size as needed
    pdf.output(output_pdf_path)

# Function to summarize text using Gemini API

def summarize_with_gemini(text):
    prompt = f"Please summarize the following text in 3-4 sentences:\n\n{text[:10000]}"  # Limiting to first 10000 characters
    try:
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts'):
            return ' '.join(part.text for part in response.parts)
        else:
            return "Unable to extract summary from the API response."
    except Exception as e:
        st.error(f"An error occurred while generating the summary: {str(e)}")
        return None

 #Function to convert doc to pdf

def create_download_link(file_content, file_name):
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" class="pulse">Download {file_name}</a>'
    return href


# Streamlit App
st.markdown(
    '<h1 class="fade-in" style="text-align: center; font-size: 65px;">SnapPDF</h1>',
    unsafe_allow_html=True
)
st.markdown("---")

st.markdown('<p class="slide-in" style="text-align:center;">Streamline Your Workflow with Our All-in-One Solution For Tasks Which consume your more time</p>',
            unsafe_allow_html=True)
st.markdown("---")

# To-Do List Section
st.sidebar.header("To-Do List")
todo_item = st.sidebar.text_input("Add a new task:")
if st.sidebar.button("Add"):
    if todo_item:
        st.session_state.todo_list.append(todo_item)

if st.sidebar.button("Clear All Tasks"):
    st.session_state.todo_list.clear()

for item in st.session_state.todo_list:
    st.sidebar.write(f"- {item}")

# PDF to Images Section
st.markdown('<h2 class="fade-in">PDF to Images</h2>', unsafe_allow_html=True)

uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

if uploaded_pdf is not None:
    st.session_state.pdf_uploaded = True
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    if st.button("Convert PDF to Images"):
        with st.spinner('Converting PDF to images and creating ZIP file...'):
            try:
                zip_content = convert_pdf_to_images_zip("temp.pdf")
                st.success("PDF converted to images successfully!")
                zip_filename = f"{uploaded_pdf.name.rsplit('.', 1)[0]}_images.zip"
                download_link = create_download_link(zip_content, zip_filename)
                st.markdown(download_link, unsafe_allow_html=True)
                st.balloons()
                st.session_state.conversion_done = True
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Button to clear the uploaded PDF and conversion results
if st.session_state.pdf_uploaded:
    if st.button("Convert Another PDF"):
        st.session_state.pdf_uploaded = False
        st.session_state.conversion_done = False
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
        st.rerun()

st.markdown("---")

# Images to PDF Section
st.markdown('<h2 class="fade-in">Images to PDF</h2>', unsafe_allow_html=True)

uploaded_images = st.file_uploader("Choose image files (JPG/PNG)", type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=True)

if uploaded_images:
    image_paths = []
    for uploaded_image in uploaded_images:
        # Save the uploaded image temporarily
        with open(uploaded_image.name, "wb") as f:
            f.write(uploaded_image.getbuffer())
        image_paths.append(uploaded_image.name)

    if st.button("Convert Images to PDF"):
        with st.spinner('Converting images to PDF...'):
            try:
                if not image_paths:
                    st.error("Please upload at least one image.")
                else:
                    output_pdf_path = "output.pdf"
                    convert_images_to_pdf(image_paths, output_pdf_path)
                    st.success("Images converted to PDF successfully!")
                    # Create download link for the generated PDF
                    with open(output_pdf_path, "rb") as pdf_file:
                        pdf_content = pdf_file.read()
                    download_link = create_download_link(pdf_content, "converted_images.pdf")
                    st.markdown(download_link, unsafe_allow_html=True)
                    st.balloons()

                # Clean up temporary files
                for img in image_paths:
                    if os.path.exists(img):
                        os.remove(img)
                if os.path.exists(output_pdf_path):
                    os.remove(output_pdf_path)
            except Exception as e:
                st.error(f"Error: {str(e)}")


st.markdown("---")

st.markdown("---")
st.markdown('<h2 class="fade-in">Doc to PDF</h2>', unsafe_allow_html=True)

uploaded_doc = st.file_uploader("Choose a Word document (.doc or .docx)", type=["doc", "docx"], key="doc_uploader")

if uploaded_doc is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_doc.name) as tmp_file:
        tmp_file.write(uploaded_doc.getvalue())
        tmp_file_path = tmp_file.name

    if st.button("Convert Doc to PDF"):
        with st.spinner('Converting Doc to PDF...'):
            try:
                # Create a temporary directory for the output PDF
                with tempfile.TemporaryDirectory() as tmpdirname:
                    output_pdf_path = os.path.join(tmpdirname, "output.pdf")

                    # Convert the document to PDF
                    convert(tmp_file_path, output_pdf_path)

                    st.success("Document converted to PDF successfully!")

                    # Create download link for the generated PDF
                    with open(output_pdf_path, "rb") as pdf_file:
                        pdf_content = pdf_file.read()
                    download_link = create_download_link(pdf_content, "converted_document.pdf")
                    st.markdown(download_link, unsafe_allow_html=True)
                    st.balloons()

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                # Clean up the temporary input file
                os.unlink(tmp_file_path)

# ... [Previous code remains the same] ...

# PDF Summarizer Section
st.markdown("---")
st.markdown('<h2 class="fade-in">PDF Summarizer</h2>', unsafe_allow_html=True)

uploaded_pdf_for_summary = st.file_uploader("Choose a PDF file for summarization", type="pdf", key="pdf_summarizer")

if uploaded_pdf_for_summary is not None:
    with st.spinner('Extracting text from PDF and generating summary...'):
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_pdf_for_summary)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Generate summary using Gemini API
        summary = summarize_with_gemini(text)
        if summary:
            st.subheader("Summary")
            st.write(summary)

            # Option to download the summary
            if st.button("Download Summary"):
                st.download_button(
                    label="Download Summary as Text",
                    data=summary,
                    file_name="pdf_summary.txt",
                    mime="text/plain"
                )
        else:
            st.error("Failed to generate summary. Please try again later or contact the developer.")


# Clean up the temporary PDF file after the app is closed
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")