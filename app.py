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
from pdf2docx import parse
from dotenv import load_dotenv
from PyPDF2 import PdfMerger
from docx2pdf import convert
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import time
import base64
import pdf2image




# Configure the Gemini API with the key (replace with your actual API key)
load_dotenv()  # Load environment variables from .env file
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
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

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Add custom CSS to hide the GitHub icon
hide_github_icon = """
#GithubIcon {
  visibility: hidden;
}
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

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


# Helper function for watermark creation
def createWatermark(text, color):
    # Create a temporary file for the watermark
    temp_watermark = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    c = canvas.Canvas(temp_watermark.name, pagesize=letter)

    # Set font and color
    c.setFont("Helvetica", 60)
    r, g, b = tuple(int(color.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4))
    c.setFillColorRGB(r, g, b)

    # Set transparency
    c.setFillAlpha(0.3)

    # Add text
    c.translate(letter[0] / 2, letter[1] / 2)
    c.rotate(45)
    c.drawCentredString(0, 0, text)
    c.save()

    return temp_watermark.name


# Streamlit App
st.markdown(
    '<h1 class="fade-in" style="text-align: center; font-size: 65px;">SnapPDF</h1>',
    unsafe_allow_html=True
)
st.markdown("---")

st.markdown(
    '<p class="slide-in" style="text-align:center;">Streamline Your Workflow with Our All-in-One Solution For Tasks '
    'Which consume your more time</p>',
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

# Add PDF to DOC section
st.markdown("---")
st.markdown('<h2 class="fade-in">PDF to DOC</h2>', unsafe_allow_html=True)

uploaded_pdf_for_doc = st.file_uploader("Choose a PDF file to convert to DOC", type="pdf", key="pdf_doc_uploader")

if uploaded_pdf_for_doc is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf_file:
        tmp_pdf_file.write(uploaded_pdf_for_doc.getvalue())
        pdf_path = tmp_pdf_file.name

    if st.button("Convert PDF to DOC"):
        with st.spinner('Converting PDF to DOC...'):
            try:
                # Create temporary output path for DOCX file
                output_docx = tempfile.NamedTemporaryFile(delete=False, suffix='.docx').name

                # Convert PDF to DOCX using parse function
                parse(pdf_path, output_docx)

                # Read the converted file
                with open(output_docx, 'rb') as docx_file:
                    docx_content = docx_file.read()

                # Create download link
                download_link = create_download_link(docx_content, "converted_document.docx")
                st.success("PDF converted to DOC successfully!")
                st.markdown(download_link, unsafe_allow_html=True)
                st.balloons()

                # Clean up temporary files
                os.unlink(output_docx)

            except Exception as e:
                st.error(f"Error during conversion: {str(e)}")
            finally:
                # Clean up the temporary PDF file
                os.unlink(pdf_path)

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

# 1. PDF Merger Section
st.markdown("---")
st.markdown('<h2 class="fade-in">PDF Merger</h2>', unsafe_allow_html=True)

uploaded_pdfs = st.file_uploader("Upload multiple PDFs to merge", type="pdf", accept_multiple_files=True,
                                 key="pdf_merger")

if uploaded_pdfs:
    if st.button("Merge PDFs"):
        with st.spinner('Merging PDFs...'):
            try:
                merger = PdfMerger()

                # Create temporary files for each uploaded PDF
                temp_paths = []
                for pdf in uploaded_pdfs:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    temp_file.write(pdf.read())
                    temp_paths.append(temp_file.name)
                    temp_file.close()

                # Merge PDFs
                for temp_path in temp_paths:
                    merger.append(temp_path)

                # Save merged PDF
                output_path = "merged_document.pdf"
                merger.write(output_path)
                merger.close()

                # Create download link
                with open(output_path, "rb") as f:
                    pdf_content = f.read()
                download_link = create_download_link(pdf_content, "merged_document.pdf")
                st.success("PDFs merged successfully!")
                st.markdown(download_link, unsafe_allow_html=True)
                st.balloons()

                # Cleanup
                for temp_path in temp_paths:
                    os.unlink(temp_path)
                os.unlink(output_path)

            except Exception as e:
                st.error(f"Error during merging: {str(e)}")

# PDF Splitter Section
st.markdown("---")
st.markdown('<h2 class="fade-in">PDF Splitter</h2>', unsafe_allow_html=True)

uploaded_pdf_split = st.file_uploader("Choose a PDF to split", type="pdf", key="pdf_splitter")

if uploaded_pdf_split is not None:
    try:
        # Create a bytes buffer instead of temporary file
        pdf_bytes = uploaded_pdf_split.read()
        pdf = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(pdf.pages)

        st.write(f"Total pages: {num_pages}")
        page_range = st.text_input("Enter page range (e.g., '1-3,5,7-9'):")

        if st.button("Split PDF"):
            with st.spinner('Splitting PDF...'):
                try:
                    # Parse page range
                    pages_to_extract = set()  # Using set to avoid duplicates
                    for part in page_range.strip().split(','):
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            # Validate page range
                            if 1 <= start <= end <= num_pages:
                                pages_to_extract.update(range(start - 1, end))
                            else:
                                raise ValueError(f"Invalid page range: {start}-{end}")
                        else:
                            page = int(part)
                            # Validate single page
                            if 1 <= page <= num_pages:
                                pages_to_extract.add(page - 1)
                            else:
                                raise ValueError(f"Invalid page number: {page}")

                    if not pages_to_extract:
                        raise ValueError("No valid pages specified")

                    # Create new PDF with selected pages
                    output = PyPDF2.PdfWriter()
                    for page_num in sorted(pages_to_extract):
                        output.add_page(pdf.pages[page_num])

                    # Write to bytes buffer instead of file
                    output_buffer = io.BytesIO()
                    output.write(output_buffer)
                    output_buffer.seek(0)

                    # Create download link
                    pdf_content = output_buffer.getvalue()
                    download_link = create_download_link(pdf_content, "split_document.pdf")
                    st.success("PDF split successfully!")
                    st.markdown(download_link, unsafe_allow_html=True)
                    st.balloons()

                except ValueError as ve:
                    st.error(f"Error: {str(ve)}")
                except Exception as e:
                    st.error(f"An error occurred during splitting: {str(e)}")

    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")

# PDF Compression Section
st.markdown("---")
st.markdown('<h2 class="fade-in">PDF Compressor</h2>', unsafe_allow_html=True)


def compress_pdf(input_path, output_path, compression_level):
    """
    Compress PDF based on compression level (1-5)
    1: Minimal compression
    2: Low compression
    3: Medium compression
    4: High compression
    5: Maximum compression
    """
    doc = fitz.open(input_path)

    # Define compression parameters based on level
    compression_params = {
        1: {"garbage": 1, "clean": True, "deflate": True, "linear": False},
        2: {"garbage": 2, "clean": True, "deflate": True, "linear": True},
        3: {"garbage": 3, "clean": True, "deflate": True, "linear": True},
        4: {"garbage": 4, "clean": True, "deflate": True, "linear": True},
        5: {
            "garbage": 4,
            "clean": True,
            "deflate": True,
            "linear": True,
            "pretty": False,
            "sanitize": True,
        }
    }

    # Apply compression based on level
    params = compression_params[compression_level]

    # Additional compression for images if level is high
    if compression_level >= 4:
        for page_num in range(len(doc)):
            page = doc[page_num]
            img_list = page.get_images()

            for img_index, img in enumerate(img_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image:
                        image_data = base_image["image"]
                        # Convert to PIL Image
                        image = Image.open(io.BytesIO(image_data))

                        # Determine quality based on compression level
                        quality = 60 if compression_level == 4 else 40  # Level 5 uses more aggressive compression

                        # Save with compression
                        output_buffer = io.BytesIO()
                        image.save(output_buffer, format="JPEG", quality=quality, optimize=True)

                        # Replace image in PDF
                        doc.replace_image(xref, output_buffer.getvalue())
                except Exception as e:
                    continue  # Skip problematic images

    # Save the compressed PDF
    doc.save(output_path, **params)
    doc.close()
    return output_path


uploaded_pdf_compress = st.file_uploader("Choose a PDF to compress", type="pdf", key="pdf_compress")

if uploaded_pdf_compress is not None:
    compression_level = st.slider("Select compression level (1: Minimal, 5: Maximum)", 1, 5, 3)

    st.info("""
    Compression Levels:
    - Level 1: Minimal compression (Best quality)
    - Level 2: Low compression
    - Level 3: Medium compression (Balanced)
    - Level 4: High compression
    - Level 5: Maximum compression (Smallest file size)
    """)

    if st.button("Compress PDF"):
        with st.spinner('Compressing PDF...'):
            try:
                # Create temporary files for input and output
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as input_tmp:
                    input_tmp.write(uploaded_pdf_compress.getvalue())
                    input_path = input_tmp.name

                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name

                # Compress the PDF
                compress_pdf(input_path, output_path, compression_level)

                # Get file sizes
                original_size = len(uploaded_pdf_compress.getvalue())

                with open(output_path, 'rb') as f:
                    compressed_content = f.read()
                compressed_size = len(compressed_content)

                # Calculate reduction percentage
                reduction = ((original_size - compressed_size) / original_size) * 100

                # Display compression results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Size", f"{original_size / 1024:.1f} KB")
                with col2:
                    st.metric("Compressed Size", f"{compressed_size / 1024:.1f} KB")
                with col3:
                    st.metric("Reduction", f"{reduction:.1f}%")

                # Create download link
                download_link = create_download_link(compressed_content, "compressed_document.pdf")
                st.success("PDF compressed successfully!")
                st.markdown(download_link, unsafe_allow_html=True)
                st.balloons()

                # Cleanup temporary files
                os.unlink(input_path)
                os.unlink(output_path)

            except Exception as e:
                st.error(f"Error during compression: {str(e)}")
                st.error("Please make sure the PDF is not encrypted and is valid.")


# 3. PDF Watermark Section
st.markdown("---")
st.markdown('<h2 class="fade-in">PDF Watermarker</h2>', unsafe_allow_html=True)

uploaded_pdf_watermark = st.file_uploader("Choose  a PDF to watermark", type="pdf", key="pdf_watermark")
watermark_text = st.text_input("Enter watermark text:")
watermark_color = st.color_picker("Choose watermark color", "#808080")

if uploaded_pdf_watermark is not None and watermark_text:
    if st.button("Add Watermark"):
        with st.spinner('Adding watermark...'):
            try:
                # Create temporary file for the uploaded PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_pdf_watermark.getvalue())
                    pdf_path = tmp_file.name

                # Create PDF reader object
                reader = PyPDF2.PdfReader(pdf_path)
                writer = PyPDF2.PdfWriter()

                # Create watermark
                watermark = PyPDF2.PdfWriter()
                watermark_page = PyPDF2.PdfReader(createWatermark(watermark_text, watermark_color)).pages[0]

                # Apply watermark to each page
                for page in reader.pages:
                    page.merge_page(watermark_page)
                    writer.add_page(page)

                # Save watermarked PDF
                output_path = "watermarked_document.pdf"
                with open(output_path, "wb") as output_file:
                    writer.write(output_file)

                # Create download link
                with open(output_path, "rb") as f:
                    pdf_content = f.read()
                download_link = create_download_link(pdf_content, "watermarked_document.pdf")
                st.success("Watermark added successfully!")
                st.markdown(download_link, unsafe_allow_html=True)
                st.balloons()

                # Cleanup
                os.unlink(pdf_path)
                os.unlink(output_path)

            except Exception as e:
                st.error(f"Error adding watermark: {str(e)}")

# Clean up the temporary PDF file after the app is closed
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")
