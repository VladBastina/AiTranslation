import streamlit as st
import os
import sys
import time
import io # Needed for handling file streams in memory
from pathlib import Path

# --- Import necessary libraries ---
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError:
    print(sys.path)
    print(sys.executable)
    st.error("Error: google-generativeai library not found. Please install it: `pip install google-generativeai`")
    st.stop()

try:
    import pypdf
except ImportError:
    st.error("Error: pypdf library not found. Please install it: `pip install pypdf`")
    st.stop()

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    # Attempt to register a font that supports a wider range of characters
    try:
        # Assume DejaVuSans.ttf is in the same directory as the script
        font_path = Path(__file__).parent / 'DejaVuSans.ttf'
        if font_path.exists():
            pdfmetrics.registerFont(TTFont('DejaVuSans', str(font_path)))
            DEFAULT_FONT = 'DejaVuSans'
            print("Using DejaVuSans font.") # Log to console
        else:
            DEFAULT_FONT = 'Helvetica'
            print("Warning: DejaVuSans.ttf not found. Using Helvetica.")
            # Display warning in Streamlit app as well
            st.warning("⚠️ Warning: DejaVuSans font not found. Non-Latin characters might not render correctly in the output PDF. Consider placing `DejaVuSans.ttf` in the app directory.")
    except Exception as font_e:
        st.warning(f"⚠️ Warning: Error registering font. Using Helvetica. Details: {font_e}")
        DEFAULT_FONT = 'Helvetica'

except ImportError:
    st.error("Error: reportlab library not found. Please install it: `pip install reportlab`")
    st.stop()

# --- Configuration (Moved API Key handling) ---
# GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") # Handled via Streamlit input/secrets later
MODEL_NAME = "gemini-1.5-pro" # Or "gemini-1.5-flash-latest" etc.
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
BATCH_SIZE = 50 # Number of pages to process per batch if PDF is large
API_CALL_DELAY = 0.5 # Optional delay in seconds between API calls
DEFAULT_PDF_PATH = Path(__file__).parent / "default_pharma.pdf" # Path to your default PDF
LANGUAGES = ["russian", "romanian", "english", "german", "french", "spanish"]

# --- Core Functions (Adapted from your script) ---

# Global variable to hold the configured model
gemini_model = None

def configure_gemini(api_key):
    """Configures the Gemini client."""
    global gemini_model
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        gemini_model = genai.GenerativeModel(MODEL_NAME, safety_settings=SAFETY_SETTINGS)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        gemini_model = None # Ensure model is None if config fails
        return False

def extract_text_from_pdf(pdf_file_obj):
    """Extracts text from each page of the PDF file object."""
    page_texts = []
    try:
        reader = pypdf.PdfReader(pdf_file_obj)
        num_pages = len(reader.pages)
        st.info(f"Found {num_pages} page(s) in the PDF.")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    page_texts.append(text.strip())
                else:
                    page_texts.append("") # Keep page count consistent
                status_text.text(f"Extracting text from page {i + 1}/{num_pages}")
                progress_bar.progress((i + 1) / num_pages)

            except Exception as e:
                st.warning(f"Warning: Could not extract text from page {i + 1}: {e}")
                page_texts.append("") # Add empty string on error

        status_text.text("Text extraction complete.")
        return page_texts
    except pypdf.errors.PdfReadError as e:
        st.error(f"Error reading PDF file: {e}. The file might be corrupted, password-protected, or not a valid PDF.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during PDF processing: {e}")
        return None

def extract_text_from_txt(txt_file_obj):
    """Reads text content from a TXT file object."""
    try:
        # Read as bytes first, then decode smartly
        content_bytes = txt_file_obj.read()
        try:
            # Try UTF-8 first
            text = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1 (or cp1252 for Windows files)
                text = content_bytes.decode('latin-1')
                st.warning("Decoded TXT file using 'latin-1'. Some characters might be misinterpreted if the encoding is different.")
            except Exception as decode_err:
                 st.error(f"Error decoding TXT file: {decode_err}. Please ensure it's UTF-8 or Latin-1 encoded.")
                 return None
        st.info(f"Successfully read text file.")
        return text
    except Exception as e:
        st.error(f"An error occurred reading the TXT file: {e}")
        return None

def translate_text_gemini(text, source_lang, target_lang, page_num_for_log=""):
    """Translates text using the Gemini API."""
    global gemini_model
    if gemini_model is None:
        st.error("Gemini model not configured. Cannot translate.")
        return None # Indicate failure

    if not text:
        return "" # Nothing to translate

    log_prefix = f"Page {page_num_for_log}: " if page_num_for_log else "Text block: "

    prompt = f"""Translate the following text from {source_lang} to {target_lang}.
Preserve paragraph breaks where appropriate. Output *only* the translated text, without any introductory phrases like "Here is the translation:", or any explanations or markdown formatting. If the input text is empty or nonsensical for translation, output nothing.

Text to translate:
---
{text}
---

Translation:"""

    try:
        # Optional: Add delay between calls
        if API_CALL_DELAY > 0:
            time.sleep(API_CALL_DELAY)

        response = gemini_model.generate_content(prompt)

        # Robust check for content
        translated_text = ""
        if response.parts:
            translated_text = "".join(part.text for part in response.parts).strip()
        elif hasattr(response, 'text'): # Fallback for simpler response structures
             translated_text = response.text.strip()

        # Handle potential blocking or empty responses even if parts exist but are empty
        if not translated_text:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 st.warning(f"{log_prefix}Translation blocked. Reason: {response.prompt_feedback.block_reason}")
                 return f"[Translation blocked on {log_prefix.strip(':')}: {response.prompt_feedback.block_reason}]"
             else:
                 finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                 if finish_reason == 'STOP':
                     # Don't warn if input was likely just whitespace/empty
                     if text.strip():
                         st.warning(f"{log_prefix}Received no translated content (finish reason STOP). Original text might have been empty or untranslatable.")
                     return "" # Return empty if no content and no blocking
                 else:
                     st.warning(f"{log_prefix}Received empty response from API. Finish Reason: {finish_reason}, Feedback: {response.prompt_feedback}")
                     return f"[Translation failed on {log_prefix.strip(':')}: Empty API response]"

        return translated_text

    except google_exceptions.ResourceExhausted as e:
         st.error(f"{log_prefix}Error: Gemini API quota exceeded: {e}. Consider increasing API_CALL_DELAY or checking your quota.")
         return f"[Translation failed on {log_prefix.strip(':')}: Quota Exceeded - {e}]"
    except google_exceptions.InvalidArgument as e:
         st.error(f"{log_prefix}Error: Invalid argument passed to Gemini API: {e}")
         # st.error(f"     Problematic text snippet (first 100 chars): {text[:100]}...") # Debugging
         return f"[Translation failed on {log_prefix.strip(':')}: Invalid Argument - {e}]"
    except Exception as e:
        st.error(f"{log_prefix}Error during Gemini API call: {e}")
        return f"[Translation failed on {log_prefix.strip(':')}: {e}]"


def translate_pages_in_batches(original_pages_text, source_lang, target_lang):
    """Translates list of page texts, batching if necessary."""
    global gemini_model
    if gemini_model is None:
        st.error("Gemini model not configured. Cannot translate.")
        return None

    translated_pages = []
    total_pages = len(original_pages_text)

    if total_pages == 0:
        st.warning("No text pages found to translate.")
        return []

    st.info(f"Starting translation of {total_pages} page(s)...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    if total_pages <= BATCH_SIZE:
        # Single batch processing
        for i, text in enumerate(original_pages_text):
            page_num = i + 1
            status_text.text(f"Translating page {page_num}/{total_pages}...")
            if not text.strip():
                # st.write(f"    - Page {page_num}: Skipping empty page.") # Optional verbose logging
                translated_pages.append("")
            else:
                translated = translate_text_gemini(text, source_lang, target_lang, page_num_for_log=page_num)
                if translated is None: return None # Propagate failure
                translated_pages.append(translated)
            progress_bar.progress((i + 1) / total_pages)
    else:
        # Batch processing
        num_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
        st.info(f"Translating {total_pages} pages in {num_batches} batches of up to {BATCH_SIZE}...")
        pages_processed = 0
        for batch_num in range(num_batches):
            start_index = batch_num * BATCH_SIZE
            end_index = min((batch_num + 1) * BATCH_SIZE, total_pages)
            batch_texts = original_pages_text[start_index:end_index]
            start_page = start_index + 1
            end_page = end_index

            # st.write(f"-- Processing Batch {batch_num + 1}/{num_batches} (Pages {start_page}-{end_page}) --")

            for i, text in enumerate(batch_texts):
                current_page_number = start_index + i + 1
                status_text.text(f"Translating page {current_page_number}/{total_pages} (Batch {batch_num + 1}/{num_batches})...")
                if not text.strip():
                    # st.write(f"    - Page {current_page_number}: Skipping empty page.")
                    translated_pages.append("")
                else:
                    translated = translate_text_gemini(text, source_lang, target_lang, page_num_for_log=current_page_number)
                    if translated is None: return None # Propagate failure
                    translated_pages.append(translated)

                pages_processed += 1
                progress_bar.progress(pages_processed / total_pages)
            # st.write(f"-- Finished Batch {batch_num + 1}/{num_batches} --")

    status_text.text("Translation step complete.")
    return translated_pages


def create_pdf_from_text(translated_pages):
    """Creates a new PDF document from the translated text pages in memory."""
    pdf_buffer = io.BytesIO()
    try:
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        style = styles["Normal"]
        style.fontName = DEFAULT_FONT
        style.fontSize = 10
        style.alignment = TA_JUSTIFY

        style_bold = styles["Heading2"] # Use a heading style for page markers
        style_bold.fontName = DEFAULT_FONT
        style_bold.fontSize = 8 # Make header smaller
        style_bold.alignment = TA_JUSTIFY

        story = []
        st.info(f"Reconstructing PDF with {len(translated_pages)} page(s)...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, page_text in enumerate(translated_pages):
            page_num = i + 1
            status_text.text(f"Adding translated page {page_num}/{len(translated_pages)} to PDF...")

            # Add a header indicating the original page number
            story.append(Paragraph(f"--- Translated Page {page_num} ---", style_bold))
            story.append(Spacer(1, 6)) # Add smaller space after header

            if page_text:
                # Replace newline characters with <br/> tags for ReportLab Paragraphs
                formatted_text = page_text.replace('\n', '<br/>\n')
                try:
                     para = Paragraph(formatted_text, style)
                     story.append(para)
                except Exception as e:
                     st.warning(f"Warning: Could not add text from page {page_num} to PDF (potential encoding/font issue): {e}")
                     try:
                         error_para = Paragraph(f"[Could not render text for page {page_num} due to error. See logs/warnings.]", style)
                         story.append(error_para)
                     except: pass # Skip if even the error message fails
            else:
                story.append(Paragraph(f"[No translatable text found or translation failed for page {page_num}]", style))

            # Add a page break after each page's content, except the last one
            if i < len(translated_pages) - 1:
                story.append(PageBreak())

            progress_bar.progress((i + 1) / len(translated_pages))

        doc.build(story)
        status_text.text("PDF reconstruction complete.")
        pdf_buffer.seek(0) # Rewind the buffer to the beginning
        return pdf_buffer

    except Exception as e:
        st.error(f"Error creating output PDF: {e}")
        return None

def create_txt_from_text(translated_text):
    """Creates a TXT file content in memory."""
    try:
        txt_buffer = io.StringIO()
        txt_buffer.write(translated_text)
        txt_buffer.seek(0)
        # We need BytesIO for download button, so encode it
        txt_bytes_buffer = io.BytesIO(txt_buffer.getvalue().encode('utf-8'))
        st.info("TXT file content prepared.")
        return txt_bytes_buffer
    except Exception as e:
        st.error(f"Error creating output TXT: {e}")
        return None


# --- Streamlit App UI ---
st.title("📄 Document Translator")

configure_gemini(None)

st.sidebar.image('zega_logo.PNG',use_container_width=True)

st.sidebar.markdown("---") # Separator

# --- File Input Options ---
st.sidebar.subheader("📁 Input File")
use_default = st.sidebar.checkbox("Use default Russian pharma PDF", value=False)

uploaded_file = None
source_lang_selected = None
input_file_type = None # To track 'pdf' or 'txt'

if use_default:
    if not DEFAULT_PDF_PATH.exists():
        st.sidebar.error(f"Default PDF '{DEFAULT_PDF_PATH.name}' not found in the app directory!")
        st.stop()
    else:
        st.sidebar.info(f"Using default file: `{DEFAULT_PDF_PATH.name}`")
        source_lang_selected = "russian" # Default file is Russian
        input_file_type = "pdf"
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload your PDF or TXT file",
        type=["pdf", "txt"],
        accept_multiple_files=False
    )
    if uploaded_file:
        input_file_type = uploaded_file.type.split('/')[-1].lower() # pdf or plain (->txt)
        if input_file_type == 'plain':
            input_file_type = 'txt'

        # Dropdown for source language ONLY if uploading
        st.sidebar.markdown("👇 Select the **source** language of your uploaded file:")
        source_lang_selected = st.sidebar.selectbox(
            "Source Language",
            options=[""] + LANGUAGES, # Add empty option for prompt
            index=0, # Default to empty
            key="source_lang_uploader"
        )
        if not source_lang_selected:
             st.sidebar.warning("Please select the source language of your document.")


st.sidebar.markdown("---") # Separator

# --- Target Language Selection ---
st.sidebar.subheader("🎯 Target Language")
target_lang_selected = None
# Ensure a source is defined before showing target selection
if source_lang_selected:
    target_lang_selected = st.sidebar.selectbox(
        "Translate To",
        options=[""] + [lang for lang in LANGUAGES if lang != source_lang_selected], # Exclude source lang
        index=0, # Default to empty
        key="target_lang",
        help="Select the language you want to translate the document into."
    )
    if not target_lang_selected:
        st.sidebar.warning("Please select the target language.")
else:
    st.sidebar.info("Select or upload a file and its source language first.")


st.sidebar.markdown("---") # Separator

# --- Translate Button ---
translate_button = st.sidebar.button("Translate Document", disabled=(gemini_model is None or not target_lang_selected))

if not source_lang_selected:
     st.sidebar.markdown("_(Select/Upload file and source language to enable translation)_")
elif not target_lang_selected:
     st.sidebar.markdown("_(Select target language to enable translation)_")

# --- Main Area for Processing and Results ---
if translate_button:
    st.subheader("🚀 Translation Progress")
    output_buffer = None
    output_filename = "translation_failed" # Default filename

    with st.spinner("Processing... Please wait."):
        # 1. Get Input Data
        input_data = None
        if use_default:
            try:
                with open(DEFAULT_PDF_PATH, "rb") as f:
                    input_data = io.BytesIO(f.read())
                st.write(f"Processing default file: {DEFAULT_PDF_PATH.name} (PDF)")
            except Exception as e:
                st.error(f"Error reading default PDF: {e}")
                st.stop()
        elif uploaded_file:
            input_data = io.BytesIO(uploaded_file.getvalue()) # Use BytesIO for consistency
            st.write(f"Processing uploaded file: {uploaded_file.name} ({input_file_type.upper()})")
        else:
            st.error("No input file selected!")
            st.stop()

        # Basic validation passed in UI, but double-check
        if not input_data or not source_lang_selected or not target_lang_selected:
             st.error("Missing required input (file, source language, or target language).")
             st.stop()
        if source_lang_selected == target_lang_selected:
             st.error("Source and Target languages cannot be the same.")
             st.stop()

        # --- Start Processing based on file type ---
        if input_file_type == "pdf":
            st.markdown("---")
            st.write("**Step 1: Extracting Text from PDF...**")
            original_pages = extract_text_from_pdf(input_data)

            if original_pages is not None:
                st.markdown("---")
                st.write(f"**Step 2: Translating {len(original_pages)} pages from {source_lang_selected} to {target_lang_selected}...**")
                translated_pages = translate_pages_in_batches(original_pages, source_lang_selected, target_lang_selected)

                if translated_pages is not None:
                    st.markdown("---")
                    st.write("**Step 3: Creating Translated PDF...**")
                    output_buffer = create_pdf_from_text(translated_pages)
                    if output_buffer:
                        output_filename = f"{Path(uploaded_file.name if uploaded_file else DEFAULT_PDF_PATH.name).stem}_translated_{target_lang_selected}.pdf"
                        st.success("✅ Translation and PDF creation successful!")
                else:
                     st.error("Translation failed. Cannot create PDF.")
            else:
                st.error("Text extraction failed. Cannot proceed.")

        elif input_file_type == "txt":
            st.markdown("---")
            st.write("**Step 1: Reading Text from TXT...**")
            original_text = extract_text_from_txt(input_data)

            if original_text is not None:
                 st.markdown("---")
                 st.write(f"**Step 2: Translating text from {source_lang_selected} to {target_lang_selected}...**")
                 # Use the single text translation function - treat TXT as one block
                 status_text_txt = st.empty()
                 status_text_txt.text("Sending text to translation API...")
                 translated_text = translate_text_gemini(original_text, source_lang_selected, target_lang_selected, page_num_for_log="TXT content")
                 status_text_txt.text("Translation received.")


                 if translated_text is not None: # Check if translation call succeeded
                     st.markdown("---")
                     st.write("**Step 3: Creating Translated TXT file...**")
                     output_buffer = create_txt_from_text(translated_text)
                     if output_buffer:
                        output_filename = f"{Path(uploaded_file.name).stem}_translated_{target_lang_selected}.txt"
                        st.success("✅ Translation and TXT creation successful!")
                 else:
                     st.error("Translation failed. Cannot create TXT file.")
            else:
                st.error("Reading TXT file failed. Cannot proceed.")

        else:
            st.error(f"Unsupported file type: {input_file_type}")

    # --- Offer Download ---
    if output_buffer:
        st.markdown("---")
        st.subheader("📥 Download Result")
        file_mime = "application/pdf" if output_filename.endswith(".pdf") else "text/plain"
        st.download_button(
            label=f"Download {output_filename}",
            data=output_buffer,
            file_name=output_filename,
            mime=file_mime,
        )
        # Display a snippet of the translation (optional)
        # try:
        #     if output_filename.endswith(".pdf"):
        #         st.info("PDF generated. Download to view content.")
        #     else: # TXT file
        #         output_buffer.seek(0)
        #         snippet = output_buffer.read(500).decode('utf-8', errors='ignore')
        #         st.text_area("Translation Snippet:", snippet + "...", height=200)
        # except Exception as e:
        #     st.warning(f"Could not display snippet: {e}")

# --- Initial Instructions ---
if not translate_button:
    st.markdown(
        """
        ## How to Use:

        1.  **Choose Input:**
            *   Check the box to use the **default Russian pharma PDF**.
            *   Or, **upload** your own PDF or TXT file using the uploader.
        2.  **Select Languages:**
            *   If uploading, select the **source language** of your file.
            *   Select the **target language** you want to translate to.
        3.  **Translate:** Click the "Translate Document" button in the sidebar.
        4.  **Download:** Once processed, a download button for the translated file will appear.

        **Note:**
        *   PDF translation attempts to preserve page structure but loses original formatting (images, fonts, layout).
        """
    )