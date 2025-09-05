import validators
import streamlit as st
import os
import subprocess
from langchain_community.document_loaders import UnstructuredURLLoader, TextLoader
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
# --- ADD THIS NEW IMPORT ---
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text", page_icon="🦜")
st.title("Summarize Text From YT or Website")
st.subheader('Summarize any URL with a click')

# Sidebar for API Key Input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# Main Input for URL
generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="Enter YouTube or Website URL")

# Main button to trigger summarization
if st.button("Summarize Content"):
    # 1. Validate all inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide your Groq API key and a URL to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Processing......"):
                # 2. Load the document content
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    output_template = "transcript.%(ext)s"
                    subprocess.run(
                        [
                            "yt-dlp", "--write-auto-sub", "--sub-format", "vtt",
                            "--skip-download", "-o", output_template, generic_url
                        ],
                        capture_output=True, text=True, check=True
                    )
                    files = [f for f in os.listdir('.') if f.startswith('transcript') and f.endswith('.vtt')]
                    if not files:
                        st.error("Transcript file not found after download attempt.")
                        st.stop()
                    transcript_filename = files[0]
                    loader = TextLoader(transcript_filename, encoding='utf-8')
                    loaded_documents = loader.load()
                    os.remove(transcript_filename)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    loaded_documents = loader.load()

                # --- START: NEW MANUAL TEXT SPLITTING BLOCK ---
                # 3. Manually split the document into smaller chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,  # The size of each chunk in characters
                    chunk_overlap=150 # Characters to overlap between chunks
                )
                split_docs = text_splitter.split_documents(loaded_documents)
                # --- END: NEW MANUAL TEXT SPLITTING BLOCK ---

                # 4. Initialize the LLM and Summarization Chain
                llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)
                
                # Use "refine" as it's the most robust chain type for this task
                chain = load_summarize_chain(llm, chain_type="refine")
                
                # 5. Run the chain on the MANUALLY SPLIT documents
                output_summary = chain.run(split_docs)

                # 6. Display the summary
                st.success("Summary:")
                st.write(output_summary)

        except subprocess.CalledProcessError:
             st.error("Failed to download transcript. The video might not have subtitles or is unavailable.")
        except Exception as e:
            st.exception(f"An error occurred: {e}")