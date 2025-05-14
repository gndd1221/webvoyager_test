import os
import re
import base64
from typing import List, Dict, Optional, Tuple, Any, Protocol
import pymupdf4llm
import pdfplumber
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# 改用獨立套件導入
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import google.generativeai as genai
from typing_extensions import Literal
import logging
from instruction_manual_generator import InstructionManualGenerator

class EmbeddingModel(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...

class DocumentConverter:
    # Unchanged from original
    def __init__(self):
        pass

    def pdf_to_markdown(self, pdf_path: str, output_dir: str = "output", image_dir: str = "images", image_format: str = "png", dpi: int = 300) -> Tuple[str, List[str]]:
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, image_dir)
        os.makedirs(image_path, exist_ok=True)
        pdf_filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(pdf_filename)[0]
        markdown_content = pymupdf4llm.to_markdown(pdf_path, write_images=True, image_path=image_path, image_format=image_format, dpi=dpi)
        markdown_content = markdown_content.replace(f"{output_dir}/{image_dir}/", f"{image_dir}/")
        output_md_path = os.path.join(output_dir, f"{base_name}.md")
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        image_paths = self._extract_image_paths(markdown_content)
        return output_md_path, image_paths

    def _extract_image_paths(self, markdown_content: str) -> List[str]:
        image_pattern = r"!\[[^\]]*\]\(([^)]+)\)"
        matches = re.findall(image_pattern, markdown_content)
        return matches

    def pdf_to_text(self, pdf_path: str) -> List[Tuple[int, str]]:
        result = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    result.append((page_num, text))
        return result

    def extract_toc(self, pdf_path: str) -> Optional[List[Tuple[int, str, int, int]]]:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()
        if not toc:
            doc.close()
            return None
        result = []
        for i, (level, title, start_page) in enumerate(toc):
            start_page = max(0, start_page - 1)
            end_page = toc[i + 1][2] - 2 if i + 1 < len(toc) else len(doc) - 1
            result.append((level, title, start_page, end_page))
        doc.close()
        return result

class ImageProcessor:
    def __init__(self, gemini_api_key: str, logger: logging.Logger, description_model: str = "gemini-1.5-flash"):
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(description_model)
        self.logger = logger

    def get_image_descriptions(self, output_dir: str, image_paths: List[str]) -> Dict[str, str]:
        descriptions = {}
        for img_path in image_paths:
            try:
                description = self.describe_image(os.path.join(output_dir, img_path))
                descriptions[img_path] = description
            except Exception as e:
                self.logger.error(f"Error processing image {img_path}: {e}")
                descriptions[img_path] = "Unable to describe image"
        return descriptions

    def describe_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        response = self.gemini_model.generate_content([
            "Briefly describe this image:",
            {"inline_data": {"mime_type": f"image/{self._get_image_type(image_path)}", "data": image_base64}}
        ])
        return response.text

    def _get_image_type(self, image_path: str) -> str:
        _, ext = os.path.splitext(image_path)
        if ext.lower() == ".png":
            return "png"
        elif ext.lower() in [".jpg", ".jpeg"]:
            return "jpeg"
        else:
            raise ValueError(f"Unsupported image type: {ext}")

    def enhance_markdown_with_descriptions(self, markdown_path: str, image_descriptions: Dict[str, str]) -> str:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        for img_path, description in image_descriptions.items():
            img_pattern = re.escape(os.path.basename(img_path))
            pattern = f'!\\[[^\\]]*\\]\\([^)]*{img_pattern}[^)]*\\)'
            replacement = f'![{self._escape_markdown(description)}]({img_path})'
            content = re.sub(pattern, replacement, content)
        output_path = os.path.splitext(markdown_path)[0] + "_enhanced.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path

    def _escape_markdown(self, text: str) -> str:
        text = text.replace("\n", " ")
        special_chars = r"\[](){}*_#<>|!"
        return re.sub(f"([{re.escape(special_chars)}])", r"\\\1", text)

class EmbeddingFactory:
    @staticmethod
    def create(embedding_type: Literal["bge-m3"], model_kwargs: Optional[Dict[str, Any]] = None) -> EmbeddingModel:
        if embedding_type == "bge-m3":
            default_kwargs = {"model_name": "BAAI/bge-m3", "model_kwargs": {"device": "cpu"}, "encode_kwargs": {"normalize_embeddings": True}}
            if model_kwargs:
                default_kwargs.update(model_kwargs)
            return HuggingFaceEmbeddings(**default_kwargs)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

class TextSplitter:
    # Unchanged from original
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False)

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        if metadata is None:
            metadata = {}
        doc = Document(page_content=text, metadata=metadata)
        return self.splitter.split_documents([doc])

    def split_documents(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)

class RAGEngine:
    # Unchanged from original except embedding model initialization
    def __init__(self, embedding_model: EmbeddingModel, persist_directory: str = "./chroma_db"):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.text_splitter = TextSplitter()
        self.vectordb = self._load_or_create_db()

    def _load_or_create_db(self) -> Optional[Chroma]:
        if os.path.exists(self.persist_directory):
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_model)
        return None

    def index_document(self, document_path: str, document_type: str, mode: Literal["append", "overwrite"] = "append", metadata: Optional[Dict[str, Any]] = None) -> None:
        if metadata is None:
            metadata = {}
        base_metadata = {"source": os.path.basename(document_path), "full_path": document_path, "type": document_type}
        base_metadata.update(metadata)
        if document_type == "pdf":
            self._index_pdf(document_path, mode, base_metadata)
        elif document_type == "markdown":
            self._index_markdown(document_path, mode, base_metadata)
        else:
            raise ValueError(f"Unsupported document type: {document_type}")

    def _index_pdf(self, pdf_path: str, mode: str = "append", metadata: Dict[str, Any] = None) -> None:
        converter = DocumentConverter()
        toc = converter.extract_toc(pdf_path)
        documents = []
        if toc:
            doc = fitz.open(pdf_path)
            for level, title, start_page, end_page in toc:
                text = ""
                for page_num in range(start_page, end_page + 1):
                    text += doc[page_num].get_text()
                if text.strip():
                    section_metadata = metadata.copy() if metadata else {}
                    section_metadata.update({"section": title, "level": level, "page_range": f"{start_page + 1}-{end_page + 1}"})
                    doc_obj = Document(page_content=text, metadata=section_metadata)
                    documents.append(doc_obj)
            doc.close()
        else:
            pages = converter.pdf_to_text(pdf_path)
            for page_num, text in pages:
                page_metadata = metadata.copy() if metadata else {}
                page_metadata["page"] = page_num
                doc_obj = Document(page_content=text, metadata=page_metadata)
                documents.append(doc_obj)
        chunks = self.text_splitter.split_documents(documents)
        self._add_to_database(chunks, mode)

    def _index_markdown(self, markdown_path: str, mode: str = "append", metadata: Dict[str, Any] = None) -> None:
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()
        sections = self._split_markdown_by_headers(content)
        documents = []
        for header, text in sections:
            section_metadata = metadata.copy() if metadata else {}
            if header:
                section_metadata["section"] = header
            doc = Document(page_content=text, metadata=section_metadata)
            documents.append(doc)
        chunks = self.text_splitter.split_documents(documents)
        self._add_to_database(chunks, mode)

    def _split_markdown_by_headers(self, content: str) -> List[Tuple[str, str]]:
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        sections = []
        current_header = ""
        current_content = []
        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                if current_content:
                    sections.append((current_header, '\n'.join(current_content)))
                    current_content = []
                current_header = header_match.group(2).strip()
                current_content.append(line)
            else:
                current_content.append(line)
        if current_content:
            sections.append((current_header, '\n'.join(current_content)))
        if not sections:
            sections.append(("", content))
        return sections

    def _add_to_database(self, chunks: List[Document], mode: str) -> None:
        if mode == "append" and self.vectordb is not None:
            self.vectordb.add_documents(chunks)
            # self.vectordb.persist()
        else:
            self.vectordb = Chroma.from_documents(documents=chunks, embedding=self.embedding_model, persist_directory=self.persist_directory)
            # self.vectordb.persist()

    def search(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.vectordb is None:
            raise RuntimeError("No database found. Please index a document first.")
        search_kwargs = {}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        results = self.vectordb.similarity_search(query, k=k, **search_kwargs)
        return [{"content": doc.page_content, "metadata": doc.metadata, "source": doc.metadata.get("source", "Unknown"), "section": doc.metadata.get("section", "N/A"), "page": doc.metadata.get("page", "N/A"), "page_range": doc.metadata.get("page_range", "N/A")} for doc in results]

    def get_stats(self) -> Dict[str, Any]:
        if self.vectordb is None:
            return {"total_documents": 0, "persist_directory": self.persist_directory}
        return {"total_documents": self.vectordb._collection.count(), "persist_directory": self.persist_directory}

class PDFEnhancementPipeline:
    def __init__(self, gemini_api_key: str, logger: logging.Logger, embedding_type: Literal["bge-m3"] = "bge-m3", persist_directory: str = "./chroma_db", image_description_model: str = "gemini-1.5-flash"):
        self.logger = logger
        self.doc_converter = DocumentConverter()
        self.image_processor = ImageProcessor(gemini_api_key=gemini_api_key, logger=self.logger, description_model=image_description_model)
        self.embedding_model = EmbeddingFactory.create(embedding_type=embedding_type)
        self.rag_engine = RAGEngine(embedding_model=self.embedding_model, persist_directory=persist_directory)

    def process_pdf(self, pdf_path: str, output_dir: str = "output", add_image_descriptions: bool = True, index_for_rag: bool = True, rag_mode: Literal["append", "overwrite"] = "append", overwrite_enhanced_md: bool = False) -> Dict[str, Any]:
        result = {"original_pdf": pdf_path, "output_directory": output_dir}
        self.logger.info(f"Converting {pdf_path} to Markdown...")
        markdown_path, image_paths = self.doc_converter.pdf_to_markdown(pdf_path=pdf_path, output_dir=output_dir)
        result["markdown_path"] = markdown_path
        result["image_count"] = len(image_paths)
        if add_image_descriptions and image_paths:
            enhanced_md_path = os.path.splitext(markdown_path)[0] + "_enhanced.md"
            if os.path.exists(enhanced_md_path) and not overwrite_enhanced_md:
                self.logger.info(f"Enhanced Markdown already exists at {enhanced_md_path}, skipping generation.")
                result["enhanced_markdown_path"] = enhanced_md_path
            else:
                self.logger.info(f"Generating descriptions for {len(image_paths)} images...")
                image_descriptions = self.image_processor.get_image_descriptions(output_dir, image_paths)
                self.logger.info("Enhancing Markdown with image descriptions...")
                enhanced_md_path = self.image_processor.enhance_markdown_with_descriptions(markdown_path=markdown_path, image_descriptions=image_descriptions)
                result["enhanced_markdown_path"] = enhanced_md_path
        else:
            result["enhanced_markdown_path"] = None
        if index_for_rag:
            self.logger.info(f"Indexing original PDF for RAG...")
            self.rag_engine.index_document(document_path=pdf_path, document_type="pdf", mode=rag_mode)
            if result.get("enhanced_markdown_path"):
                self.logger.info(f"Indexing enhanced Markdown for RAG...")
                self.rag_engine.index_document(document_path=result["enhanced_markdown_path"], document_type="markdown", mode="append", metadata={"enhanced": True, "original_pdf": pdf_path})
            stats = self.rag_engine.get_stats()
            result["rag_stats"] = stats
        return result

    def search(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.rag_engine.search(query, k, filter_dict)

def main() -> None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set")
        return
    pipeline = PDFEnhancementPipeline(gemini_api_key=api_key, logger=logger, embedding_type="bge-m3", persist_directory="./chroma_db")
    pdf_path = "data/arXiv.pdf"
    output_dir = "output"
    logger.info(f"Starting to process {pdf_path}...")
    result = pipeline.process_pdf(pdf_path=pdf_path, output_dir=output_dir, add_image_descriptions=True, index_for_rag=True, overwrite_enhanced_md=False)
    logger.info("Processing completed:")
    logger.info(f"- Original PDF: {result['original_pdf']}")
    logger.info(f"- Markdown file: {result['markdown_path']}")
    logger.info(f"- Number of processed images: {result['image_count']}")
    if 'enhanced_markdown_path' in result:
        logger.info(f"- Enhanced Markdown: {result['enhanced_markdown_path']}")
    task_goal = "Search for papers on 'neural networks for image processing' in the Computer Science category on ArXiv and report how many were submitted in the last week."
    results = pipeline.search(query=task_goal, k=20)
    filtered_results = [{k: d[k] for k in ["section", "content", "source"] if k in d} for d in results]
    manual_generator = InstructionManualGenerator(gemini_api_key=api_key, task_goal=task_goal, results=filtered_results, logger=logger)
    manual = manual_generator.generate_instruction_manual()
    logger.info(manual)

if __name__ == "__main__":
    main()