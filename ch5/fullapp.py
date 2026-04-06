# Chapter 5: Building an ArXiv Paper Search System with PostgreSQL pgvector
# O'Reilly Vector Database Book - Complete Code Examples

# ================================================================================
# SECTION 1: INSTALLATION AND SETUP
# ================================================================================

# Installation commands for different systems
"""
# Ubuntu/Debian
sudo apt install postgresql-common
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
sudo apt install postgresql-14 postgresql-server-dev-14
sudo apt install postgresql-14-pgvector

# RHEL/CentOS
sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-8-x86_64/pgdg-redhat-repo-latest.noarch.rpm
sudo dnf install -y postgresql14-server postgresql14-devel pgvector

# Docker
FROM postgres:14
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-14
RUN git clone https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install
"""

# ================================================================================
# SECTION 2: PGVECTOR EXTENSION AND BASIC OPERATIONS
# ================================================================================

import psycopg2
import numpy as np

# Enable pgvector extension
def setup_pgvector(conn):
    """Enable pgvector extension in PostgreSQL"""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

# Verify installation
def verify_pgvector(conn):
    """Check if pgvector is installed"""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        result = cur.fetchone()
        return result is not None

# ================================================================================
# SECTION 3: VECTOR DATA TYPES AND OPERATIONS
# ================================================================================

# SQL for vector operations
vector_operations_sql = """
-- Creating vectors
SELECT ARRAY[1,2,3]::vector;
SELECT '[1,2,3]'::vector;
SELECT vector(ARRAY[1,2,3]);

-- Creating table with vector column
CREATE TABLE items (
    id bigserial PRIMARY KEY,
    embedding vector(384)  -- specify dimension
);

-- Vector operations
SELECT embedding + other_embedding AS sum_vector FROM items;
SELECT embedding - other_embedding AS diff_vector FROM items;
SELECT embedding * 2.0 AS scaled_vector FROM items;

-- Distance functions
SELECT embedding <-> query_vector AS l2_distance FROM items;  -- L2 distance
SELECT embedding <#> query_vector AS ip_distance FROM items;  -- Inner product
SELECT embedding <=> query_vector AS cosine_distance FROM items;  -- Cosine distance
"""

# Python functions for vector operations
def numpy_to_pgvector(array):
    """Convert numpy array to pgvector format"""
    return f"ARRAY[{','.join(map(str, array))}]::vector"

def normalize_vector(conn, vector):
    """Normalize a vector using PostgreSQL"""
    sql = """
    CREATE OR REPLACE FUNCTION normalize_vector(v vector) 
    RETURNS vector AS $$
    DECLARE
        magnitude float;
    BEGIN
        magnitude := sqrt(l2_distance(v, vector_zeros(array_length(v::float4[], 1))));
        RETURN v * (1.0/magnitude);
    END;
    $$ LANGUAGE plpgsql;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()

# ================================================================================
# SECTION 4: INDEXING STRATEGIES
# ================================================================================

# HNSW Index creation
hnsw_index_sql = """
-- Basic HNSW index creation
CREATE INDEX hnsw_idx ON vectors 
USING hnsw (embedding vector_l2_ops)
WITH (
    dim = 384,              -- Vector dimension
    m = 16,                 -- Max connections per node
    ef_construction = 64,   -- Construction time/quality trade-off
    ef_search = 40         -- Search time/quality trade-off
);

-- HNSW for cosine distance
CREATE INDEX hnsw_cosine_idx ON vectors 
USING hnsw (embedding vector_cosine_ops)
WITH (dim=384, m=16, ef_construction=64);

-- IVF index
CREATE INDEX ivf_idx ON vectors 
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

-- Set search quality at query time
SET hnsw.ef_search = 100;
"""

# Performance monitoring
def monitor_index_performance(conn):
    """Monitor pgvector index performance"""
    sql = """
    SELECT 
        pg_size_pretty(pg_relation_size('hnsw_idx')) as index_size,
        pg_stat_get_numscans('hnsw_idx'::regclass) as num_scans,
        pg_stat_get_tuples_returned('hnsw_idx'::regclass) as tuples_returned
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()

# ================================================================================
# SECTION 5: ARXIV PAPER SCHEMA DESIGN
# ================================================================================

arxiv_schema_sql = """
-- Papers table
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,  -- arxiv id
    title TEXT NOT NULL,
    abstract TEXT,
    published_date DATE NOT NULL,
    updated_date DATE,
    paper_url TEXT,
    pdf_url TEXT,
    category_primary TEXT,
    categories TEXT[],
    download_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    pdf_processed BOOLEAN DEFAULT FALSE,
    embedding vector(768),  -- Paper-level embedding
    CHECK (paper_id ~* '^[\d.]+$')  -- Validate arxiv id format
);

-- Create index on paper embeddings
CREATE INDEX ON papers USING hnsw (embedding vector_cosine_ops)
WITH (dim=768, m=16, ef_construction=64);

-- Authors table
CREATE TABLE authors (
    author_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    affiliation TEXT,
    embedding vector(384),  -- Author embedding based on their papers
    UNIQUE (name, email)
);

-- Create index on author embeddings
CREATE INDEX ON authors USING hnsw (embedding vector_cosine_ops)
WITH (dim=384, m=16, ef_construction=64);

-- Paper-Authors junction table
CREATE TABLE paper_authors (
    paper_id TEXT REFERENCES papers(paper_id),
    author_id INTEGER REFERENCES authors(author_id),
    author_position INTEGER NOT NULL,
    is_corresponding BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (paper_id, author_id)
);

-- Indexes for paper_authors
CREATE INDEX ON paper_authors(author_id);
CREATE INDEX ON paper_authors(paper_id, author_position);

-- Paper sections table
CREATE TABLE paper_sections (
    section_id SERIAL PRIMARY KEY,
    paper_id TEXT REFERENCES papers(paper_id),
    section_type TEXT NOT NULL,  -- abstract, introduction, methods, etc.
    section_number INTEGER,
    title TEXT,
    content TEXT,
    embedding vector(768),  -- Section-level embedding
    UNIQUE (paper_id, section_type, section_number)
);

-- Create index on section embeddings
CREATE INDEX ON paper_sections USING hnsw (embedding vector_cosine_ops)
WITH (dim=768, m=16, ef_construction=64);

-- Images table
CREATE TABLE paper_images (
    image_id SERIAL PRIMARY KEY,
    paper_id TEXT REFERENCES papers(paper_id),
    section_id INTEGER REFERENCES paper_sections(section_id),
    figure_number TEXT,
    caption TEXT,
    image_data BYTEA,
    embedding vector(512),  -- Image embedding from vision model
    metadata JSONB,  -- Store additional image metadata
    position_in_text INTEGER,
    UNIQUE (paper_id, figure_number)
);

-- Create index on image embeddings
CREATE INDEX ON paper_images USING hnsw (embedding vector_cosine_ops)
WITH (dim=512, m=16, ef_construction=64);

-- Formulas table
CREATE TABLE paper_formulas (
    formula_id SERIAL PRIMARY KEY,
    paper_id TEXT REFERENCES papers(paper_id),
    section_id INTEGER REFERENCES paper_sections(section_id),
    equation_number TEXT,
    latex_source TEXT NOT NULL,
    rendered_image BYTEA,
    embedding vector(128),  -- Formula embedding
    context TEXT,  -- Surrounding text
    position_in_text INTEGER,
    UNIQUE (paper_id, equation_number)
);

-- Create index on formula embeddings
CREATE INDEX ON paper_formulas USING hnsw (embedding vector_cosine_ops)
WITH (dim=128, m=16, ef_construction=64);

-- Tables content
CREATE TABLE paper_tables (
    table_id SERIAL PRIMARY KEY,
    paper_id TEXT REFERENCES papers(paper_id),
    section_id INTEGER REFERENCES paper_sections(section_id),
    table_number TEXT,
    caption TEXT,
    content JSONB,  -- Structured table data
    embedding vector(384),  -- Table content embedding
    position_in_text INTEGER,
    UNIQUE (paper_id, table_number)
);

-- Create index on table embeddings
CREATE INDEX ON paper_tables USING hnsw (embedding vector_cosine_ops)
WITH (dim=384, m=16, ef_construction=64);

-- Citations table
CREATE TABLE citations (
    citation_id SERIAL PRIMARY KEY,
    citing_paper_id TEXT REFERENCES papers(paper_id),
    cited_paper_id TEXT,  -- May not be in our database
    citation_context TEXT,
    section_id INTEGER REFERENCES paper_sections(section_id),
    position_in_text INTEGER,
    confidence FLOAT,  -- Confidence in citation extraction
    CHECK (citing_paper_id != cited_paper_id)  -- Prevent self-citation loops
);

-- Indexes for citations
CREATE INDEX ON citations(citing_paper_id);
CREATE INDEX ON citations(cited_paper_id);

-- Bibliography table
CREATE TABLE bibliography (
    bib_id SERIAL PRIMARY KEY,
    paper_id TEXT REFERENCES papers(paper_id),
    reference_text TEXT NOT NULL,
    parsed_data JSONB,  -- Structured reference data
    doi TEXT,
    matched_paper_id TEXT REFERENCES papers(paper_id),
    UNIQUE (paper_id, reference_text)
);
"""

# ================================================================================
# SECTION 6: ARXIV PAPER COLLECTION PIPELINE
# ================================================================================

import arxiv
import asyncio
import aiohttp
import logging
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import PyPDF2
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import backoff
import fitz  # PyMuPDF
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import configparser
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PaperMetadata:
    """Data class for paper metadata"""
    paper_id: str
    title: str
    abstract: str
    authors: List[Dict[str, str]]
    categories: List[str]
    submission_date: datetime
    last_updated_date: datetime
    pdf_url: str

class ArxivPipeline:
    def __init__(self, db_config: Dict[str, str], model_name: str = 'all-mpnet-base-v2'):
        """Initialize the pipeline with database configuration and embedding model."""
        self.db_config = db_config
        self.embedding_model = SentenceTransformer(model_name)
        self.conn = self._create_db_connection()
        self.session = None
        
    def _create_db_connection(self) -> psycopg2.extensions.connection:
        """Create and return a database connection."""
        try:
            conn = psycopg2.connect(**self.db_config)
            logger.info("Successfully connected to database")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    @backoff.on_exception(backoff.expo, arxiv.ArxivError, max_tries=5)
    async def fetch_papers(self, query: str, max_results: int = 100) -> List[PaperMetadata]:
        """Fetch papers from arXiv API with exponential backoff."""
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            papers = []
            async for result in client.results(search):
                paper = PaperMetadata(
                    paper_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    abstract=result.summary,
                    authors=[{"name": author.name, "affiliation": author.affiliation}
                            for author in result.authors],
                    categories=result.categories,
                    submission_date=result.published,
                    last_updated_date=result.updated,
                    pdf_url=result.pdf_url
                )
                papers.append(paper)

            logger.info(f"Retrieved {len(papers)} papers from arXiv")
            return papers

        except arxiv.ArxivError as e:
            logger.error(f"Error fetching papers from arXiv: {e}")
            raise

    async def download_pdf(self, paper: PaperMetadata) -> Optional[bytes]:
        """Download PDF with rate limiting and retry logic."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(paper.pdf_url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error(f"Failed to download PDF for {paper.paper_id}: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Error downloading PDF for {paper.paper_id}: {e}")
                return None

    def process_pdf(self, pdf_data: bytes) -> Dict[str, Any]:
        """Extract content and structure from PDF."""
        try:
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            
            # Initialize containers for different content types
            sections = []
            images = []
            formulas = []
            tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text and structure
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    # Process text blocks
                    if block["type"] == 0:  # Text
                        sections.append({
                            "content": block["text"],
                            "page_number": page_num + 1,
                            "bbox": block["bbox"]
                        })
                    
                    # Extract images
                    image_list = page.get_images()
                    for img_idx, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            image = doc.extract_image(xref)
                            images.append({
                                "data": image["image"],
                                "page_number": page_num + 1,
                                "format": image["ext"],
                                "width": image["width"],
                                "height": image["height"]
                            })
                        except Exception as e:
                            logger.warning(f"Failed to extract image: {e}")

            return {
                "sections": sections,
                "images": images,
                "formulas": formulas,  # Would need specialized formula extraction
                "tables": tables      # Would need specialized table extraction
            }

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for text content."""
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def store_paper(self, paper: PaperMetadata, content: Dict[str, Any]) -> None:
        """Store paper metadata and content in database."""
        try:
            # Generate abstract embedding
            abstract_embedding = self.generate_embeddings(paper.abstract)

            # Store paper metadata
            with self.conn.cursor() as cur:
                # Insert paper
                cur.execute("""
                    INSERT INTO papers (
                        paper_id, title, abstract, abstract_embedding,
                        submission_date, last_updated_date, categories
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (paper_id) DO UPDATE SET
                        last_updated_date = EXCLUDED.last_updated_date
                    RETURNING paper_id
                """, (
                    paper.paper_id, paper.title, paper.abstract, abstract_embedding.tolist(),
                    paper.submission_date, paper.last_updated_date, paper.categories
                ))

                # Insert authors
                for author in paper.authors:
                    cur.execute("""
                        INSERT INTO authors (name, affiliation)
                        VALUES (%s, %s)
                        ON CONFLICT (name) DO UPDATE SET
                            affiliation = EXCLUDED.affiliation
                        RETURNING author_id
                    """, (author["name"], author.get("affiliation")))
                    
                    author_id = cur.fetchone()[0]
                    
                    # Link author to paper
                    cur.execute("""
                        INSERT INTO paper_authors (paper_id, author_id, author_position)
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (paper.paper_id, author_id, paper.authors.index(author)))

                # Store sections with embeddings
                for section in content["sections"]:
                    section_embedding = self.generate_embeddings(section["content"])
                    cur.execute("""
                        INSERT INTO paper_sections (
                            paper_id, content, content_embedding,
                            page_number, sequence_order
                        ) VALUES (%s, %s, %s, %s, %s)
                    """, (
                        paper.paper_id, section["content"], section_embedding.tolist(),
                        section["page_number"], content["sections"].index(section)
                    ))

                # Store images
                for image in content["images"]:
                    cur.execute("""
                        INSERT INTO paper_images (
                            paper_id, image_data, page_number,
                            width, height, format
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        paper.paper_id, image["data"], image["page_number"],
                        image["width"], image["height"], image["format"]
                    ))

            self.conn.commit()
            logger.info(f"Successfully stored paper {paper.paper_id}")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing paper {paper.paper_id}: {e}")
            raise

    async def run_pipeline(self, query: str, max_results: int = 100) -> None:
        """Run the complete pipeline."""
        try:
            # Fetch papers
            papers = await self.fetch_papers(query, max_results)
            
            # Process each paper
            for paper in papers:
                # Download PDF
                pdf_data = await self.download_pdf(paper)
                if pdf_data is None:
                    continue
                
                # Process PDF content
                content = self.process_pdf(pdf_data)
                
                # Store paper and content
                await self.store_paper(paper, content)
                
                # Rate limiting
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()

# ================================================================================
# SECTION 7: CONTENT EXTRACTION AND EMBEDDING GENERATION
# ================================================================================

import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
import io
import re
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import camelot
import pdfplumber
from latex2mathml.converter import convert as latex2mathml
import logging
from collections import defaultdict

@dataclass
class ExtractedSection:
    """Represents a section of text from the paper"""
    heading: str
    content: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    embedding: Optional[np.ndarray] = None

@dataclass
class ExtractedImage:
    """Represents an extracted image with its metadata"""
    image_data: bytes
    caption: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    reference_text: str
    embedding: Optional[np.ndarray] = None

@dataclass
class ExtractedFormula:
    """Represents a mathematical formula"""
    latex_source: str
    mathml_source: str
    is_inline: bool
    page_number: int
    bbox: Tuple[float, float, float, float]
    reference_text: str
    embedding: Optional[np.ndarray] = None

@dataclass
class ExtractedTable:
    """Represents a table from the paper"""
    table_data: List[List[str]]
    caption: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    reference_text: str
    embedding: Optional[np.ndarray] = None

class ContentProcessor:
    def __init__(self):
        """Initialize content processor with necessary models"""
        # Text embedding model
        self.text_embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # Image embedding model (CLIP)
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Regular expressions for content detection
        self.section_pattern = re.compile(r'^(?:\d+\.)?\s*([A-Z][^\n]+)$', re.MULTILINE)
        self.formula_pattern = re.compile(r'\$(.*?)\$|\\[(.*?)\\]|\\((.*?)\\)', re.DOTALL)
        self.reference_pattern = re.compile(r'(Fig\.|Figure|Table|Equation)\s+(\d+[a-z]?)')

    def extract_sections(self, doc: fitz.Document) -> List[ExtractedSection]:
        """Extract sections with hierarchical structure"""
        sections = []
        current_section = None
        current_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] == 0:  # Text block
                    text = block["text"].strip()
                    if text:
                        # Check if this is a section heading
                        if self.section_pattern.match(text):
                            # Save previous section if exists
                            if current_section and current_content:
                                sections.append(ExtractedSection(
                                    heading=current_section,
                                    content="\n".join(current_content),
                                    page_number=page_num,
                                    bbox=block["bbox"]
                                ))
                            current_section = text
                            current_content = []
                        else:
                            current_content.append(text)
        
        # Add last section
        if current_section and current_content:
            sections.append(ExtractedSection(
                heading=current_section,
                content="\n".join(current_content),
                page_number=len(doc) - 1,
                bbox=(0, 0, 0, 0)  # Placeholder bbox
            ))
        
        return sections

    def extract_images(self, doc: fitz.Document) -> List[ExtractedImage]:
        """Extract images with captions and references"""
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get image locations and captions
            image_list = page.get_images()
            blocks = page.get_text("dict")["blocks"]
            
            # Find captions (text blocks near images)
            captions = []
            for block in blocks:
                if block["type"] == 0:  # Text block
                    text = block["text"].strip()
                    if text.startswith("Figure") or text.startswith("Fig."):
                        captions.append((text, block["bbox"]))
            
            # Extract images and match with captions
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # Find nearest caption
                    nearest_caption = min(captions, 
                                       key=lambda x: abs(x[1][1] - img[1]),  # Compare y-coordinates
                                       default=("", (0,0,0,0)))
                    
                    images.append(ExtractedImage(
                        image_data=base_image["image"],
                        caption=nearest_caption[0],
                        page_number=page_num,
                        bbox=img[1],  # bbox from image list
                        reference_text=self._find_references(blocks, f"Fig. {img_idx + 1}")
                    ))
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image: {e}")
        
        return images

    def extract_formulas(self, doc: fitz.Document) -> List[ExtractedFormula]:
        """Extract mathematical formulas"""
        formulas = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Find all LaTeX formulas
            matches = self.formula_pattern.finditer(text)
            for match in matches:
                latex = match.group(1) or match.group(2) or match.group(3)
                try:
                    mathml = latex2mathml(latex)
                    
                    # Determine if formula is inline
                    is_inline = bool(match.group(1) or match.group(3))
                    
                    # Get approximate bbox
                    start_pos = match.start()
                    end_pos = match.end()
                    start_bbox = page.get_text("dict", clip=(0, 0, page.rect.width, start_pos))
                    end_bbox = page.get_text("dict", clip=(0, 0, page.rect.width, end_pos))
                    
                    formulas.append(ExtractedFormula(
                        latex_source=latex,
                        mathml_source=mathml,
                        is_inline=is_inline,
                        page_number=page_num,
                        bbox=(start_bbox["bbox"], end_bbox["bbox"]),
                        reference_text=self._find_references(page.get_text("dict")["blocks"], "Equation")
                    ))
                    
                except Exception as e:
                    logger.warning(f"Failed to process formula: {e}")
        
        return formulas

    def extract_tables(self, pdf_path: str) -> List[ExtractedTable]:
        """Extract tables using multiple methods"""
        tables = []
        
        # Try Camelot first
        try:
            camelot_tables = camelot.read_pdf(pdf_path, pages='all')
            for table in camelot_tables:
                tables.append(ExtractedTable(
                    table_data=table.data,
                    caption=self._find_table_caption(table),
                    page_number=table.page,
                    bbox=table._bbox,
                    reference_text=""  # Will be filled later
                ))
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")
        
        # Fallback to pdfplumber if needed
        if not tables:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        for table in page.extract_tables():
                            tables.append(ExtractedTable(
                                table_data=table,
                                caption="",  # Need to find caption separately
                                page_number=page_num + 1,
                                bbox=page.bbox,
                                reference_text=""
                            ))
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
        
        return tables

    def generate_embeddings(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for all content types"""
        # Process sections
        for section in content['sections']:
            section.embedding = self.text_embedder.encode(
                section.heading + " " + section.content
            )
        
        # Process images
        for image in content['images']:
            try:
                # Convert bytes to PIL Image
                img = Image.open(io.BytesIO(image.image_data))
                
                # Generate CLIP embedding
                inputs = self.image_processor(images=img, return_tensors="pt")
                image_features = self.image_model.get_image_features(**inputs)
                image.embedding = image_features.detach().numpy()
            except Exception as e:
                logger.warning(f"Failed to generate image embedding: {e}")
        
        # Process formulas
        for formula in content['formulas']:
            # Generate embedding for formula context
            formula.embedding = self.text_embedder.encode(
                formula.latex_source + " " + formula.reference_text
            )
        
        # Process tables
        for table in content['tables']:
            # Generate embedding for table content and caption
            table_text = "\n".join([" ".join(row) for row in table.table_data])
            table.embedding = self.text_embedder.encode(
                table.caption + " " + table_text
            )
        
        return content

    def _find_references(self, blocks: List[Dict], ref_type: str) -> str:
        """Find references to content in text blocks"""
        references = []
        for block in blocks:
            if block["type"] == 0:  # Text block
                text = block["text"]
                matches = self.reference_pattern.finditer(text)
                for match in matches:
                    if match.group(1) == ref_type:
                        context_start = max(0, match.start() - 50)
                        context_end = min(len(text), match.end() + 50)
                        references.append(text[context_start:context_end])
        return "\n".join(references)

    def _find_table_caption(self, table: Any) -> str:
        """Find caption for a table"""
        # Implementation depends on the specific table extraction method
        return ""  # Placeholder

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF and extract all content with embeddings"""
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Extract content
            content = {
                'sections': self.extract_sections(doc),
                'images': self.extract_images(doc),
                'formulas': self.extract_formulas(doc),
                'tables': self.extract_tables(pdf_path)
            }
            
            # Generate embeddings
            content = self.generate_embeddings(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()

# ================================================================================
# SECTION 8: SEMANTIC SEARCH QUERIES
# ================================================================================

class ArxivSearchEngine:
    def __init__(self, db_params: Dict[str, str]):
        """Initialize the search engine with database connection parameters."""
        self.conn = psycopg2.connect(**db_params)
        self.cursor = self.conn.cursor()

    def search_papers(
        self,
        query_embedding: np.ndarray,
        categories: List[str] = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for papers using vector similarity and optional filters.
        
        Args:
            query_embedding: Vector embedding of the search query
            categories: List of arXiv categories to filter by
            date_from: Start date for publication date filter
            date_to: End date for publication date filter
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing paper information and similarity scores
        """
        query = """
            SELECT 
                p.arxiv_id,
                p.title,
                p.abstract,
                p.publication_date,
                p.categories,
                ps.embedding <=> %s::vector AS similarity,
                (SELECT COUNT(*) FROM citations c WHERE c.cited_paper_id = p.arxiv_id) as citation_count
            FROM 
                papers p
                JOIN paper_sections ps ON p.arxiv_id = ps.arxiv_id
            WHERE 
                ps.section_type = 'abstract'
        """
        
        params = [query_embedding.tolist()]
        
        if categories:
            query += " AND p.categories && %s"
            params.append(categories)
            
        if date_from:
            query += " AND p.publication_date >= %s"
            params.append(date_from)
            
        if date_to:
            query += " AND p.publication_date <= %s"
            params.append(date_to)
            
        query += """
            ORDER BY similarity
            LIMIT %s
        """
        params.append(limit)
        
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        
        return [
            {
                'arxiv_id': r[0],
                'title': r[1],
                'abstract': r[2],
                'publication_date': r[3],
                'categories': r[4],
                'similarity': r[5],
                'citation_count': r[6]
            }
            for r in results
        ]

    def find_similar_papers(self, arxiv_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find papers similar to a given paper by its arxiv_id."""
        query = """
            WITH paper_embedding AS (
                SELECT embedding 
                FROM paper_sections 
                WHERE arxiv_id = %s AND section_type = 'abstract'
            )
            SELECT 
                p.arxiv_id,
                p.title,
                p.abstract,
                ps.embedding <=> (SELECT embedding FROM paper_embedding) AS similarity
            FROM 
                papers p
                JOIN paper_sections ps ON p.arxiv_id = ps.arxiv_id
            WHERE 
                ps.section_type = 'abstract'
                AND p.arxiv_id != %s
            ORDER BY 
                similarity
            LIMIT %s
        """
        
        self.cursor.execute(query, (arxiv_id, arxiv_id, limit))
        results = self.cursor.fetchall()
        
        return [
            {
                'arxiv_id': r[0],
                'title': r[1],
                'abstract': r[2],
                'similarity': r[3]
            }
            for r in results
        ]

    def close(self):
        """Close database connections."""
        self.cursor.close()
        self.conn.close()

# Advanced search patterns
def calculate_paper_relevance(conn):
    """Create function for calculating paper relevance"""
    sql = """
    CREATE OR REPLACE FUNCTION calculate_paper_relevance(
        similarity float,
        publication_date date,
        citation_count int
    ) RETURNS float AS $$
    BEGIN
        RETURN (
            0.6 * (1.0 - similarity) +  -- Vector similarity (60% weight)
            0.2 * (1.0 - EXTRACT(YEAR FROM age(current_date, publication_date))::float / 10.0) +  -- Recency (20% weight)
            0.2 * LEAST(citation_count::float / 1000.0, 1.0)  -- Citation impact (20% weight)
        );
    END;
    $$ LANGUAGE plpgsql;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()

# ================================================================================
# SECTION 9: ADVANCED ANALYTICS
# ================================================================================

import networkx as nx
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

def build_citation_network(citations: List[Dict]) -> nx.DiGraph:
    """Construct a directed graph from citation data"""
    G = nx.DiGraph()
    for citation in citations:
        G.add_edge(citation['citing_paper_id'], 
                  citation['cited_paper_id'])
    return G

def calculate_paper_influence(G: nx.DiGraph) -> Dict[str, float]:
    """Calculate PageRank scores for papers in the network"""
    return nx.pagerank(G, alpha=0.85)

def cluster_similar_images(embeddings: np.ndarray, 
                         eps: float = 0.5,
                         min_samples: int = 5) -> np.ndarray:
    """Cluster images based on embedding similarity"""
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    return clustering.fit_predict(embeddings)

def analyze_table_trends(table_data: List[Dict]) -> pd.DataFrame:
    """Analyze trends in numerical table data across papers"""
    df = pd.DataFrame(table_data)
    
    # Extract numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate statistics
    stats = df[numeric_cols].agg(['mean', 'std', 'min', 'max'])
    
    return stats

def detect_outliers(df: pd.DataFrame, 
                   threshold: float = 2.0) -> pd.DataFrame:
    """Detect unusual results in table data"""
    z_scores = pd.DataFrame()
    for column in df.select_dtypes(include=[np.number]):
        z_scores[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df[z_scores.abs().max(axis=1) > threshold]

def analyze_multimodal_patterns(paper_id: str,
                              conn: psycopg2.extensions.connection) -> Dict:
    """
    Analyze patterns across text, images, and formulas within a paper
    
    Args:
        paper_id: The arXiv ID of the paper to analyze
        conn: PostgreSQL database connection
        
    Returns:
        Dictionary containing multimodal analysis results
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    from collections import defaultdict
    
    # Initialize results container
    analysis_results = {
        'paper_id': paper_id,
        'cross_modal_similarities': {},
        'content_clusters': {},
        'key_concepts': [],
        'visual_themes': [],
        'mathematical_themes': []
    }
    
    try:
        # Extract section embeddings
        sections_query = """
            SELECT section_id, section_type, embedding, content_text
            FROM paper_sections
            WHERE arxiv_id = %s
        """
        sections_df = pd.read_sql(sections_query, conn, params=[paper_id])
        
        # Extract image embeddings
        images_query = """
            SELECT image_id, image_number, embedding, caption
            FROM paper_images
            WHERE arxiv_id = %s
        """
        images_df = pd.read_sql(images_query, conn, params=[paper_id])
        
        # Extract formula embeddings
        formulas_query = """
            SELECT formula_id, formula_number, embedding, latex_source
            FROM paper_formulas
            WHERE arxiv_id = %s
        """
        formulas_df = pd.read_sql(formulas_query, conn, params=[paper_id])
        
        # 1. Calculate cross-modal similarities
        if not sections_df.empty and not images_df.empty:
            # Convert embeddings from database format to numpy arrays
            section_embeddings = np.vstack(sections_df['embedding'].apply(np.array).values)
            image_embeddings = np.vstack(images_df['embedding'].apply(np.array).values)
            
            # Calculate similarity matrix between sections and images
            text_image_similarity = cosine_similarity(section_embeddings, image_embeddings)
            
            # Find highest similarity pairs
            text_image_pairs = []
            for i, section_id in enumerate(sections_df['section_id']):
                for j, image_id in enumerate(images_df['image_id']):
                    similarity = text_image_similarity[i, j]
                    if similarity > 0.6:  # Threshold for meaningful similarity
                        text_image_pairs.append({
                            'section_id': int(section_id),
                            'section_type': sections_df.iloc[i]['section_type'],
                            'image_id': int(image_id),
                            'image_number': images_df.iloc[j]['image_number'],
                            'similarity': float(similarity)
                        })
            
            analysis_results['cross_modal_similarities']['text_image'] = text_image_pairs
        
        # Similar analysis for text-formula relationships...
        # [Additional analysis code continues...]
        
        return analysis_results
        
    except Exception as e:
        # Log error and return partial results
        import logging
        logging.error(f"Error in multimodal analysis for paper {paper_id}: {str(e)}")
        analysis_results['error'] = str(e)
        return analysis_results

def visualize_citation_network(G: nx.DiGraph,
                             influence_scores: Dict[str, float]) -> None:
    """Create interactive visualization of citation network"""
    pos = nx.spring_layout(G)
    
    # Node sizes based on influence
    node_sizes = [influence_scores[node] * 1000 for node in G.nodes()]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos,
            node_size=node_sizes,
            with_labels=True,
            node_color='lightblue',
            font_size=8)
    plt.title("Citation Network Analysis")
    plt.show()

def generate_trend_report(conn: psycopg2.extensions.connection,
                         timeframe: str = '1 year') -> pd.DataFrame:
    """Generate comprehensive trend report across all modalities"""
    # Query for temporal trends
    trend_query = """
        SELECT DATE_TRUNC('month', publication_date) as month,
               COUNT(DISTINCT p.arxiv_id) as paper_count,
               COUNT(DISTINCT i.id) as image_count,
               COUNT(DISTINCT f.id) as formula_count
        FROM papers p
        LEFT JOIN paper_images i ON p.arxiv_id = i.arxiv_id
        LEFT JOIN paper_formulas f ON p.arxiv_id = f.arxiv_id
        WHERE publication_date >= CURRENT_DATE - INTERVAL %s
        GROUP BY DATE_TRUNC('month', publication_date)
        ORDER BY month;
    """
    
    # Execute query and return results
    return pd.read_sql(trend_query, conn, params=[timeframe])

# ================================================================================
# SECTION 10: RESEARCH ASSISTANT DASHBOARD
# ================================================================================

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional

app = FastAPI()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        dbname="research_db",
        user="user",
        password="password",
        host="localhost",
        port="5432"
    )

# HTML templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search-form/{search_type}", response_class=HTMLResponse)
async def get_search_form(request: Request, search_type: str):
    """Return the appropriate search form based on type"""
    if search_type == "semantic":
        return """
        <div class="search-options">
            <input type="text" 
                   name="semantic_query" 
                   placeholder="Enter text to find similar papers..."
                   class="search-box"
                   hx-get="/search/semantic"
                   hx-trigger="keyup changed delay:500ms"
                   hx-target="#results">
        </div>
        """
    elif search_type == "formula":
        return """
        <div class="search-options">
            <textarea name="formula" 
                      placeholder="Enter LaTeX formula..."
                      class="search-box"
                      hx-post="/search/formula"
                      hx-trigger="keyup changed delay:500ms"
                      hx-target="#results"></textarea>
        </div>
        """
    elif search_type == "author":
        return """
        <div class="search-options">
            <input type="text" 
                   name="author" 
                   placeholder="Search by author name..."
                   class="search-box"
                   hx-get="/search/author"
                   hx-trigger="keyup changed delay:500ms"
                   hx-target="#results">
        </div>
        """

@app.get("/search/semantic", response_class=HTMLResponse)
async def semantic_search(request: Request, semantic_query: str):
    """Perform semantic search using vector similarity"""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Simplified query - in reality, would use vector similarity
        cur.execute("""
            SELECT title, authors, abstract 
            FROM papers 
            WHERE to_tsvector('english', abstract) @@ plainto_tsquery('english', %s)
            LIMIT 10
        """, (semantic_query,))
        results = cur.fetchall()
        
        # Return results as HTML table
        html = "<table class='result-table'><tr><th>Title</th><th>Authors</th><th>Abstract</th></tr>"
        for r in results:
            html += f"<tr><td>{r['title']}</td><td>{r['authors']}</td><td>{r['abstract'][:200]}...</td></tr>"
        html += "</table>"
        return html
    finally:
        conn.close()

@app.post("/search/formula", response_class=HTMLResponse)
async def formula_search(request: Request, formula: str = Form(...)):
    """Search for papers with similar formulas"""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Simplified query - in reality, would use vector similarity for formulas
        cur.execute("""
            SELECT title, formula_text, similarity
            FROM paper_formulas
            WHERE formula_text LIKE %s
            LIMIT 5
        """, (f"%{formula}%",))
        results = cur.fetchall()
        
        html = "<table class='result-table'><tr><th>Title</th><th>Formula</th><th>Similarity</th></tr>"
        for r in results:
            html += f"<tr><td>{r['title']}</td><td>{r['formula_text']}</td><td>{r['similarity']:.2f}</td></tr>"
        html += "</table>"
        return html
    finally:
        conn.close()

@app.get("/search/author", response_class=HTMLResponse)
async def author_search(request: Request, author: str):
    """Search for papers by author"""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT authors, title, publication_date
            FROM papers
            WHERE authors ILIKE %s
            ORDER BY publication_date DESC
            LIMIT 10
        """, (f"%{author}%",))
        results = cur.fetchall()
        
        html = "<table class='result-table'><tr><th>Authors</th><th>Title</th><th>Date</th></tr>"
        for r in results:
            html += f"<tr><td>{r['authors']}</td><td>{r['title']}</td><td>{r['publication_date']}</td></tr>"
        html += "</table>"
        return html
    finally:
        conn.close()

async def search_papers(
    conn,
    text_query: Optional[str] = None,
    formula_query: Optional[str] = None,
    image_file: Optional[UploadFile] = None,
    filters: dict = None
):
    """
    Combined search function supporting multiple query types
    """
    results = []
    
    if text_query:
        # Text similarity search
        text_embedding = generate_text_embedding(text_query)
        text_results = search_by_text_embedding(conn, text_embedding)
        results.extend(text_results)
    
    if formula_query:
        # Formula similarity search
        formula_embedding = generate_formula_embedding(formula_query)
        formula_results = search_by_formula_embedding(conn, formula_embedding)
        results.extend(formula_results)
        
    if image_file:
        # Image similarity search
        image_embedding = generate_image_embedding(image_file)
        image_results = search_by_image_embedding(conn, image_embedding)
        results.extend(image_results)
    
    # Apply filters and ranking
    final_results = apply_filters_and_rank(results, filters)
    return final_results

# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    # Load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    db_config = {
        'dbname': config['Database']['dbname'],
        'user': config['Database']['user'],
        'password': config['Database']['password'],
        'host': config['Database']['host'],
        'port': config['Database']['port']
    }
    
    # Initialize and run pipeline
    pipeline = ArxivPipeline(db_config)
    
    # Example query for LLM papers
    query = 'cat:cs.CL AND (abs:"large language models" OR abs:LLM)'
    
    asyncio.run(pipeline.run_pipeline(query, max_results=100))

if __name__ == "__main__":
    import uvicorn
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Or run the pipeline
    # main()
