import hashlib
import json
import os
import tiktoken
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# ==============================================================================
# Constants & Constraints
# ==============================================================================
# Strict architectural constraint: max 16,000 tokens per LLM call.
# We reserve 2,000 tokens for system prompts, few-shot examples, and output.
MAX_CHUNK_TOKENS = 14000
# For batching smaller chunks to minimize total API requests
BATCH_THRESHOLD_TOKENS = 8000

# ==============================================================================
# Data Models
# ==============================================================================
class TextChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the chunk (hash of content)")
    text: str = Field(..., description="The textual content of the chunk")
    token_count: int = Field(..., description="Number of tokens in the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata like paper ID, section, etc.")

class ChunkBatch(BaseModel):
    batch_id: str = Field(..., description="Unique identifier for the batch")
    chunks: List[TextChunk] = Field(default_factory=list, description="List of chunks in this batch")
    total_tokens: int = Field(0, description="Total tokens in the batch")

# ==============================================================================
# Utilities
# ==============================================================================
def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Counts the number of tokens in a string using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback approximation if tiktoken fails (e.g. ~4 chars per token)
        return len(text) // 4

def generate_hash(text: str) -> str:
    """Generates an MD5 hash of the text for caching and ID purposes."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# ==============================================================================
# Chunker Implementation
# ==============================================================================
class PaperChunker:
    """
    Handles chunking of full paper texts to enforce the 16k token limit,
    with logic to batch smaller chunks and optimize LLM API calls.
    """
    
    def __init__(self, max_tokens: int = MAX_CHUNK_TOKENS, batch_threshold: int = BATCH_THRESHOLD_TOKENS):
        self.max_tokens = max_tokens
        self.batch_threshold = batch_threshold
        # We use a standard character text splitter approach but driven by token counts
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=500, # overlap in tokens (approx)
            length_function=count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_text(self, text: str, paper_id: str = "unknown") -> List[TextChunk]:
        """Splits text into chunks strictly within the token limit."""
        raw_chunks = self.splitter.split_text(text)
        
        processed_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            tokens = count_tokens(chunk_text)
            chunk_hash = generate_hash(chunk_text)
            
            processed_chunks.append(TextChunk(
                chunk_id=f"{paper_id}_{chunk_hash}",
                text=chunk_text,
                token_count=tokens,
                metadata={"paper_id": paper_id, "index": i}
            ))
            
        return processed_chunks

    def batch_chunks(self, chunks: List[TextChunk]) -> List[ChunkBatch]:
        """
        Batches smaller chunks together into a single prompt if they are < 8k tokens combined.
        This minimizes total API calls while respecting the 16k limit.
        """
        batches: List[ChunkBatch] = []
        current_batch = ChunkBatch(batch_id=generate_hash("batch_0"))
        
        for chunk in chunks:
            # If adding this chunk exceeds max tokens, finalize current batch and start new
            if current_batch.total_tokens + chunk.token_count > self.max_tokens:
                if current_batch.chunks:
                    batches.append(current_batch)
                current_batch = ChunkBatch(
                    batch_id=generate_hash(chunk.chunk_id),
                    chunks=[chunk],
                    total_tokens=chunk.token_count
                )
            else:
                current_batch.chunks.append(chunk)
                current_batch.total_tokens += chunk.token_count
                
                # If current batch crosses the optimization threshold, we can close it
                # or keep adding up to max_tokens. For strict 8k limit batching logic:
                if current_batch.total_tokens >= self.batch_threshold:
                    batches.append(current_batch)
                    current_batch = ChunkBatch(batch_id=generate_hash(chunk.chunk_id + "next"))
                    
        # Append any remaining chunks
        if current_batch.chunks:
            batches.append(current_batch)
            
        # Fix batch IDs to reflect their contents
        for i, batch in enumerate(batches):
            content_hash = generate_hash("".join([c.chunk_id for c in batch.chunks]))
            batch.batch_id = f"batch_{i}_{content_hash}"
            
        return batches

# ==============================================================================
# Caching Implementation
# ==============================================================================
class LLMCacheManager:
    """
    Implements caching for LLM responses mapped to specific text chunk hashes
    to minimize redundant API requests.
    """
    
    def __init__(self, cache_dir: str = ".cache/llm_responses"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
    def _get_cache_path(self, chunk_id: str, agent_role: str) -> str:
        return os.path.join(self.cache_dir, f"{agent_role}_{chunk_id}.json")

    def get_cached_response(self, chunk_id: str, agent_role: str) -> Optional[Dict[str, Any]]:
        """Retrieves a cached response if it exists."""
        cache_path = self._get_cache_path(chunk_id, agent_role)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_response(self, chunk_id: str, agent_role: str, response: Dict[str, Any]) -> None:
        """Saves a response to the cache."""
        cache_path = self._get_cache_path(chunk_id, agent_role)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to cache response for {chunk_id}: {e}")
