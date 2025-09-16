#!/usr/bin/env python3

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import sys
from pathlib import Path
import asyncio
import threading
import uuid
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.load_documents import main as load_documents_main

class UploadRequest(BaseModel):
    process_immediately: bool = Field(True, description="Process the file immediately after upload")

class UploadResponse(BaseModel):
    message: str
    filename: str
    file_path: str
    file_size: int
    processing_job_id: Optional[str] = None
    processing_status: str

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    filename: str
    progress: str
    chunks_created: Optional[int] = None
    error_message: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None

class FileInfo(BaseModel):
    filename: str
    size: int
    modified: float

class FileListResponse(BaseModel):
    message: str
    total_files: int
    files: List[FileInfo]

class ProcessResponse(BaseModel):
    message: str
    success: bool
    vector_db_updated: bool

class DeleteResponse(BaseModel):
    message: str
    filename: str

# Global dictionary to track processing jobs
processing_jobs = {}

documents_dir = project_root / "data" / "documents"

router = APIRouter()

async def trigger_complete_rag_pipeline():
    try:
        print("Starting complete RAG pipeline...")
        
        # Step 1: Process documents to JSON
        success = load_documents_main()
        if not success:
            print("Document processing failed")
            return False, 0
        
        print("Document processing completed, now updating RAG system...")
        
        # Step 2: Update RAG service and vector database
        try:
            # Import RAG service
            from src.services.rag_service import RAGService
            
            import main
            if not hasattr(main, 'rag_service') or main.rag_service is None:
                print("Initializing RAG service...")
                main.rag_service = RAGService()
            
            # Trigger document update which includes vector DB update
            await main.rag_service.ensure_documents_current_on_startup()
            print("RAG service updated successfully")
            
            import json
            chunks_file = project_root / "data" / "processed" / "chunks.json"
            chunks_count = 0
            if chunks_file.exists():
                with open(chunks_file, 'r') as f:
                    chunks_data = json.load(f)
                    chunks_count = len(chunks_data)
            
            return True, chunks_count
            
        except Exception as rag_error:
            print(f"RAG service update failed: {rag_error}")
            
            # Fallback: Try direct vector store update
            try:
                from src.services.vector_store import VectorStoreService
                from src.services.embedding_service import EmbeddingService
                
                vector_store = VectorStoreService()
                embedding_service = EmbeddingService()
                
                # Initialize services
                await vector_store.initialize()
                await embedding_service.initialize()
                
                # Load processed chunks
                import json
                chunks_file = project_root / "data" / "processed" / "chunks.json"
                if not chunks_file.exists():
                    print("No chunks file found")
                    return False, 0
                
                with open(chunks_file, 'r') as f:
                    chunks_data = json.load(f)
                
                print(f"Processing {len(chunks_data)} chunks...")
                
                # Process chunks in batches
                batch_size = 10
                for i in range(0, len(chunks_data), batch_size):
                    batch = chunks_data[i:i + batch_size]
                    
                    # Extract content for embedding
                    contents = [chunk['content'] for chunk in batch]
                    
                    # Generate embeddings
                    embeddings = await embedding_service.generate_embeddings(contents)
                    
                    # Prepare points for vector store
                    points = []
                    for chunk, embedding in zip(batch, embeddings):
                        point = {
                            "id": chunk['chunk_id'],
                            "vector": embedding,
                            "payload": {
                                "content": chunk['content'],
                                "document_id": chunk['parent_document_id'],
                                "document_type": chunk['document_type'],
                                "chunk_index": chunk['chunk_index'],
                                "metadata": chunk.get('metadata', {})
                            }
                        }
                        points.append(point)
                    
                    # Add to vector store
                    await vector_store.add_points(points)
                    print(f"Added batch {i//batch_size + 1}/{(len(chunks_data)-1)//batch_size + 1}")
                
                print(f"Successfully added {len(chunks_data)} chunks to vector database")
                return True, len(chunks_data)
                
            except Exception as vector_error:
                print(f"Direct vector store update failed: {vector_error}")
                return False, 0
    
    except Exception as e:
        print(f"Complete pipeline failed: {e}")
        return False, 0

def background_processing_worker(job_id: str, filename: str):
    try:
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = "Starting document processing..."
        
        print(f"Background processing started for job. IT TAKES ~5 SECONDS TO COMPLETE")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the complete RAG pipeline
            processing_jobs[job_id]["progress"] = "Processing documents and generating embeddings..."
            success, chunks_created = loop.run_until_complete(trigger_complete_rag_pipeline())
            
            if success:
                processing_jobs[job_id]["status"] = "completed"
                processing_jobs[job_id]["progress"] = f"Processing completed successfully - {chunks_created} chunks created"
                processing_jobs[job_id]["chunks_created"] = chunks_created
                processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                print(f"Background processing completed for job")
            else:
                processing_jobs[job_id]["status"] = "failed"
                processing_jobs[job_id]["progress"] = "Processing completed with errors"
                processing_jobs[job_id]["error_message"] = "RAG pipeline failed"
                processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        finally:
            loop.close()
    
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["progress"] = "Processing failed"
        processing_jobs[job_id]["error_message"] = str(e)
        processing_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        print(f"Background processing failed for job {job_id}: {e}")

@router.post("/", response_model=UploadResponse)
async def upload_markdown_file(
    file: UploadFile = File(..., description="Markdown file to upload"),
    process_immediately: bool = Form(True, description="Process the file immediately after upload")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.md', '.markdown']:
        raise HTTPException(
            status_code=400,
            detail=f"Only markdown files are supported (.md, .markdown). Got: {file_ext}"
        )
    
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        try:
            text_content = content.decode('utf-8')
            if len(text_content.strip()) < 10:
                raise HTTPException(status_code=400, detail="File content is too short")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    try:
        file_path = documents_dir / file.filename
        
        if file_path.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            counter = 1
            
            while file_path.exists():
                new_name = f"{stem}_{counter}{suffix}"
                file_path = documents_dir / new_name
                counter += 1
            
            print(f"File renamed to avoid conflict: {file_path.name}")
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        print(f"File saved: {file_path}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Prepare response
    job_id = None
    processing_status = "file_saved"
    
    if process_immediately:
        job_id = str(uuid.uuid4())
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": file.filename,
            "progress": "File uploaded, queuing for processing...",
            "chunks_created": None,
            "error_message": None,
            "started_at": datetime.now().isoformat(),
            "completed_at": None
        }
        thread = threading.Thread(
            target=background_processing_worker,
            args=(job_id, file.filename),
            daemon=True
        )
        thread.start()
        
        processing_status = "processing_queued"
        print(f"Started background processing job {job_id} for file {file.filename}")
    return UploadResponse(
        message="File uploaded successfully" + (" - processing started in background" if process_immediately else ""),
        filename=file.filename,
        file_path=str(file_path.relative_to(project_root)),
        file_size=len(content),
        processing_job_id=job_id,
        processing_status=processing_status
    )

@router.get("/files", response_model=FileListResponse)
async def list_uploaded_files():
    try:
        markdown_files = []
        
        for file_path in documents_dir.glob("*.md"):
            file_stat = file_path.stat()
            markdown_files.append(FileInfo(
                filename=file_path.name,
                size=file_stat.st_size,
                modified=file_stat.st_mtime
            ))
        
        for file_path in documents_dir.glob("*.markdown"):
            file_stat = file_path.stat()
            markdown_files.append(FileInfo(
                filename=file_path.name,
                size=file_stat.st_size,
                modified=file_stat.st_mtime
            ))
        
        markdown_files.sort(key=lambda x: x.modified, reverse=True)
        
        return FileListResponse(
            message="Files retrieved successfully",
            total_files=len(markdown_files),
            files=markdown_files
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@router.delete("/files/{filename}", response_model=DeleteResponse)
async def delete_file(filename: str):
    try:
        file_path = documents_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_path.suffix.lower() not in ['.md', '.markdown']:
            raise HTTPException(status_code=400, detail="Can only delete markdown files")
        
        file_path.unlink()
        
        # Start background processing to update vector DB after deletion
        job_id = str(uuid.uuid4())
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": f"DELETION: {filename}",
            "progress": "File deleted, updating vector database...",
            "chunks_created": None,
            "error_message": None,
            "started_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        thread = threading.Thread(
            target=background_processing_worker,
            args=(job_id, f"DELETION: {filename}"),
            daemon=True
        )
        thread.start()
        
        return DeleteResponse(
            message=f"File '{filename}' deleted successfully - vector database update started in background",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
