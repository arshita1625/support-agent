#!/usr/bin/env python3

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import sys
from pathlib import Path

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
    processing_triggered: bool

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

class DeleteResponse(BaseModel):
    message: str
    filename: str

documents_dir = project_root / "data" / "documents"

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
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
    
    processing_triggered = False
    if process_immediately:
        try:
            success = load_documents_main()
            processing_triggered = success
            
            if success:
                import os
                import signal
                print("Triggering application restart to pick up new documents...")
                os.kill(os.getpid(), signal.SIGUSR1) 
            else:
                print("Document processing completed with warnings")
                
        except Exception as e:
            print(f"Error during document processing: {e}")
            processing_triggered = False
    
    return UploadResponse(
        message="File uploaded successfully",
        filename=file.filename,
        file_path=str(file_path.relative_to(project_root)),
        file_size=len(content),
        processing_triggered=processing_triggered
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

# @router.post("/process", response_model=ProcessResponse)
# async def process_documents():
#     try:
#         print("Starting document processing...")
#         success = load_documents_main()
        
#         if success:
#             return ProcessResponse(
#                 message="Document processing completed successfully",
#                 success=True
#             )
#         else:
#             return JSONResponse(
#                 status_code=207,
#                 content={
#                     "message": "Document processing completed with warnings",
#                     "success": False
#                 }
#             )
            
#     except Exception as e:
#         print(f"Processing error: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Document processing failed: {str(e)}"
#         )

@router.delete("/files/{filename}", response_model=DeleteResponse)
async def delete_file(filename: str):
    try:
        file_path = documents_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_path.suffix.lower() not in ['.md', '.markdown']:
            raise HTTPException(status_code=400, detail="Can only delete markdown files")
        
        file_path.unlink()
        
        return DeleteResponse(
            message=f"File '{filename}' deleted successfully",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
