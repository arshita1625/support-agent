#!/usr/bin/env python3
"""Load support documents into the knowledge base."""
from typing import List
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.document import Document, DocumentChunk

class DocumentLoader:
    """Load and process support documents."""
    
    def __init__(self, documents_dir: str = "data/documents"):
        self.documents_dir = Path(documents_dir)
        self.processed_dir = Path("data/processed")
        
        # Make sure we're working with absolute paths
        if not self.documents_dir.is_absolute():
            self.documents_dir = project_root / self.documents_dir
        if not self.processed_dir.is_absolute():
            self.processed_dir = project_root / self.processed_dir
            
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        
    def load_document_from_file(self, file_path: Path) -> Document:
        """Load a document from a markdown file with enhanced metadata extraction."""
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine document type from filename or content
        filename = file_path.name.lower()
        
        if 'policy' in filename or 'policies' in filename:
            document_type = "policy"
        elif 'faq' in filename:
            document_type = "faq"
        elif 'procedure' in filename:
            document_type = "procedure"
        elif 'guide' in filename:
            document_type = "guide"
        else:
            document_type = "knowledge_article"
        
        # Extract title from filename or first heading
        title = file_path.stem.replace('_', ' ').title()
        if content.startswith('#'):
            first_line = content.split('\n')[0]
            title = first_line.strip('# ').strip()
        
        # Create safe relative path for metadata
        try:
            relative_path = file_path.relative_to(project_root)
        except ValueError:
            relative_path = file_path.name
        
        # ENHANCED METADATA EXTRACTION
        metadata = self._extract_comprehensive_metadata(content, file_path, document_type)
        
        # Add basic file metadata
        metadata.update({
            "source_file": str(relative_path),
            "file_size": len(content),
            "word_count": len(content.split()),
            "filename": file_path.name,
            "document_structure": self._analyze_document_structure(content)
        })
        
        # Create document
        document = Document(
            content=content,
            title=title,
            document_type=document_type,
            source=str(relative_path),
            metadata=metadata
        )
        
        return document

    def _extract_comprehensive_metadata(self, content: str, file_path: Path, document_type: str) -> dict:
        """Extract comprehensive metadata from document content."""
        
        content_lower = content.lower()
        metadata = {
            "tags": [],
            "topics": [],
            "actions": [],
            "escalation_triggers": [],
            "time_estimates": [],
            "contact_info": [],
            "priority": "medium",
            "complexity": "medium",
            "customer_impact": "medium",
            "departments": [],
            "systems_mentioned": [],
            "policies_referenced": [],
            "urls_mentioned": [],
            "email_addresses": [],
            "phone_numbers": []
        }
        
        # TOPIC AND TAG EXTRACTION
        topic_keywords = {
            # Domain-related topics
            "domain": ["domain", "website", "site", "url", "dns"],
            "suspension": ["suspend", "suspended", "suspension", "disable", "disabled"],
            "registration": ["register", "registration", "registrar", "registrant"],
            "renewal": ["renew", "renewal", "expire", "expiration", "expiry"],
            "transfer": ["transfer", "transferring", "move", "migration"],
            
            # Technical topics
            "whois": ["whois", "contact information", "registrant", "admin contact"],
            "dns": ["dns", "nameserver", "name server", "a record", "cname", "mx record"],
            "ssl": ["ssl", "certificate", "https", "security certificate", "tls"],
            "email": ["email", "mail", "smtp", "pop", "imap", "mailbox"],
            "hosting": ["hosting", "server", "hosting account", "web hosting"],
            
            # Business topics  
            "billing": ["billing", "payment", "invoice", "charge", "fee", "cost", "price"],
            "refund": ["refund", "refunds", "money back", "reimbursement"],
            "account": ["account", "login", "password", "username", "profile"],
            "support": ["support", "help", "assistance", "ticket", "customer service"],
            
            # Policy topics
            "terms": ["terms", "conditions", "agreement", "policy", "rules"],
            "privacy": ["privacy", "data protection", "personal information", "gdpr"],
            "abuse": ["abuse", "violation", "malware", "spam", "phishing", "fraud"],
            "compliance": ["compliance", "icann", "regulation", "requirement", "mandatory"],
            
            # Process topics
            "verification": ["verify", "verification", "confirm", "validate", "authenticate"],
            "documentation": ["document", "documentation", "proof", "evidence", "screenshot"],
            "escalation": ["escalate", "escalation", "manager", "supervisor", "senior"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata["topics"].append(topic)
                metadata["tags"].append(topic)
        
        # ACTION EXTRACTION
        action_patterns = {
            "update_whois": ["update whois", "change contact", "modify registrant"],
            "escalate_to_abuse_team": ["contact abuse", "abuse team", "policy violation"],
            "escalate_to_billing": ["billing issue", "payment problem", "refund request"],
            "escalate_to_technical": ["technical issue", "dns problem", "server issue"],
            "verify_identity": ["verify identity", "id verification", "identity proof"],
            "provide_documentation": ["provide documentation", "submit documents", "upload proof"],
            "contact_billing": ["billing department", "payment support", "invoice question"],
            "wait_for_processing": ["processing time", "review period", "business days"],
            "check_email": ["check email", "email confirmation", "email verification"],
            "update_payment": ["update payment", "change card", "payment method"]
        }
        
        for action, patterns in action_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                metadata["actions"].append(action)
        
        # TIME ESTIMATE EXTRACTION
        import re
        time_patterns = [
            r"(\d+)-(\d+)\s+(hours?|days?|weeks?)",
            r"(\d+)\s+(hours?|days?|weeks?|minutes?)",
            r"(same day|next day|immediately|instant)",
            r"(business days?|working days?)"
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                if isinstance(match, tuple):
                    metadata["time_estimates"].append(" ".join(match))
                else:
                    metadata["time_estimates"].append(match)
        
        # CONTACT INFORMATION EXTRACTION
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        metadata["email_addresses"] = emails
        
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, content)
        metadata["phone_numbers"] = [''.join(phone) for phone in phones]
        
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, content)
        metadata["urls_mentioned"] = urls
        
        # DEPARTMENT EXTRACTION
        departments = {
            "abuse_team": ["abuse team", "abuse department", "policy enforcement"],
            "billing": ["billing department", "accounting", "payment team"],
            "technical": ["technical team", "engineering", "tech support", "it department"],
            "customer_support": ["customer support", "customer service", "help desk"],
            "legal": ["legal team", "compliance", "legal department"],
            "management": ["management", "supervisor", "manager", "director"]
        }
        
        for dept, keywords in departments.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata["departments"].append(dept)
        
        # SYSTEM MENTIONS
        systems = {
            "control_panel": ["control panel", "dashboard", "account panel", "user interface"],
            "whois_database": ["whois database", "whois system", "registrant database"],
            "billing_system": ["billing system", "payment system", "invoice system"],
            "support_system": ["support system", "ticket system", "help desk system"],
            "dns_servers": ["dns server", "nameserver", "name server"],
            "email_system": ["email server", "mail server", "email system"]
        }
        
        for system, keywords in systems.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata["systems_mentioned"].append(system)
        
        # PRIORITY CALCULATION
        priority_indicators = {
            "urgent": ["urgent", "emergency", "critical", "immediate", "asap"],
            "high": ["suspend", "suspended", "down", "not working", "broken", "fraud", "security"],
            "medium": ["billing", "payment", "refund", "question", "help"],
            "low": ["information", "general", "how to", "tutorial", "guide"]
        }
        
        priority_scores = {"urgent": 4, "high": 3, "medium": 2, "low": 1}
        max_priority = "low"
        max_score = 0
        
        for priority, indicators in priority_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                if priority_scores[priority] > max_score:
                    max_score = priority_scores[priority]
                    max_priority = priority
        
        metadata["priority"] = max_priority
        
        # COMPLEXITY ASSESSMENT
        complexity_factors = {
            "high": len(metadata["departments"]) > 2 or len(metadata["actions"]) > 3,
            "medium": len(metadata["topics"]) > 5 or len(metadata["actions"]) > 1,
            "low": len(metadata["topics"]) <= 3 and len(metadata["actions"]) <= 1
        }
        
        if complexity_factors["high"]:
            metadata["complexity"] = "high"
        elif complexity_factors["medium"]:
            metadata["complexity"] = "medium"
        else:
            metadata["complexity"] = "low"
        
        # CUSTOMER IMPACT ASSESSMENT
        impact_keywords = {
            "high": ["suspend", "down", "not working", "broken", "lost", "cannot access"],
            "medium": ["delayed", "slow", "issue", "problem", "difficulty"],
            "low": ["question", "information", "clarification", "general inquiry"]
        }
        
        for impact, keywords in impact_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata["customer_impact"] = impact
                break
        
        # ESCALATION TRIGGERS
        escalation_keywords = [
            "policy violation", "fraud", "security breach", "malware", 
            "legal issue", "compliance violation", "urgent", "critical",
            "manager", "supervisor", "escalate", "complaint"
        ]
        
        for keyword in escalation_keywords:
            if keyword in content_lower:
                metadata["escalation_triggers"].append(keyword)
        
        # Remove duplicates and empty values
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = list(set(value))  # Remove duplicates
                metadata[key] = [v for v in metadata[key] if v]  # Remove empty strings
        
        return metadata

    def _analyze_document_structure(self, content: str) -> dict:
        """Analyze the structure of the document."""
        
        structure = {
            "has_headings": False,
            "heading_levels": [],
            "has_lists": False,
            "has_tables": False,
            "has_code_blocks": False,
            "has_links": False,
            "paragraph_count": 0,
            "section_count": 0
        }
        
        lines = content.split('\n')
        code_block_count = 0  # Track open/close code blocks
        
        for line in lines:
            line = line.strip()
            
            # Check for headings
            if line.startswith('#'):
                structure["has_headings"] = True
                level = len(line) - len(line.lstrip('#'))
                structure["heading_levels"].append(level)
                structure["section_count"] += 1
            
            # Check for lists
            elif line.startswith(('-', '*', '+')) or (line and line[0].isdigit() and '. ' in line):
                structure["has_lists"] = True
            
            # Check for tables
            elif '|' in line and line.count('|') >= 2:  # More robust table detection
                structure["has_tables"] = True
            
            # FIXED: Proper code block detection
            elif line.startswith('```'):
                code_block_count += 1
            
            # Check for inline code
            elif '`' in line and line.count('`') >= 2:
                structure["has_code_blocks"] = True
            
            # Check for links
            elif '[' in line and '](' in line:
                structure["has_links"] = True
            
            # Count paragraphs (non-empty lines that aren't special elements)
            elif line and not any(line.startswith(marker) for marker in ['#', '-', '*', '+', '>', '|']):
                structure["paragraph_count"] += 1
        
        # Check if we have paired code blocks (even number means properly closed)
        if code_block_count > 0 and code_block_count % 2 == 0:
            structure["has_code_blocks"] = True
        elif code_block_count % 2 == 1:
            # Odd number means unclosed code block, but still has code blocks
            structure["has_code_blocks"] = True
        
        # Remove duplicates from heading levels
        structure["heading_levels"] = list(set(structure["heading_levels"]))
        
        return structure


    def chunk_document(self, document: Document, chunk_size: int = 1000):
        """Split document into chunks for vector storage."""
        
        chunks = []
        content = document.content
        print(f"📄 Chunking document: {len(content)} characters")
        
        # Try to split by markdown sections first
        if '##' in content:
            print("🔍 Found markdown sections, splitting by headers...")
            
            # Find all section headers and their positions
            import re
            section_pattern = r'^##\s+(.+)$'
            sections = []
            
            # Split content by lines to preserve structure
            lines = content.split('\n')
            current_section = []
            current_title = "Introduction"
            start_pos = 0
            
            for i, line in enumerate(lines):
                if re.match(section_pattern, line.strip()):
                    # Save previous section if it has content
                    if current_section:
                        section_content = '\n'.join(current_section).strip()
                        if len(section_content) > 50:  # Skip tiny sections
                            sections.append({
                                'title': current_title,
                                'content': section_content,
                                'start_pos': start_pos,
                                'end_pos': start_pos + len(section_content)
                            })
                    
                    # Start new section
                    current_title = re.match(section_pattern, line.strip()).group(1)
                    current_section = [line]
                    start_pos = content.find(line)
                else:
                    current_section.append(line)
            
            # Don't forget the last section
            if current_section:
                section_content = '\n'.join(current_section).strip()
                if len(section_content) > 50:
                    sections.append({
                        'title': current_title,
                        'content': section_content,
                        'start_pos': start_pos,
                        'end_pos': start_pos + len(section_content)
                    })
            
            print(f"📝 Found {len(sections)} sections")
            
            # Create chunks from sections
            for i, section in enumerate(sections):
                section_content = section['content']
                
                if len(section_content) > chunk_size:
                    # Split large sections into smaller chunks
                    print(f"🔄 Section '{section['title']}' too large, splitting...")
                    sub_chunks = self._split_long_content(section_content, chunk_size)
                    
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk = DocumentChunk(
                            parent_document_id=document.id,
                            content=sub_chunk,
                            chunk_index=len(chunks),
                            start_char=section['start_pos'] + (j * chunk_size),  # Approximate
                            end_char=section['start_pos'] + (j * chunk_size) + len(sub_chunk),
                            document_type=document.document_type,
                            metadata={
                                **document.metadata,
                                "chunk_focus": self._extract_chunk_focus(sub_chunk),
                                "parent_title": document.title,
                                "section_title": section['title'],
                                "section_index": i,
                                "sub_chunk_index": j,
                                "is_sub_chunk": True
                            }
                        )
                        chunks.append(chunk)
                else:
                    # Section fits in one chunk
                    chunk = DocumentChunk(
                        parent_document_id=document.id,
                        content=section_content,
                        chunk_index=len(chunks),
                        start_char=section['start_pos'],
                        end_char=section['end_pos'],
                        document_type=document.document_type,
                        metadata={
                            **document.metadata,
                            "chunk_focus": self._extract_chunk_focus(section_content),
                            "parent_title": document.title,
                            "section_title": section['title'],
                            "section_index": i,
                            "is_sub_chunk": False
                        }
                    )
                    chunks.append(chunk)
        else:
            # No clear sections, use sliding window approach
            print("📝 No markdown sections found, using sliding window...")
            chunks = self._sliding_window_chunks(document, chunk_size)
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    def _split_long_content(self, content: str, chunk_size: int) -> List[str]:
        """Split long content into smaller chunks with overlap."""
        
        chunks = []
        words = content.split()
        current_chunk = []
        current_size = 0
        overlap_size = chunk_size // 4  # 25% overlap
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else current_chunk
                current_chunk = overlap_words + [word]
                current_size = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_size += word_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    
    def _sliding_window_chunks(self, document: Document, chunk_size: int):
        """Create chunks using sliding window approach."""
        chunks = []
        content = document.content
        overlap = 200
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Try to break at sentence boundaries
            if end < len(content):
                sentence_end = content.rfind('.', start, end)
                if sentence_end > start + chunk_size - 200:
                    end = sentence_end + 1
            
            chunk_content = content[start:end].strip()
            
            if len(chunk_content) < 50:
                break
            
            chunk = DocumentChunk(
                parent_document_id=document.id,
                content=chunk_content,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
                document_type=document.document_type,
                metadata={
                    **document.metadata,
                    "chunk_focus": self._extract_chunk_focus(chunk_content),
                    "parent_title": document.title
                }
            )
            
            chunks.append(chunk)
            start = end - overlap
            chunk_index += 1
            
            if chunk_index > 50:  # Safety check
                break
        
        return chunks
    
    def _extract_chunk_focus(self, content: str) -> str:
        """Extract the main focus/topic of a chunk with enhanced detection."""
        content_lower = content.lower()
        
        # More specific focus mapping with priority
        focus_mapping = {
            # High priority (specific issues)
            "domain_suspension": ["domain suspend", "site suspend", "website suspend"],
            "account_suspension": ["account suspend", "user suspend", "login suspend"],
            "billing_dispute": ["billing dispute", "charge dispute", "payment dispute"],
            "fraud_security": ["fraud", "security breach", "malware", "phishing", "hack"],
            
            # Medium priority (common topics)
            "whois_update": ["whois", "contact information", "registrant", "update contact"],
            "domain_transfer": ["domain transfer", "transfer domain", "move domain"],
            "dns_management": ["dns", "nameserver", "name server", "dns record"],
            "email_issues": ["email", "mail", "smtp", "pop", "imap"],
            "ssl_certificate": ["ssl", "certificate", "https", "security certificate"],
            
            # Process-focused
            "reactivation": ["reactivate", "reactivation", "restore", "reinstate"],
            "verification": ["verify", "verification", "confirm", "validate"],
            "payment_processing": ["payment", "billing", "invoice", "charge"],
            "refund_processing": ["refund", "refunds", "money back", "reimbursement"],
            
            # Support processes
            "escalation": ["escalate", "escalation", "manager", "supervisor"],
            "documentation": ["document", "documentation", "proof", "evidence"],
            "technical_support": ["technical", "tech support", "server", "hosting"],
            
            # General categories
            "policy": ["policy", "terms", "conditions", "agreement", "rules"],
            "faq": ["question", "answer", "q:", "a:", "frequently asked"],
            "procedure": ["step", "process", "procedure", "how to", "instructions"],
            "general": []  # Default fallback
        }
        
        # Check in priority order (most specific first)
        for focus, keywords in focus_mapping.items():
            if focus == "general":
                continue
            if any(keyword in content_lower for keyword in keywords):
                return focus
        
        return "general"

    
    def load_all_documents(self):
        """Load all documents from the documents directory."""
        
        documents = []
        all_chunks = []
        
        print(f"🔍 Scanning directory: {self.documents_dir}")
        
        # Check if directory exists
        if not self.documents_dir.exists():
            print(f"❌ Directory {self.documents_dir} does not exist!")
            return documents, all_chunks
        
        # Find all markdown files (both .md and .markdown)
        markdown_files = list(self.documents_dir.rglob("*.md")) + list(self.documents_dir.rglob("*.markdown"))
        
        if not markdown_files:
            print(f"❌ No markdown files found in {self.documents_dir}")
            print("Available files:")
            for item in self.documents_dir.iterdir():
                print(f"   {item.name}")
            return documents, all_chunks
        
        print(f"📄 Found {len(markdown_files)} markdown files")
        
        for file_path in markdown_files:
            # FIX: Use just the filename for display instead of relative_to
            print(f"\n📄 Loading: {file_path.name}")
            
            try:
                # Load document
                document = self.load_document_from_file(file_path)
                documents.append(document)
                
                # Create chunks
                chunks = self.chunk_document(document)
                all_chunks.extend(chunks)
                
                print(f"   ✅ Created document: {document.title}")
                print(f"   📦 Generated {len(chunks)} chunks")
                print(f"   🏷️  Type: {document.document_type}")
                print(f"   📊 Tags: {document.metadata.get('tags', [])}")
                
            except Exception as e:
                print(f"   ❌ Error loading {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return documents, all_chunks
    
    def save_processed_documents(self, documents, chunks):
        """Save processed documents and chunks as JSON."""
        
        # Convert documents to dictionaries
        documents_data = [doc.to_dict() for doc in documents]
        with open(self.processed_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, indent=2, default=str)
        
        # Convert chunks to dictionaries  
        chunks_data = [chunk.to_dict() for chunk in chunks]
        with open(self.processed_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, default=str)
        
        print(f"\n💾 Saved {len(documents)} documents to documents.json")
        print(f"💾 Saved {len(chunks)} chunks to chunks.json")
    
    def print_summary(self, documents, chunks):
        """Print loading summary."""
        
        print(f"\n📋 DOCUMENT LOADING SUMMARY")
        print("=" * 50)
        print(f"Total Documents: {len(documents)}")
        print(f"Total Chunks: {len(chunks)}")
        
        if documents:
            print(f"\n📄 Documents Found:")
            for doc in documents:
                print(f"   • {doc.title} ({doc.document_type})")
                print(f"     File: {doc.metadata['filename']}")
                print(f"     Size: {doc.metadata['file_size']} chars")
                print(f"     Tags: {doc.metadata.get('tags', [])}")
        
        if chunks:
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            
            print(f"\n📦 Chunk Statistics:")
            print(f"   Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"   Min chunk size: {min(chunk_sizes)}")
            print(f"   Max chunk size: {max(chunk_sizes)}")

def main():
    """Main function to load documents."""
    
    print("🚀 Loading Support Documents...\n")
    
    # Initialize loader
    loader = DocumentLoader()
    
    # Load all documents
    documents, chunks = loader.load_all_documents()
    
    if not documents:
        print("❌ No documents found!")
        print("Make sure you have .md files in data/documents/")
        return False
    
    # Save processed documents
    loader.save_processed_documents(documents, chunks)
    
    # Print summary
    loader.print_summary(documents, chunks)
    
    print("\n🎉 Document loading completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
