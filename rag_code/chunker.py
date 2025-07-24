import re
from typing import List, Dict, Optional, Tuple
import warnings

class BengaliChunker:
    """
    Enhanced chunker optimized for Bengali and mixed-language text with:
    - Improved script mixing detection
    - Word-boundary aware overlap
    - Bengali discourse markers
    - Better clause detection
    - Semantic unit preservation
    """

    def __init__(
        self,
        max_chunk_size: int = 200,
        overlap: int = 50,
        min_chunk_size: int = 20,
        separators: Optional[List[str]] = None,
        discourse_markers: Optional[List[str]] = None
    ):
        """
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap: Target overlap size between chunks (in chars)
            min_chunk_size: Minimum chunk size to keep
            separators: Custom separators (default: Bengali + English punctuation)
            discourse_markers: Bengali discourse markers for better splitting
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
        # Enhanced separators for Bengali + English
        self.sentence_separators = separators or [
            '।', '॥',  # Bengali danda and double danda
            '.', '?', '!', '…'  # English punctuation
        ]
        
        self.clause_separators = [',', ';', '|', ':', '-', '–', '—']
        
        # Bengali discourse markers for better semantic splitting
        self.discourse_markers = discourse_markers or [
            'কিন্তু', 'তবে', 'সেজন্য', 'তাই', 'অতএব', 'কারণ', 
            'যদিও', 'তথাপি', 'তবুও', 'অথচ', 'বরং', 'আর',
            'এবং', 'ও', 'অথবা', 'কিংবা', 'নাকি'
        ]

    def detect_script_composition(self, text: str) -> Dict[str, float]:
        """Analyze script composition of text"""
        if not text.strip():
            return {"bengali": 0.0, "english": 0.0, "other": 0.0}
        
        # Remove whitespace for accurate counting
        clean_text = re.sub(r'\s+', '', text)
        total_chars = len(clean_text)
        
        if total_chars == 0:
            return {"bengali": 0.0, "english": 0.0, "other": 0.0}
        
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', clean_text))
        english_chars = len(re.findall(r'[a-zA-Z]', clean_text))
        other_chars = total_chars - bengali_chars - english_chars
        
        return {
            "bengali": bengali_chars / total_chars,
            "english": english_chars / total_chars,
            "other": other_chars / total_chars
        }

    def classify_language(self, text: str) -> str:
        """Classify text language based on script composition"""
        composition = self.detect_script_composition(text)
        
        if composition["bengali"] > 0.6:
            return "bn"
        elif composition["english"] > 0.6:
            return "en"
        elif composition["bengali"] > 0.3 or composition["english"] > 0.3:
            return "mixed"
        else:
            return "other"

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using language-appropriate separators"""
        if not text.strip():
            return []

        # Create regex pattern for sentence boundaries
        separators_pattern = '|'.join(re.escape(sep) for sep in self.sentence_separators)
        pattern = f'(?<=[{separators_pattern}])\\s+'
        
        sentences = re.split(pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def split_by_discourse_markers(self, text: str) -> List[str]:
        """Split text using Bengali discourse markers"""
        if not text.strip():
            return [text]
        
        # Create pattern for discourse markers (word boundaries)
        markers_pattern = '|'.join(re.escape(marker) for marker in self.discourse_markers)
        pattern = f'\\b({markers_pattern})\\b'
        
        # Split but keep the markers
        parts = re.split(f'({pattern})', text)
        
        # Reconstruct with markers attached to following text
        result = []
        current = ""
        
        for part in parts:
            if part.strip():
                if any(marker in part for marker in self.discourse_markers):
                    if current:
                        result.append(current.strip())
                        current = part
                    else:
                        current = part
                else:
                    current += part
        
        if current:
            result.append(current.strip())
        
        return [r for r in result if r.strip()]

    def split_clauses(self, text: str) -> List[str]:
        """Enhanced clause splitting with discourse markers"""
        # First try discourse markers
        discourse_splits = self.split_by_discourse_markers(text)
        
        if len(discourse_splits) > 1:
            return discourse_splits
        
        # Fallback to punctuation-based splitting
        pattern = '|'.join(re.escape(sep) for sep in self.clause_separators)
        pattern = f'(?<=[{pattern}])\\s+'
        
        clauses = re.split(pattern, text)
        return [c.strip() for c in clauses if c.strip()]

    def find_word_boundary_split(self, text: str, target_pos: int) -> int:
        """Find the nearest word boundary before target_pos"""
        if target_pos >= len(text):
            return len(text)
        
        # Look backwards from target_pos to find word boundary
        for i in range(target_pos, max(0, target_pos - 50), -1):
            if i == 0 or text[i-1].isspace():
                return i
        
        # If no word boundary found, return target_pos
        return target_pos

    def create_word_boundary_overlap(self, chunks_so_far: List[str], target_overlap: int) -> List[str]:
        """Create overlap that respects word boundaries"""
        if not chunks_so_far:
            return []
        
        overlap_text = ""
        overlap_chunks = []
        
        # Add chunks from the end until we reach target overlap
        for chunk in reversed(chunks_so_far):
            if len(overlap_text) + len(chunk) > target_overlap:
                # Find word boundary within this chunk
                remaining_space = target_overlap - len(overlap_text)
                if remaining_space > 20:  # Only if significant space remains
                    split_pos = self.find_word_boundary_split(chunk, remaining_space)
                    partial_chunk = chunk[:split_pos].strip()
                    if partial_chunk:
                        overlap_chunks.insert(0, partial_chunk)
                break
            
            overlap_chunks.insert(0, chunk)
            overlap_text += chunk + " "
        
        return overlap_chunks

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Enhanced chunking pipeline with improved overlap and language detection
        """
        if not text.strip():
            return []

        # Detect overall language composition
        overall_composition = self.detect_script_composition(text)
        
        # Split into sentences
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sent_length = len(sentence)
            
            # Case 1: Sentence fits in current chunk
            if current_length + sent_length + 1 <= self.max_chunk_size:  # +1 for space
                current_chunk.append(sentence)
                current_length += sent_length + (1 if current_chunk else 0)
                continue
            
            # Case 2: Current chunk needs to be finalized
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                overlap_chunks = self.create_word_boundary_overlap(current_chunk, self.overlap)
                
                chunks.append(self._create_chunk_dict(
                    chunk_text, 
                    overall_composition,
                    len(" ".join(overlap_chunks))
                ))
                
                # Start new chunk with overlap
                current_chunk = overlap_chunks.copy()
                current_length = sum(len(chunk) + 1 for chunk in current_chunk)
                current_length = max(0, current_length - 1)  # Remove last space
            
            # Case 3: Handle long sentences
            if sent_length > self.max_chunk_size:
                clauses = self.split_clauses(sentence)
                
                for clause in clauses:
                    clause = clause.strip()
                    if not clause:
                        continue
                        
                    clause_length = len(clause)
                    
                    # If clause still too long, force split at word boundary
                    if clause_length > self.max_chunk_size:
                        words = clause.split()
                        temp_clause = ""
                        
                        for word in words:
                            if len(temp_clause) + len(word) + 1 > self.max_chunk_size:
                                if temp_clause:
                                    current_chunk.append(temp_clause.strip())
                                    current_length += len(temp_clause) + 1
                                    
                                    if current_length >= self.max_chunk_size:
                                        chunk_text = " ".join(current_chunk)
                                        overlap_chunks = self.create_word_boundary_overlap(
                                            current_chunk, self.overlap
                                        )
                                        
                                        chunks.append(self._create_chunk_dict(
                                            chunk_text,
                                            overall_composition,
                                            len(" ".join(overlap_chunks))
                                        ))
                                        
                                        current_chunk = overlap_chunks.copy()
                                        current_length = sum(len(c) + 1 for c in current_chunk)
                                        current_length = max(0, current_length - 1)
                                
                                temp_clause = word
                            else:
                                temp_clause += (" " + word) if temp_clause else word
                        
                        if temp_clause:
                            current_chunk.append(temp_clause.strip())
                            current_length += len(temp_clause) + 1
                    else:
                        # Normal clause processing
                        if current_length + clause_length + 1 > self.max_chunk_size:
                            if current_chunk:
                                chunk_text = " ".join(current_chunk)
                                overlap_chunks = self.create_word_boundary_overlap(
                                    current_chunk, self.overlap
                                )
                                
                                chunks.append(self._create_chunk_dict(
                                    chunk_text,
                                    overall_composition,
                                    len(" ".join(overlap_chunks))
                                ))
                                
                                current_chunk = overlap_chunks.copy()
                                current_length = sum(len(c) + 1 for c in current_chunk)
                                current_length = max(0, current_length - 1)
                        
                        current_chunk.append(clause)
                        current_length += clause_length + 1
            else:
                # Normal sentence
                current_chunk.append(sentence)
                current_length += sent_length + 1

        # Handle final chunk
        if current_chunk and current_length >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk_dict(
                chunk_text,
                overall_composition,
                0  # No overlap for final chunk
            ))

        return chunks

    def _create_chunk_dict(self, text: str, overall_composition: Dict[str, float], overlap_length: int) -> Dict:
        """Create a standardized chunk dictionary with metadata"""
        chunk_composition = self.detect_script_composition(text)
        language = self.classify_language(text)
        
        return {
            "text": text,
            "metadata": {
                "language": language,
                "length_chars": len(text),
                "bengali_ratio": chunk_composition["bengali"],
                "english_ratio": chunk_composition["english"],
                "has_sentence_end": any(
                    text.rstrip().endswith(sep) for sep in self.sentence_separators
                ),
                "overlap_with_next": overlap_length,
                "word_count": len(text.split()),
                "has_discourse_markers": any(
                    marker in text for marker in self.discourse_markers
                )
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Sample Bengali text with mixed content
    sample_text = """
    মামা বিবাহ-বাড়িতে ঢুকিয়া খুশি হইলেন না। একে তো উঠানটাতে বরযাত্রীদের জায়গা সংকুলান হওয়াই শক্ত,
    তাহার পরে সমস্ত আয়োজন নিতান্ত মধ্যম রকমের। ইহার পরে শঙ্কুনাথবাবুর ব্যবহারটাও নেহাত ঠান্ডা। 
    কিন্তু তার বিনয়টা অজাত্র নয়। This is an English sentence mixed in. মুখে তো কথাই নাই কোমরে চাদর বাঁধা, 
    গলা ভাঙা, টাক-পড়া, মিশ-কালো এবং বিপুল-কণ্ঠের একটা লোক AI technology ব্যবহার করে কাজ করছে।
    তবে আমি সভায় বসিবার কিছুক্ষণ পরেই মামা শঙ্কুনাথবাবুকে পাশের ঘরে ডাকিয়া লইয়া গেলেন।
    """

    # Initialize enhanced chunker
    chunker = BengaliChunker(
        max_chunk_size=300,
        overlap=80,
        min_chunk_size=50
    )

    # Process text
    chunks = chunker.chunk_text(sample_text)

    # Display results
    print(f"Generated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"Text: {chunk['text'][:100]}{'...' if len(chunk['text']) > 100 else ''}")
        print(f"Language: {chunk['metadata']['language']}")
        print(f"Bengali ratio: {chunk['metadata']['bengali_ratio']:.2f}")
        print(f"English ratio: {chunk['metadata']['english_ratio']:.2f}")
        print(f"Length: {chunk['metadata']['length_chars']} chars")
        print(f"Has discourse markers: {chunk['metadata']['has_discourse_markers']}")
        print(f"Overlap with next: {chunk['metadata']['overlap_with_next']}")
        print("-" * 50)