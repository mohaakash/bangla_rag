import re
from typing import List, Dict, Optional
import warnings

class BengaliChunker:
    """
    A robust chunker optimized for Bengali and mixed-language text with:
    - Sentence-aware splitting (Danda, punctuation)
    - Clause-level fallback for long sentences
    - Dynamic overlap for context preservation
    - Language detection
    - Metadata tracking
    """

    def __init__(
        self,
        max_chunk_size: int = 500,
        overlap: int = 100,
        min_chunk_size: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap: Overlap size between chunks (in chars)
            min_chunk_size: Minimum chunk size to keep
            separators: Custom separators (default: Bengali + English punctuation)
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
        # Default separators for Bengali + English
        self.sentence_separators = separators or [
            '।', '?', '!',  # Bengali
            '.', '?', '!'    # English
        ]
        
        self.clause_separators = [',', ';', '|', ':', '-', '–']

    def is_bengali_dominant(self, text: str) -> bool:
        """Check if text contains >50% Bengali characters"""
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        total_chars = len(re.sub(r'\s+', '', text))
        return (bengali_chars / total_chars) > 0.5 if total_chars > 0 else False

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using language-appropriate separators"""
        if not text.strip():
            return []

        # Create regex pattern for sentence boundaries
        pattern = r'(?<=[{}])\s+'.format(''.join(
            re.escape(sep) for sep in self.sentence_separators
        ))
        
        sentences = re.split(pattern, text.strip())
        return [s for s in sentences if s.strip()]

    def split_clauses(self, text: str) -> List[str]:
        """Split long sentences into clauses"""
        pattern = r'(?<=[{}])\s+'.format(''.join(
            re.escape(sep) for sep in self.clause_separators
        ))
        return [c for c in re.split(pattern, text) if c.strip()]

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Main chunking pipeline with metadata tracking
        Returns:
            List of chunks with metadata: {
                "text": chunk content,
                "metadata": {
                    "language": "bn"/"en"/"mixed",
                    "length_chars": int,
                    "has_sentence_end": bool,
                    "overlap_with_next": int
                }
            }
        """
        if not text.strip():
            return []

        # Detect language
        is_bengali = self.is_bengali_dominant(text)
        language = "bn" if is_bengali else "en"
        
        # Split sentences
        sentences = self.split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_buffer = []

        for sentence in sentences:
            sent_length = len(sentence)
            
            # Case 1: Sentence fits within current chunk
            if current_length + sent_length <= self.max_chunk_size:
                current_chunk.append(sentence)
                current_length += sent_length
                continue
                
            # Case 2: Sentence is too long - split into clauses
            if sent_length > self.max_chunk_size:
                clauses = self.split_clauses(sentence)
                for clause in clauses:
                    clause_length = len(clause)
                    
                    if current_length + clause_length > self.max_chunk_size:
                        if current_chunk:
                            chunks.append(self._finalize_chunk(
                                current_chunk, language, overlap_buffer
                            ))
                            current_chunk = overlap_buffer.copy()
                            current_length = sum(len(x) for x in overlap_buffer)
                            overlap_buffer = []
                            
                    current_chunk.append(clause)
                    current_length += clause_length
                continue
                
            # Case 3: Sentence doesn't fit - finalize current chunk
            if current_chunk:
                chunks.append(self._finalize_chunk(
                    current_chunk, language, overlap_buffer
                ))
                current_chunk = overlap_buffer.copy()
                current_length = sum(len(x) for x in overlap_buffer)
                overlap_buffer = []
                
            current_chunk.append(sentence)
            current_length += sent_length
            
            # Update overlap buffer
            if self.overlap > 0:
                overlap_buffer = self._update_overlap_buffer(
                    current_chunk, self.overlap
                )

        # Add final chunk
        if current_chunk and current_length >= self.min_chunk_size:
            chunks.append({
                "text": " ".join(current_chunk),
                "metadata": {
                    "language": language,
                    "length_chars": current_length,
                    "has_sentence_end": any(
                        sentence.endswith(tuple(self.sentence_separators))
                        for sentence in current_chunk
                    ),
                    "overlap_with_next": 0  # No next chunk
                }
            })

        return chunks

    def _finalize_chunk(self, chunk: List[str], language: str, overlap_buffer: List[str]) -> Dict:
        """Prepare a complete chunk with metadata"""
        chunk_text = " ".join(chunk)
        return {
            "text": chunk_text,
            "metadata": {
                "language": language,
                "length_chars": len(chunk_text),
                "has_sentence_end": any(
                    s.endswith(tuple(self.sentence_separators)) for s in chunk
                ),
                "overlap_with_next": sum(len(x) for x in overlap_buffer)
            }
        }

    def _update_overlap_buffer(self, current_chunk: List[str], target_overlap: int) -> List[str]:
        """Maintain overlap context between chunks"""
        overlap_buffer = []
        overlap_length = 0
        
        # Add sentences from end until we reach target overlap
        for sentence in reversed(current_chunk):
            sent_length = len(sentence)
            if overlap_length + sent_length > target_overlap:
                break
            overlap_buffer.insert(0, sentence)
            overlap_length += sent_length
            
        return overlap_buffer

# Example Usage
if __name__ == "__main__":
    # Sample Bengali text with mixed content
    sample_text = """
    মামা বিবাহ-বাড়িতে ঢুকিয়া খুশি হইলেন না। একে তো উঠানটাতে বরযাত্রীদের জায়গা সংকুলান হওয়াই শক্ত,
তাহার পরে সমস্ত আয়োজন নিতান্ত মধ্যম রকমের। ইহার পরে শন্কুনাথবাবুর ব্যবহারটাও নেহাত ঠান্ডা। তার
বিনয়টা অজজ্র নয়। মুখে তো কথাই নাই কোমরে চাদর বাধা, গলা ভাঙা, টাক-পড়া, মিশ-কালো এবং বিপুল-
কন্দর্ট পাটির করতাল-বাজিয়ে হইতে শুরু করিয়া বরকর্তাদের প্রত্যেককে বার বার প্রচুররূপে অভিষিক্ত
করিয়া না দিতেন তবে গোড়াতেই এটা এসপার-ওসপার হইত।
আমি সভায় বসিবার কিছুক্ষণ পরেই মামা শস্তুনাথবাবুকে পাশের ঘরে ডাকিয়া লইয়া গেলেন। কী কথা হইল
জানি না, কিছুক্ষণ পরেই শস্তুনাথবাবু আমাকে আসিইয়া বলিলেন, “বাবাজি, একবার এই দিকে আসতে
হচ্ছে।”
ব্যাপারখানা এই। -সকলে না হউক, কিন্তু কোনো চর পন + ভু
সওগাদ লোক-বিদায় প্রভৃতি সম্বন্ধে যেরকম এ ্
টানাটানির পরিচয় পাওয়া গেছে তাহাতে মামা
ঠিক করিয়াছিলেন- দেওয়া-থোওয়া সম্বন্ধে এ লোকটির শুধু মুখের কথার উপর ভর করা চলিবে না। সেইজন্য
বাড়ির সেকরাকে সুদ্ধ সঙ্গেআনিয়াছিলেন। পাশের ঘরে গিয়া দেখিলাম, মামা এক তক্তপোশে এবং সেকরা
তাহার দীড়িপাল্লা কষ্টিপাথর প্রভৃতি লইয়া মেঝেয় বসিয়া আছে।
শস্তুনাথবাবু আমাকে বলিলেন, “তোমার মামা বলিতেছেন বিবাহের কাজ শুরু হইবার আগেই তিনি কনের
আমি মাথা হেট করিয়া চুপ করিয়া রহিলাম।
মামা বলিলেন, “ও আবার কী বলিবে। আমি যা বলিব তাই হইবে।”
শস্তুনাথবাবু আমার দিকে চাহিয়া কহিলেন, “সেই কথা তবে ঠিক? উনি যা বলিলেন তাই হইবে? এ সম্বন্ধে
আমি একটু ঘাড়-নাড়ার ইঙ্গিতে জানাইলাম, এসব কথায় আমার সম্পুর্ণ অনধিকার।
মামা বলিলেন, “অনুপম এখানে কি করিবে। ও সভায় গিয়ে বসুক।”
কিছুক্ষণ পরে তিনি একখানা গামছায় বাঁধা গহনা আনিয়া তক্তপোশের উপর মেলিয়া ধরিলেন। সমস্তই তাহার
সেকরা গহনা হাতে তুলিয়া লইয়া বলিল, “এ আর দেখিব কী। ইহাতে খাদ নাই-এমন সোনা এখনকার দিনে
ব্যবহারই হয় না।
    """

    # Initialize chunker with recommended settings for Bengali
    chunker = BengaliChunker(
        max_chunk_size=300,
        overlap=50,
        min_chunk_size=30
    )

    # Process text
    chunks = chunker.chunk_text(sample_text)

    # Display results
    print(f"Generated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"Text: {chunk['text']}")
        print(f"Metadata: {chunk['metadata']}\n")