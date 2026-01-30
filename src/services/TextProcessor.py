import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from logger_config import get_logger

logger = get_logger(__name__)

class FileLoader:
    def __init__(self, directory: str = "data"):
        self.directory = directory

    def load_files(self):
        #Load all .txt files from the specified directory.
        files = []
        if not os.path.exists(self.directory):
            logger.error(f"Directory does not exist: {self.directory}")
            return files

        for root, _, filenames in os.walk(self.directory):
            for fname in filenames:
                if fname.lower().endswith('.txt'):
                    path = os.path.join(root, fname)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            files.append((os.path.splitext(fname)[0].replace("_", " ").title(), content))
                            logger.info(f"Loaded file: {path}")
                    except Exception as e:
                        logger.error(f"Failed to load file {path}: {e}")
        if not files:
            logger.warning("No text files found in directory.")
        return files

      
class TextChunker:
    def __init__(self, documents: List[dict]=None, chunk_size: int=1000, chunk_overlap: int=100):
      if documents is None:
          raise ValueError("documents must be provided")

      if not documents:
          raise ValueError("documents list is empty")

      self.documents = documents
      self.chunk_size = chunk_size
      self.chunk_overlap = chunk_overlap
      self.text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=self.chunk_size,
          chunk_overlap=self.chunk_overlap
      )

    def split_into_chunks(self, filename: str, text: str) -> List[dict]:
        #Split a single document into chunks with metadata.
        metadata = {
            'filename': filename,
        }

        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split {filename} into {len(chunks)} chunks.")
            return [{'text': chunk, 'metadata': metadata} for chunk in chunks]
        except Exception as e:
            logger.error(f"Failed to split file {filename}: {e}")
            return []

    def split_docs(self) -> List[dict]:
        #Split all loaded documents into chunks.
        all_chunks = []
        try:
          for fname, content in self.documents:
              chunks = self.split_into_chunks(fname, content)
              all_chunks.extend(chunks)
          logger.info(f"Total chunks generated: {len(all_chunks)}")
          return all_chunks
        except Exception as e:
            logger.error(f"Failed to split file: {e}")
            return []