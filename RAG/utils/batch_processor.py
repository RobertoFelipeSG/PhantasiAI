import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Optional
from tqdm import tqdm
from .processor import DocumentProcessor

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles batch processing of large document collections with progress tracking."""

    def __init__(
        self,
        max_workers: int = 4,
        processor: Optional[DocumentProcessor] = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        splitting_strategy: str = 'recursive'
    ):
        self.max_workers = max_workers
        self.processor = processor or DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitting_strategy=splitting_strategy
        )
        self.progress_callbacks = []

    def register_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """Register a callback for progress updates.

        Args:
            callback: Function that takes (current_count, total_count, status_message)
        """
        self.progress_callbacks.append(callback)

    def _update_progress(self, current: int, total: int, message: str) -> None:
        """Send progress updates to all registered callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(current, total, message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def process_batch(
        self,
        file_paths: List[str],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Process multiple documents in parallel with progress tracking.

        Args:
            file_paths: List of paths to documents to process
            show_progress: Whether to display a progress bar (uses tqdm)

        Returns:
            Dict containing results and statistics
        """
        start_time = time.time()
        results = []
        errors = []

        def process_file(file_path):
            try:
                return {
                    "file_path": file_path,
                    "success": True,
                    "chunks": self.processor.process_document(file_path),
                    "error": None
                }
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return {
                    "file_path": file_path,
                    "success": False,
                    "chunks": None,
                    "error": str(e)
                }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_file, path): path for path in file_paths}

            if show_progress:
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing documents"
                ):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        errors.append({"file_path": file_path, "error": str(e)})

                    # Update progress through callbacks
                    self._update_progress(
                        len(results) + len(errors),
                        len(file_paths),
                        f"Processed {len(results)} documents successfully, {len(errors)} failed"
                    )
            else:
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        errors.append({"file_path": file_path, "error": str(e)})

        # Compile statistics
        total_chunks = sum(len(r["chunks"]) for r in results if r["success"])
        successful_files = sum(1 for r in results if r["success"])
        elapsed_time = time.time() - start_time

        logger.info(
            f"Processed {successful_files}/{len(file_paths)} files in {elapsed_time:.2f}s, "
            f"generated {total_chunks} chunks"
        )

        return {
            "results": results,
            "errors": errors,
            "stats": {
                "total_files": len(file_paths),
                "successful_files": successful_files,
                "failed_files": len(errors),
                "total_chunks": total_chunks,
                "elapsed_time": elapsed_time
            }
        }

    def process_large_file(
        self,
        file_path: str,
        max_size_mb: int = 100,
        temp_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle very large files by splitting them into manageable chunks first.

        This is useful for files that are too large to process in memory at once.

        Args:
            file_path: Path to the large document
            max_size_mb: Maximum chunk size in MB
            temp_dir: Directory to store temporary split files

        Returns:
            Dict containing results and statistics
        """
        # Implementation depends on file type - this is a simplified version
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        if file_size_mb <= max_size_mb:
            # File is small enough to process directly
            result = self.processor.process_document(file_path)
            return {
                "chunks": result,
                "stats": {
                    "original_size_mb": file_size_mb,
                    "chunk_count": len(result)
                }
            }

        # For larger files, we would implement a type-specific splitter here
        # This is a placeholder for that logic
        logger.warning(
            f"File {file_path} is {file_size_mb:.2f}MB which exceeds the "
            f"recommended size of {max_size_mb}MB. Processing may be slow."
        )

        # Process anyway as a demonstration
        result = self.processor.process_document(file_path)
        return {
            "chunks": result,
            "stats": {
                "original_size_mb": file_size_mb,
                "chunk_count": len(result),
                "warning": f"File exceeded recommended size of {max_size_mb}MB"
            }
        }
