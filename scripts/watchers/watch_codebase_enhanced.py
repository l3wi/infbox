#!/usr/bin/env python3
"""
Enhanced code-aware file watcher for LLM inference stack.
Monitors workspace for changes and prepopulates vLLM cache.
"""

import os
import sys
import time
import logging
import hashlib
import json
import threading
from pathlib import Path
from typing import Set, Dict, Optional, List
from datetime import datetime
from collections import deque

import requests
import xxhash
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from gitignore_parser import parse_gitignore

# Configuration from environment
WATCH_DIR = os.getenv("WATCH_DIR", "/workspace")
IGNORE_FILE = os.getenv("IGNORE_FILE", ".gitignore")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://vllm:8000")
WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CACHE_BATCH_SIZE = int(os.getenv("CACHE_BATCH_SIZE", "5"))
CACHE_MAX_FILE_SIZE = int(os.getenv("CACHE_MAX_FILE_SIZE", "100000"))  # 100KB
EXTRA_IGNORE_DIRS = os.getenv("EXTRA_IGNORE_DIRS", "").split(",") if os.getenv("EXTRA_IGNORE_DIRS") else []

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheManager:
    """Manages prepopulating the vLLM cache with file contents."""
    
    def __init__(self, vllm_endpoint: str):
        self.vllm_endpoint = vllm_endpoint
        self.cache_queue = deque()
        self.cached_files = {}  # file_path -> content_hash
        self.model_id = None
        self.max_model_len = 8192  # Default fallback
        self._get_model_info()
        
    def _get_model_info(self):
        """Get the model ID and max context length from vLLM."""
        try:
            response = requests.get(f"{self.vllm_endpoint}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                if models['data']:
                    model_data = models['data'][0]
                    self.model_id = model_data['id']
                    self.max_model_len = model_data.get('max_model_len', 8192)
                    logger.info(f"Using model: {self.model_id}")
                    logger.info(f"Max context length: {self.max_model_len} tokens")
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token for code)."""
        return len(text) // 4
    
    def _create_file_context(self, files: List[Dict]) -> str:
        """Create a context string from files."""
        context = "You are a code assistant. The following files are part of the codebase:\n\n"
        
        for file_info in files:
            context += f"=== File: {file_info['path']} ===\n"
            context += f"```{file_info.get('language', '')}\n"
            context += file_info['content'][:CACHE_MAX_FILE_SIZE]
            context += "\n```\n\n"
        
        return context
    
    def cache_file(self, file_info: Dict):
        """Cache a single file in vLLM for optimal prefix caching."""
        if not self.model_id or not file_info:
            return
        
        # Check if file is already cached with same content
        if file_info['path'] in self.cached_files:
            if self.cached_files[file_info['path']] == file_info['hash']:
                return  # Already cached
        
        # Estimate tokens for this file
        system_prompt = f"You are analyzing the file: {file_info['path']}\n\n"
        file_content = f"```{file_info.get('language', '')}\n{file_info['content']}\n```"
        total_tokens = self._estimate_tokens(system_prompt + file_content) + 100  # Reserve for messages
        
        # Skip if file is too large
        if total_tokens > self.max_model_len - 200:
            logger.warning(f"File {file_info['path']} too large ({total_tokens} tokens) - skipping")
            return
        
        # Create a cache-warming request for this specific file
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt + file_content},
                {"role": "user", "content": "Analyze this file and be ready to answer questions about it."}
            ],
            "max_tokens": 10,
            "temperature": 0
        }
        
        try:
            start = time.time()
            response = requests.post(
                f"{self.vllm_endpoint}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                logger.info(f"Cached {file_info['path']} ({len(file_info['content'])} bytes, ~{total_tokens} tokens) in {elapsed:.2f}s")
                self.cached_files[file_info['path']] = file_info['hash']
            else:
                logger.error(f"Failed to cache {file_info['path']}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error caching {file_info['path']}: {e}")
    
    def add_to_queue(self, file_info: Dict):
        """Add a file to the caching queue."""
        # Check if already cached with same content
        if file_info['path'] in self.cached_files:
            if self.cached_files[file_info['path']] == file_info['hash']:
                return  # Already cached
        
        self.cache_queue.append(file_info)
        
        # Process queue if batch size reached
        if len(self.cache_queue) >= CACHE_BATCH_SIZE:
            self.process_queue()
    
    def process_queue(self):
        """Process the caching queue - cache files individually."""
        if not self.cache_queue:
            return
        
        # Process up to CACHE_BATCH_SIZE files
        processed = 0
        while self.cache_queue and processed < CACHE_BATCH_SIZE:
            file_info = self.cache_queue.popleft()
            self.cache_file(file_info)
            processed += 1

class CodebaseWatcher(FileSystemEventHandler):
    """Watches codebase for changes and updates vLLM cache."""
    
    def __init__(self, watch_dir: str, ignore_file: str, cache_manager: CacheManager):
        self.watch_dir = Path(watch_dir)
        self.ignore_file = ignore_file
        self.cache_manager = cache_manager
        self.file_hashes: Dict[str, str] = {}
        self.gitignore_matcher = None
        self._load_gitignore()
        self._initial_scan()
    
    def _load_gitignore(self):
        """Load .gitignore patterns."""
        gitignore_path = self.watch_dir / self.ignore_file
        if gitignore_path.exists():
            self.gitignore_matcher = parse_gitignore(gitignore_path)
            logger.info(f"Loaded .gitignore from {gitignore_path}")
        else:
            self.gitignore_matcher = lambda x: False
            logger.warning(f"No .gitignore found at {gitignore_path}")
    
    def _should_ignore(self, path: Path) -> bool:
        """Check if file should be ignored."""
        # Always ignore certain patterns
        ignore_patterns = {
            '.git', '__pycache__', '.pyc', '.pyo', '.swp', 
            '.DS_Store', 'node_modules', '.env', '.venv',
            'models', 'cache', '__pycache__'
        }
        
        path_str = str(path)
        
        # Check extra ignore directories from environment
        for ignore_dir in EXTRA_IGNORE_DIRS:
            if ignore_dir and ignore_dir in path_str:
                return True
        
        for pattern in ignore_patterns:
            if pattern in path_str:
                return True
        
        # Check gitignore
        if self.gitignore_matcher and self.gitignore_matcher(path_str):
            return True
        
        # Skip binary and media files
        binary_extensions = {'.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe',
                           '.bin', '.dat', '.db', '.sqlite', '.jpg', '.jpeg', 
                           '.png', '.gif', '.bmp', '.ico', '.svg', '.mp3', '.mp4',
                           '.avi', '.mov', '.pdf', '.zip', '.tar', '.gz', '.rar',
                           '.7z', '.dmg', '.pkg', '.deb', '.rpm', '.iso', '.jar',
                           '.war', '.ear', '.whl', '.egg', '.gem', '.nupkg'}
        
        if path.suffix.lower() in binary_extensions:
            return True
        
        return False
    
    def _get_file_info(self, filepath: Path) -> Optional[Dict]:
        """Get file information for caching."""
        try:
            if not filepath.is_file() or self._should_ignore(filepath):
                return None
            
            # Skip large files
            if filepath.stat().st_size > CACHE_MAX_FILE_SIZE:
                logger.debug(f"Skipping large file: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            hasher = xxhash.xxh64()
            hasher.update(content.encode('utf-8'))
            file_hash = hasher.hexdigest()
            
            rel_path = filepath.relative_to(self.watch_dir)
            
            # Detect language from extension
            ext_to_lang = {
                '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                '.go': 'go', '.rs': 'rust', '.java': 'java', '.cpp': 'cpp',
                '.rb': 'ruby', '.php': 'php', '.swift': 'swift', '.c': 'c',
                '.h': 'c', '.hpp': 'cpp', '.cs': 'csharp', '.kt': 'kotlin',
                '.scala': 'scala', '.r': 'r', '.m': 'objc', '.sh': 'bash',
                '.yml': 'yaml', '.yaml': 'yaml', '.json': 'json', '.xml': 'xml',
                '.md': 'markdown', '.rst': 'rst', '.txt': 'text', '.toml': 'toml',
                '.ini': 'ini', '.cfg': 'ini', '.conf': 'conf', '.env': 'env',
                '.dockerfile': 'dockerfile', '.makefile': 'makefile', '.mk': 'makefile'
            }
            language = ext_to_lang.get(filepath.suffix.lower(), '')
            
            # Check for files without extensions
            if not language and filepath.name.lower() in ['dockerfile', 'makefile', 'caddyfile']:
                language = filepath.name.lower()
            
            return {
                'path': str(rel_path),
                'content': content,
                'hash': file_hash,
                'language': language
            }
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return None
    
    def _initial_scan(self):
        """Perform initial scan of workspace and cache all files."""
        logger.info(f"Starting initial scan of {self.watch_dir}")
        file_count = 0
        cached_count = 0
        
        for filepath in self.watch_dir.rglob('*'):
            file_info = self._get_file_info(filepath)
            if file_info:
                self.file_hashes[file_info['path']] = file_info['hash']
                self.cache_manager.cache_file(file_info)
                file_count += 1
                cached_count += 1
        
        logger.info(f"Initial scan complete. Indexed {file_count} files")
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        file_info = self._get_file_info(filepath)
        if file_info:
            self.file_hashes[file_info['path']] = file_info['hash']
            logger.info(f"File created: {file_info['path']}")
            self.cache_manager.add_to_queue(file_info)
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        file_info = self._get_file_info(filepath)
        if file_info:
            old_hash = self.file_hashes.get(file_info['path'])
            
            if old_hash != file_info['hash']:
                self.file_hashes[file_info['path']] = file_info['hash']
                logger.info(f"File modified: {file_info['path']}")
                self.cache_manager.add_to_queue(file_info)
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        rel_path = filepath.relative_to(self.watch_dir)
        path_str = str(rel_path)
        
        if path_str in self.file_hashes:
            del self.file_hashes[path_str]
            # Remove from cache tracking
            if path_str in self.cache_manager.cached_files:
                del self.cache_manager.cached_files[path_str]
            logger.info(f"File deleted: {path_str}")

def health_check():
    """Check service health."""
    try:
        vllm_response = requests.get(f"{VLLM_ENDPOINT}/health", timeout=5)
        return vllm_response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def periodic_cache_flush(cache_manager: CacheManager, interval: int = 5):
    """Periodically flush the cache queue."""
    while True:
        time.sleep(interval)
        cache_manager.process_queue()

def main():
    """Main entry point."""
    logger.info("Starting enhanced codebase watcher")
    logger.info(f"Watch directory: {WATCH_DIR}")
    logger.info(f"vLLM endpoint: {VLLM_ENDPOINT}")
    logger.info(f"Cache batch size: {CACHE_BATCH_SIZE}")
    
    # Wait for services to be ready
    while not health_check():
        logger.info("Waiting for vLLM to be ready...")
        time.sleep(5)
    
    logger.info("vLLM is ready")
    
    # Create cache manager
    cache_manager = CacheManager(VLLM_ENDPOINT)
    
    # Start periodic cache flush thread
    flush_thread = threading.Thread(
        target=periodic_cache_flush,
        args=(cache_manager,),
        daemon=True
    )
    flush_thread.start()
    
    # Create watcher
    event_handler = CodebaseWatcher(WATCH_DIR, IGNORE_FILE, cache_manager)
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=True)
    
    # Start watching
    observer.start()
    logger.info("File watcher started with proactive caching")
    
    try:
        while True:
            time.sleep(WATCH_INTERVAL)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Watcher stopped by user")
    
    observer.join()

if __name__ == "__main__":
    main()