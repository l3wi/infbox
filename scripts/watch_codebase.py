#!/usr/bin/env python3
"""
Code-aware file watcher for LLM inference stack.
Monitors workspace for changes and updates KV cache via LMCache.
"""

import os
import sys
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import Set, Dict, Optional
from datetime import datetime

import requests
import xxhash
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from gitignore_parser import parse_gitignore

# Configuration from environment
WATCH_DIR = os.getenv("WATCH_DIR", "/workspace")
IGNORE_FILE = os.getenv("IGNORE_FILE", ".gitignore")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://vllm:8000")
LMCACHE_ENDPOINT = os.getenv("LMCACHE_ENDPOINT", "http://lmcache:8100")
WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL", "1"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodebaseWatcher(FileSystemEventHandler):
    """Watches codebase for changes and updates KV cache."""
    
    def __init__(self, watch_dir: str, ignore_file: str):
        self.watch_dir = Path(watch_dir)
        self.ignore_file = ignore_file
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
            '.DS_Store', 'node_modules', '.env', '.venv'
        }
        
        path_str = str(path)
        for pattern in ignore_patterns:
            if pattern in path_str:
                return True
        
        # Check gitignore
        if self.gitignore_matcher and self.gitignore_matcher(path_str):
            return True
        
        return False
    
    def _hash_file(self, filepath: Path) -> Optional[str]:
        """Generate hash of file contents."""
        try:
            if not filepath.is_file() or self._should_ignore(filepath):
                return None
            
            # Skip large files
            if filepath.stat().st_size > 10 * 1024 * 1024:  # 10MB
                logger.debug(f"Skipping large file: {filepath}")
                return None
            
            hasher = xxhash.xxh64()
            with open(filepath, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {filepath}: {e}")
            return None
    
    def _initial_scan(self):
        """Perform initial scan of workspace."""
        logger.info(f"Starting initial scan of {self.watch_dir}")
        file_count = 0
        
        for filepath in self.watch_dir.rglob('*'):
            if filepath.is_file():
                file_hash = self._hash_file(filepath)
                if file_hash:
                    rel_path = filepath.relative_to(self.watch_dir)
                    self.file_hashes[str(rel_path)] = file_hash
                    file_count += 1
        
        logger.info(f"Initial scan complete. Indexed {file_count} files")
        self._update_cache(list(self.file_hashes.keys()), "initial_load")
    
    def _update_cache(self, files: list, operation: str):
        """Update LMCache with file changes."""
        try:
            payload = {
                "operation": operation,
                "files": files,
                "timestamp": datetime.utcnow().isoformat(),
                "workspace": str(self.watch_dir)
            }
            
            response = requests.post(
                f"{LMCACHE_ENDPOINT}/update",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Cache updated: {operation} for {len(files)} files")
            else:
                logger.error(f"Cache update failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Error updating cache: {e}")
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        file_hash = self._hash_file(filepath)
        if file_hash:
            rel_path = filepath.relative_to(self.watch_dir)
            self.file_hashes[str(rel_path)] = file_hash
            logger.info(f"File created: {rel_path}")
            self._update_cache([str(rel_path)], "create")
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        file_hash = self._hash_file(filepath)
        if file_hash:
            rel_path = filepath.relative_to(self.watch_dir)
            old_hash = self.file_hashes.get(str(rel_path))
            
            if old_hash != file_hash:
                self.file_hashes[str(rel_path)] = file_hash
                logger.info(f"File modified: {rel_path}")
                self._update_cache([str(rel_path)], "modify")
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        rel_path = filepath.relative_to(self.watch_dir)
        
        if str(rel_path) in self.file_hashes:
            del self.file_hashes[str(rel_path)]
            logger.info(f"File deleted: {rel_path}")
            self._update_cache([str(rel_path)], "delete")
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move/rename."""
        if event.is_directory:
            return
        
        src_path = Path(event.src_path)
        dst_path = Path(event.dest_path)
        
        src_rel = src_path.relative_to(self.watch_dir)
        dst_rel = dst_path.relative_to(self.watch_dir)
        
        # Remove old path
        if str(src_rel) in self.file_hashes:
            del self.file_hashes[str(src_rel)]
        
        # Add new path
        file_hash = self._hash_file(dst_path)
        if file_hash:
            self.file_hashes[str(dst_rel)] = file_hash
            logger.info(f"File moved: {src_rel} -> {dst_rel}")
            self._update_cache([str(src_rel), str(dst_rel)], "move")

def health_check():
    """Check service health."""
    try:
        # Check vLLM
        vllm_response = requests.get(f"{VLLM_ENDPOINT}/health", timeout=5)
        vllm_healthy = vllm_response.status_code == 200
        
        # Check LMCache
        lmcache_response = requests.get(f"{LMCACHE_ENDPOINT}/health", timeout=5)
        lmcache_healthy = lmcache_response.status_code == 200
        
        return vllm_healthy and lmcache_healthy
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def main():
    """Main entry point."""
    logger.info("Starting codebase watcher")
    logger.info(f"Watch directory: {WATCH_DIR}")
    logger.info(f"vLLM endpoint: {VLLM_ENDPOINT}")
    logger.info(f"LMCache endpoint: {LMCACHE_ENDPOINT}")
    
    # Wait for services to be ready
    while not health_check():
        logger.info("Waiting for services to be ready...")
        time.sleep(5)
    
    logger.info("Services are ready")
    
    # Create watcher
    event_handler = CodebaseWatcher(WATCH_DIR, IGNORE_FILE)
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=True)
    
    # Start watching
    observer.start()
    logger.info("File watcher started")
    
    try:
        while True:
            time.sleep(WATCH_INTERVAL)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Watcher stopped by user")
    
    observer.join()

if __name__ == "__main__":
    main()