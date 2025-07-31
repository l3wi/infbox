# Scaling to Very Large Codebases

## Current Limitations

### 1. Context Window Constraints
- **Issue**: vLLM has a maximum context length (32K tokens for our model)
- **Impact**: Can only fit ~50-100 files depending on size
- **Problem**: Large codebases have thousands of files

### 2. Memory Limitations
- **KV Cache Size**: Each cached token uses GPU memory
- **Trade-off**: More context = less memory for generation
- **Reality**: A 1M LOC codebase won't fit in any context window

### 3. File Selection Challenge
- **Current**: Static scoring based on filename and imports
- **Problem**: May miss relevant files for specific queries
- **Example**: Searching for a bug might need random utility file

## Scalability Solutions

### Solution 1: Dynamic Context Loading (Recommended)
```python
# Instead of loading everything, load based on query
class DynamicContextManager:
    def __init__(self):
        self.file_index = {}  # Full codebase index
        self.embedding_cache = {}  # File embeddings
    
    def get_context_for_query(self, query):
        # 1. Search relevant files using embeddings
        relevant_files = self.semantic_search(query)
        
        # 2. Add dependency graph neighbors
        extended_files = self.add_dependencies(relevant_files)
        
        # 3. Build hierarchical context
        return self.build_context(extended_files)
```

### Solution 2: Multi-Stage Retrieval
1. **Stage 1**: Lightweight search across all files
2. **Stage 2**: Load only relevant files into context
3. **Stage 3**: Answer query with focused context

### Solution 3: Hybrid Approach
```yaml
# Combine multiple strategies
strategies:
  - core_cache:    # Always cached: main files, configs
      max_files: 20
      selection: static_scoring
  
  - hot_cache:     # Recently accessed files
      max_files: 30
      selection: lru_cache
      
  - query_cache:   # Dynamic based on query
      max_files: 50
      selection: semantic_search
```

## Implementation Approaches

### 1. Embedding-Based Retrieval
```python
# Pre-compute embeddings for all files
def index_codebase():
    for file in all_files:
        embedding = compute_embedding(file.content)
        store_in_vector_db(file.path, embedding)

# Runtime: retrieve relevant files
def get_relevant_files(query, top_k=50):
    query_embedding = compute_embedding(query)
    return vector_db.search(query_embedding, top_k)
```

### 2. Dependency Graph Analysis
```python
# Build import/dependency graph
def build_dependency_graph():
    graph = nx.DiGraph()
    for file in all_files:
        imports = extract_imports(file)
        for imp in imports:
            graph.add_edge(file.path, imp)
    return graph

# Get related files
def get_related_files(file_path, depth=2):
    return nx.ego_graph(graph, file_path, radius=depth)
```

### 3. Intelligent Caching Layers

| Layer | Purpose | Size | Update Frequency |
|-------|---------|------|------------------|
| L1: Core | Entry points, configs | 10-20 files | Rare |
| L2: Hot | Frequently accessed | 20-30 files | Per access |
| L3: Working Set | Current task files | 30-50 files | Per query |
| L4: Cold Storage | Full codebase index | All files | On change |

## Recommended Architecture for Large Codebases

```
┌─────────────────────────────────────────┐
│          Query Interface                 │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         Context Orchestrator            │
│  - Query understanding                  │
│  - Relevance scoring                    │
│  - Context assembly                     │
└────────────────┬────────────────────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
┌────▼────┐ ┌───▼────┐ ┌───▼────┐
│ Core    │ │ Search │ │ Graph  │
│ Cache   │ │ Index  │ │ Cache  │
└─────────┘ └────────┘ └────────┘
     │           │           │
     └───────────┼───────────┘
                 │
┌────────────────▼────────────────────────┐
│            vLLM Instance                │
│  - Prefix caching enabled               │
│  - Dynamic context loading              │
└─────────────────────────────────────────┘
```

## Practical Limits

### What Works Well
- **Small-Medium Codebases**: < 10K files, < 1M LOC
- **Monorepos with Clear Structure**: Can use path-based filtering
- **Domain-Specific Code**: Focused functionality

### What Needs Enhancement
- **Large Enterprises**: 100K+ files, 10M+ LOC
- **Polyglot Codebases**: Many languages and frameworks
- **Legacy Systems**: Poor structure, many dependencies

## Quick Wins for Large Codebases

1. **Path Filtering**
   ```python
   # Only watch specific directories
   WATCH_PATHS = ["/src/core", "/src/api", "/config"]
   ```

2. **File Type Priorities**
   ```python
   # Focus on primary language
   PRIMARY_EXTENSIONS = [".py", ".pyx"]
   SECONDARY_EXTENSIONS = [".yml", ".json"]
   ```

3. **Smart Truncation**
   ```python
   # Keep headers and interfaces, truncate implementations
   def smart_truncate(content):
       # Keep imports, class definitions, function signatures
       # Truncate function bodies over 50 lines
   ```

## Conclusion

For very large codebases, you need:
1. **Selective Loading**: Not everything in context
2. **Smart Search**: Find relevant files dynamically
3. **Tiered Caching**: Different strategies for different file types
4. **External Tools**: Combine with code search engines

The current approach works well for small-medium codebases but needs enhancement for enterprise-scale systems.