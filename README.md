# ContextLex: Metadata-Enhanced RAG Chatbot

ContextLex is a specialized Retrieval-Augmented Generation (RAG) system designed to transform dense book content into an interactive Q/A experience. Unlike standard "naive" RAG systems, this project employs Smart Metadata Chunking to preserve hierarchical context, ensuring the LLM understands not just the text, but where it sits within the book's structure.

The "Smart Chunking" Logic
Standard chunking often breaks sentences in half or loses the "topic" of a paragraph. ContextLex solves this by:

      1. Header-Aware Splitting: Respecting the logical flow of the book's Table of Contents.

      2. Overlapping Windows: Maintaining a 10-15% overlap between chunks to ensure continuity.

      3. Metadata Injection: Prepended to each chunk:
         [Source: Chapter 4, Section 2.1 | Topic: Machine Learning Foundations]
