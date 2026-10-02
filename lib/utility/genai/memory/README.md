# Memory

Conversation-memory implementations for maintaining context across turns.
`ConversationMemory` defines the common interface; available strategies include
buffer, windowed, summary, and vector memory.

Choose a memory strategy based on how much history should be retained and how
it should be retrieved. `ConversationalRAG` uses a buffer by default and accepts
a memory implementation when configured explicitly. See
`Module-5/GENAI/08_rag/03_conversational_rag_memory.py`.
