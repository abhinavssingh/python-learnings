# Chunking Documents

Chunking divides long documents into smaller passages that can be embedded and retrieved. The goal is not simply to make every piece the same size: boundaries should preserve enough context for a later question to be answered accurately.

## Why chunking matters

Large documents are difficult to retrieve and reason about as a single block. A question like:

```text
What is the rule for paid parental leave?
```

may be answered better from a small chunk containing the exact policy section than from a very long document that contains many unrelated sections.

If chunking is poor, the retrieval pipeline can fail in two ways:

1. It retrieves chunks that are too large and contain mixed topics.
2. It retrieves chunks that are too small and lose the context needed to answer the question.

This is why chunking is a major design choice in any retrieval system.

## Core idea

A chunk should be:
- small enough to be precise
- large enough to include relevant context
- ideally aligned with natural boundaries like paragraphs, headings, or code blocks

Example:

```text
## Annual Leave Policy
Employees can request 20 days of annual leave each year.
Requests must be submitted 30 days in advance.
```

A good chunk preserves this whole section together.

A bad chunk might split it like this:

```text
Employees can request 20 days of annual leave each year.
```

and

```text
Requests must be submitted 30 days in advance.
```

This is still usable, but if a user asks "What is the annual leave rule?", the second chunk alone loses the context that makes it a rule.

## Lessons

### 1. Chunking strategies (`01_chunking_strategies.py`)

This lesson compares multiple chunking strategies:
- fixed-size chunks
- recursive splitting
- sentence-based splitting
- paragraph-based splitting
- Markdown-aware splitting
- code-aware splitting

#### Fixed-size chunking
This splits text into chunks of a chosen size, usually by character count or token count.

Example:

```text
Original text:
"Employees may take 20 days of annual leave. Requests must be submitted 30 days in advance."
```

If the chunk size is 40 characters, it may split in the middle of a sentence or rule.

This is easy to implement, but it can break logic across boundaries.

#### Recursive chunking
This method tries to split on natural boundaries first, such as paragraphs or sentences, before falling back to a size limit.

Example:

```text
### Annual Leave
Employees may take 20 days of annual leave.

### Sick Leave
Employees may take 10 days of sick leave.
```

A recursive splitter will keep each section together before chopping it into smaller pieces if needed.

#### Sentence-based chunking
This method splits by sentence.

Example:

```text
"Employees may take 20 days of annual leave. Requests must be submitted 30 days in advance."
```

becomes:

```text
Chunk 1: "Employees may take 20 days of annual leave."
Chunk 2: "Requests must be submitted 30 days in advance."
```

This often works well for strict factual retrieval because each chunk remains readable.

#### Paragraph-based chunking
This keeps paragraphs together.

Example:

```text
The company offers 20 days of annual leave.
Requests must be submitted 30 days in advance.
```

This is useful when one paragraph contains a complete idea.

#### Markdown-aware splitting
Markdown headings can provide semantic boundaries.

Example:

```md
## Annual Leave
Employees may take 20 days of annual leave.

## Benefits
Employees receive health insurance after 3 months.
```

A Markdown-aware splitter keeps the heading and its content together, which is valuable because headings often explain the context of the policy.

#### Code-aware splitting
For code or configuration text, boundaries should respect blocks and functions rather than arbitrary character counts.

Example:

```python
def request_leave(days):
    if days > 20:
        raise ValueError("Too many days")
    return "Approved"
```

A code-aware splitter should not cut through the function body if the function is the semantically meaningful unit.

### 2. Semantic chunking (`02_semantic_chunking.py`)

This lesson uses embedding similarity to detect topic boundaries.

Instead of splitting by fixed counts, it compares nearby segments and decides where a topic shift occurs.

Example:

```text
Paragraph 1: "Employees may request paid parental leave after one year of service."
Paragraph 2: "Travel expenses are reimbursable only with prior approval."
```

These paragraphs belong to different topics. A semantic chunker will likely detect a strong shift in meaning and create a boundary there.

#### Why semantic chunking helps
A document may have a gradual transition from one topic to another. A fixed-length chunk might cut in the middle of a policy rule, while semantic chunking tries to keep each topical section together.

Example:

```text
The company provides paid parental leave for up to 12 weeks.
Employees must apply online before the date of birth.
```

This is cohesive and should likely remain in one chunk.

But a switch to a different domain section such as:

```text
Expense reimbursement for business travel is handled separately.
```

should usually trigger a new chunk.

## Concepts to notice

- Chunk size trades context against retrieval precision and embedding cost.
- Overlap can preserve a sentence split across a boundary, but excessive overlap duplicates content and may crowd retrieval results.
- Metadata such as source and page should travel with each chunk so answers can be traced back to their origin.
- Semantic chunking requires an embedding model; the first lesson is suitable for quick local experiments without one.

## Chunking design trade-offs

### Larger chunks
Pros:
- more context
- better for multi-sentence questions
- easier to understand in context

Cons:
- more irrelevant text
- lower precision
- higher embedding and retrieval cost

### Smaller chunks
Pros:
- high precision
- easier to match exact facts
- lower retrieval cost

Cons:
- may lose context
- may remove the surrounding rule or heading
- can break a sentence or concept across chunks

### Overlap
Overlap is often used to keep a little context around boundaries.

Example:

```text
Chunk 1: "Employees may request 20 days of annual leave. Requests must be submitted"
Chunk 2: "submitted 30 days in advance. The leave is paid."
```

This can help preserve continuity, but too much overlap creates duplicate content and noisy retrieval.

## Run

From the repository root:

```powershell
python Module-5/GENAI/05_chunking/01_chunking_strategies.py
python Module-5/GENAI/05_chunking/02_semantic_chunking.py
```

The semantic lesson uses the configured local embedding model. See the [GenAI setup guide](../README.md) for Ollama prerequisites. Reports are saved in this folder's `reports/` directory.

## Practice

Choose a short document with headings and multi-sentence paragraphs. Run multiple splitting strategies, then inspect the resulting chunks manually. Test each chunk by asking: could someone understand it without the previous paragraph, and does it contain more unrelated information than necessary?

### Example practice document

```md
## Annual Leave
Employees may take 20 days of annual leave each year.
Requests must be submitted 30 days in advance.

## Benefits
Employees are eligible for health insurance after 3 months of service.

## Travel Expenses
Business travel expenses are reimbursable with approval.
```

Now compare chunking strategies:

- Fixed-size splitting may split the annual leave section across chunks.
- Sentence splitting may keep each sentence together.
- Markdown-aware splitting keeps the section label with the rule.
- Semantic chunking would likely keep each policy topic separate.

This shows that chunking is not just technical; it is a way to preserve document structure and meaning.

## Big picture

Chunking is the bridge between documents and retrieval.

The pipeline often looks like this:

```text
document -> chunking -> embedding -> vector store -> retrieval -> LLM answer
```

If chunking is poor, even a strong embedding model will retrieve weak evidence. If chunking is well designed, the model gets precise and relevant passages to work from.

This is especially important in policy and enterprise documents, where a small factual error can have major consequences.
