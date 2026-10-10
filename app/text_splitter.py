from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=400, overlap=50): 
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    return chunks


from document_loader import extract_text_from_pdf

if __name__ == "__main__":
    text = extract_text_from_pdf("data/sample.pdf")

    chunks = split_text(
        text,
        chunk_size=400,
        overlap=50
    )

    print("Document length:", len(text))
    print("Number of chunks:", len(chunks))

    print("\n--- Chunk 1 ---")
    print(chunks[0])

    print("\n--- Chunk 2 ---")
    print(chunks[1])