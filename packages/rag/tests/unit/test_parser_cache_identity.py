# Test values are synthetic fixtures for software behavior.
from core.parser import DocumentChunk, DocumentParser


def test_cached_chunks_receive_new_document_and_linked_chunk_ids():
    parent = DocumentChunk(
        chunk_id="parent-old", doc_id="old-doc", file_name="old.pdf", modality="text",
        content="content", searchable_text="content", page=1, child_images=["image-old"],
    )
    image = DocumentChunk(
        chunk_id="image-old", doc_id="old-doc", file_name="old.pdf", modality="image",
        content="image", searchable_text="image", page=1, parent_chunk_id="parent-old",
    )

    restored = DocumentParser._restore_cached_identity([parent, image], "new-doc", "new.pdf")

    assert {chunk.doc_id for chunk in restored} == {"new-doc"}
    assert {chunk.file_name for chunk in restored} == {"new.pdf"}
    assert {chunk.chunk_id for chunk in restored}.isdisjoint({"parent-old", "image-old"})
    assert restored[1].parent_chunk_id == restored[0].chunk_id
    assert restored[0].child_images == [restored[1].chunk_id]
