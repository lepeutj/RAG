from src.ingestion.loader import DocumentLoader, document_id_for_source


def test_load_file_uses_logical_source_for_uploaded_document(tmp_path):
    uploaded_file = tmp_path / "temporary-upload.md"
    uploaded_file.write_text("Document content", encoding="utf-8")

    document = DocumentLoader().load_file(uploaded_file, source="refund-policy.md")

    assert document.source == "refund-policy.md"
    assert document.metadata == {
        "document_id": document_id_for_source("refund-policy.md"),
        "filename": "refund-policy.md",
        "extension": ".md",
        "storage_path": str(uploaded_file),
        "managed_storage": False,
    }
