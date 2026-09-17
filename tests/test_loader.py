from src.ingestion.loader import DocumentLoader


def test_load_file_uses_logical_source_for_uploaded_document(tmp_path):
    uploaded_file = tmp_path / "temporary-upload.md"
    uploaded_file.write_text("Document content", encoding="utf-8")

    document = DocumentLoader().load_file(uploaded_file, source="refund-policy.md")

    assert document.source == "refund-policy.md"
    assert document.metadata == {"filename": "refund-policy.md", "extension": ".md"}
