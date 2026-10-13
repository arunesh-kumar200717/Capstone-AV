from pathlib import Path

from src.document_loader import stable_document_id, validate_pdf


def test_validate_pdf_rejects_non_pdf(tmp_path):
    file = tmp_path / 'example.txt'
    file.write_text('hello')
    ok, message = validate_pdf(str(file))
    assert ok is False
    assert 'Only PDF files are accepted.' in message


def test_stable_document_id_is_repeatable():
    a = stable_document_id('paper-one.pdf')
    b = stable_document_id('paper-one.pdf')
    assert a == b
    assert len(a) == 12
