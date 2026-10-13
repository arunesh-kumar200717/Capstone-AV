from src.evaluation import validate_citation_ids


def test_validate_citation_ids_accepts_valid_references():
    references = ['c1', 'c2']
    retrieved = ['c1', 'c2', 'c3']
    assert validate_citation_ids(references, retrieved) is True


def test_validate_citation_ids_rejects_invalid_references():
    references = ['c99']
    retrieved = ['c1', 'c2']
    assert validate_citation_ids(references, retrieved) is False
