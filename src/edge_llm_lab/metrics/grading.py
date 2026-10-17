"""Pure grading functions shared by inline, offline and Inspect evaluation."""
from edge_llm_lab.metrics.structured import structured_metrics

PROFILES = ('structured', 'text')


def grade_response(response: str, reference: str, profile: str = 'structured') -> dict:
    """Grade a response without network access, model loading or side effects."""
    if profile == 'structured':
        return structured_metrics(response, reference)
    if profile == 'text':
        return {'exact_match': {'score': float(response == reference),
                                'details': 'Literal text equality; not semantic or safety grading'}}
    raise ValueError(f'Unknown grading profile: {profile}')
