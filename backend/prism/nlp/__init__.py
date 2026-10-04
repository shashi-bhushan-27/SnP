"""The NLP Risk Engine: text -> (entity, sentiment, event type, impact, confidence).

Depends only on prism.core. Each component sits behind a protocol in core.interfaces.
"""

from prism.nlp.engine import RiskEngine

__all__ = ["RiskEngine"]
