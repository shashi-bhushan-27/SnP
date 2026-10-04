"""ETL: Extract (sources) -> Transform (normalize, filter, dedupe, analyze) -> Load (store, sinks).

Depends only on prism.core. The NLP engine and the store are injected as protocols.
"""
