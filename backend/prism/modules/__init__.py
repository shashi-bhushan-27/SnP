"""Downstream decision modules. Each consumes RiskSignals (prism.core) and nothing else.

A module never imports the ETL, the NLP engine, the store or the API; the composition root wires
it in as a SignalConsumer. A tactical rebalancer (Module A) would live next to `stress/` the same way.
"""
