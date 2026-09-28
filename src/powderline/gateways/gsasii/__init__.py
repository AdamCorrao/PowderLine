"""GSAS-II gateway package.

Do not import runtime modules (setters, project, executors, extractors, ...)
here: they import GSAS-II at module top. The engine-free gateway factory
arrives in a later branch (re/02) as ``gateways/gsasii/gateway.py``.
"""
