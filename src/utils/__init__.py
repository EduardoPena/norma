"""Utilities"""
from .result_saver import ResultSaver

try:
    from .table_context import TableContextBuilder
except (ImportError, ValueError):
    TableContextBuilder = None

__all__ = ['ResultSaver', 'TableContextBuilder']