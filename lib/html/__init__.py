"""
HTML Report Builder Module

Provides a single-interface class-based approach to building styled HTML reports.

Usage:
    from lib.html import HtmlBuilder
    
    builder = HtmlBuilder()
    html = builder.build_page(
        "Report Title",
        builder.grid([
            builder.card("Card 1", builder.render_array(np_array)),
            builder.card("Card 2", builder.render_dict({"key": "value"}))
        ])
    )
"""

from .builder import HtmlBuilder

__all__ = ["HtmlBuilder"]
