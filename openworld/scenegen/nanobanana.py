"""Back-compat shim — nanobanana is now one *mode* under ``scenegen.modes``.

The implementation moved when nanobanana stopped being the only scene-generation
backend:

=======================================  =====================================
old                                      new
=======================================  =====================================
``scenegen.nanobanana.nanobanana_edit``  ``scenegen.modes.nanobanana.nanobanana_edit``
``scenegen.nanobanana.build_suite``      ``scenegen.builder.build_suite``
``scenegen.nanobanana.resolve_base``     ``scenegen.views.resolve_base``
=======================================  =====================================

Prefer importing from :mod:`openworld.scenegen` directly. This module re-exports
the moved names so existing scripts keep working.
"""

from openworld.scenegen.builder import build_suite, build_suite_from_spec
from openworld.scenegen.modes.nanobanana import (
    EDIT_WIDTH,
    MODEL,
    NanobananaMode,
    nanobanana_edit,
)
from openworld.scenegen.views import ALL_VIEWS as VIEWS
from openworld.scenegen.views import resolve_base

__all__ = [
    "nanobanana_edit",
    "resolve_base",
    "build_suite",
    "build_suite_from_spec",
    "NanobananaMode",
    "VIEWS",
    "MODEL",
    "EDIT_WIDTH",
]
