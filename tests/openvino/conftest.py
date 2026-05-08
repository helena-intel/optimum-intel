import sys

import pytest


def pytest_configure(config):
    """Disable faulthandler on Windows to prevent conflicts with OpenVINO's signal handling.

    OpenVINO uses Windows Structured Exception Handling (SEH) internally for operations
    like CPU feature detection, which triggers and catches access violations as part of
    normal operation. Pytest's faulthandler intercepts these before OpenVINO's handler
    can catch them, causing the process to abort with 'Windows fatal exception: access
    violation'.
    """
    if sys.platform == "win32":
        import faulthandler

        faulthandler.disable()
        config.pluginmanager.unregister(name="faulthandler")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Dynamically add the 'gemma4' marker to every parameterized test whose
    name contains 'gemma4' (this also covers 'gemma4_moe')."""
    gemma4_marker = pytest.mark.gemma4
    for item in items:
        if "gemma4" in item.nodeid:
            item.add_marker(gemma4_marker)
