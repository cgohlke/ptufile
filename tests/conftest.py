# ptufile/tests/conftest.py

"""Pytest configuration."""

import os
import sys

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


def pytest_report_header(config: object) -> str:
    """Return pytest report header."""
    try:
        import ptufile

        return (
            f'Python {sys.version.splitlines()[0]}\n'
            f'packagedir: {ptufile.__path__[0]}\n'
            f'version: ptufile {ptufile.__version__}'
        )
    except Exception as exc:
        return f'pytest_report_header failed: {exc!s}'
