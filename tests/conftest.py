# ptufile/tests/conftest.py

"""Pytest configuration."""

import os
import sys

import pytest

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


def pytest_configure(config):
    """Disable matplotlib when running tests with multiple threads."""
    if int(config.getoption('parallel_threads', default=1)) > 1:
        os.environ['PTUFILE_DISABLE_PYPLOT'] = '1'


def pytest_collection_modifyitems(items):
    """Run tests using caplog in a single thread."""
    for item in items:
        if 'caplog' in getattr(item, 'fixturenames', ()):
            item.add_marker(pytest.mark.parallel_threads(1))


def pytest_report_header(config):
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


collect_ignore = ['data']

# mypy: ignore-errors
