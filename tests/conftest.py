# ptufile/tests/conftest.py

import os
import sys

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


def pytest_report_header(config):
    try:
        pyversion = f'Python {sys.version.splitlines()[0]}'
        import ptufile

        return '{}\npackagedir: {}\nversion: ptufile {}'.format(
            pyversion,
            ptufile.__path__[0],
            ptufile.__version__,
        )
    except Exception as exc:
        return f'pytest_report_header failed: {exc!s}'
