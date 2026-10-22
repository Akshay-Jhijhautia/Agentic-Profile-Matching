"""Run the ``profile-match`` command line from a source checkout: ``python main.py chat``.

Installed with pip, the same commands are available as ``profile-match``.
"""

import sys

from profile_matching.app import main

if __name__ == "__main__":
    sys.exit(main())
