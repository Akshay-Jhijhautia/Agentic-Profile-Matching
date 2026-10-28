"""Run the ``resume-shortlister`` command line from a source checkout: ``python main.py chat``.

Installed with pip, the same commands are available as ``resume-shortlister``.
"""

import sys

from resume_shortlister.app import main

if __name__ == "__main__":
    sys.exit(main())
