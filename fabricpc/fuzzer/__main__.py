"""Allow ``python -m fabricpc.fuzzer``."""

import sys

from fabricpc.fuzzer.cli import main

sys.exit(main())
