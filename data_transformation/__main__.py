"""Allow ``python -m data_transformation ...`` to dispatch the CLI."""

from .cli import main

raise SystemExit(main())
