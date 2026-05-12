"""Allow ``python -m data_scraping ...`` to dispatch the CLI."""

from .cli import main

raise SystemExit(main())
