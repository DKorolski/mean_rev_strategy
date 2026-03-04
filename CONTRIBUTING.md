# Contributing

## Setup

1. Create and activate virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Checks

```bash
pytest
ruff check .
black --check .
```

## Notes

- Keep commits focused.
- Avoid mixing strategy logic changes with housekeeping changes.
