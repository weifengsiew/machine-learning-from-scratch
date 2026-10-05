# Ruff: Linting and Formatting

Ruff checks Python source for common mistakes and keeps formatting consistent.

Install it as a development dependency if needed:

```bash
python -m pip install ruff
```

Run from the repository root:

```bash
ruff check .
ruff format --check .
```

To apply safe lint fixes and format files:

```bash
ruff check --fix .
ruff format .
```

Linting and formatting are complementary: a file can be correctly formatted while still containing an unused import or undefined name.

Before committing, review the diff after any automatic fix. Do not use unsafe fixes blindly when they may change algorithm behavior.
