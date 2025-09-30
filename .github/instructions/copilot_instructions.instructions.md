---
applyTo: '**'
---

## Error Handling
- Never use silent defaults or fallback values.  
- If a file path, configuration, or resource is missing, raise an explicit error (e.g. `FileNotFoundError`, `ValueError`, or `AssertionError`).  
- Fail fast: problems must surface immediately at runtime, not later through hidden defaults.  
- Do not generate code that silently replaces user input with arbitrary fallback values.  

## Assertions
- Prefer `assert` statements for internal sanity checks.  
- Use explicit exceptions (`raise ValueError("...")`, etc.) for user-facing validation.  
- Avoid try/except blocks that catch all errors unless explicitly required.  

## General Style
- Be strict and explicit. Do not guess or “make things work” silently.  
- Generate clean, minimal, and readable code with clear error messages.  
- Stick to standard library solutions unless the task explicitly requires external packages.  
