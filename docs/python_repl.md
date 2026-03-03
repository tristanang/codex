# Python REPL (`python_repl`)

`python_repl` runs Python code in a persistent Monty-backed REPL.

## Behavior

- State persists across calls in the same session.
- Output returns the expression result and captured `print(...)` output.
- Calls are isolated per session.

## Sandbox limits

The baseline `python_repl` integration does not wire filesystem or shell host callbacks yet.

- `read_file(...)`, `write_file(...)`, `list_files(...)`, and `shell(...)` are currently unavailable.
- Filesystem and network access are not available through this tool.

## Input schema

`python_repl` accepts a single argument:

- `code` (string): Python source to execute in the current session REPL.

## Example

```json
{
  "code": "counter = 41"
}
```

```json
{
  "code": "counter + 1"
}
```
