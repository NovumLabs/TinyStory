# Examples

`prompts.txt` contains story openings that work well with small models:
complete sentences with a named character give better results than short
fragments such as "Mia and her cat".

Generate a story for every prompt (bash):

```bash
while IFS= read -r prompt; do
  python generate.py "$prompt" --temperature 0.5 --max-new-tokens 150
  echo
done < examples/prompts.txt
```

PowerShell:

```powershell
Get-Content examples/prompts.txt | ForEach-Object {
  python generate.py $_ --temperature 0.5 --max-new-tokens 150
  ""
}
```

Useful sampling settings for a ~4M-parameter model:

| Temperature | Behaviour |
|-------------|-----------|
| 0.2         | Very safe, but loops are common |
| 0.5 - 0.6   | Best balance of coherence and variety |
| 0.8+        | More variety, but logic and grammar degrade |
