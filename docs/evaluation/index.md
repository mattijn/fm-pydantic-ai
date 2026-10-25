# Evaluation

```bash
uv run --group evals python -m evals.run --repeat 10
```

Every scenario runs ten times. Deterministic checks grade the path of each trial: tool calls and their arguments, questions to the user, rounds and answer. Findings describe the model and are not graded.

```python exec="on"
import json

results = json.load(open("evals/results.json"))
print("| Scenario | Passed |\n|---|---|")
for s in results["scenarios"]:
    if s["checks"]:
        print(f"| {s['title']} | {sum(t['passed'] for t in s['trials'])}/{len(s['trials'])} |")
print("\n| Finding | Trials |\n|---|---|")
for s in results["scenarios"]:
    for f in s["findings"]:
        print(f"| {s['title']}: {f} | {sum(t['findings'][f] for t in s['trials'])}/{len(s['trials'])} |")
print(f"\nRun on {results['generated'][:10]}, macOS {results['environment']['macos']}.")
```

The [report](report.html) shows every trial with its timeline and transcript.
