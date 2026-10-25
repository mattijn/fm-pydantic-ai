# Evaluation

```python exec="on" session="eval"
import sys

sys.path.insert(0, ".")
from evals import charts

results = charts.load()
print(charts.lead(results))
```

## The bridge holds in every trial

```python exec="on" html="on" session="eval"
print(charts.scoreboard(results))
```

## The model trusts itself over a tool

```python exec="on" html="on" session="eval"
print(charts.findings(results))
```

`get_lat_long` returns placeholder coordinates, `(0, 0)`, and the model fills in Paris and London from memory. A bare "Kampen" is the district in Oslo. A tool to ask the user goes unused, while `get_user_location`, for what the model cannot know, is called every time.

## Every tool round is another generation

```python exec="on" html="on" session="eval"
print(charts.time_split(results))
```

Each block is one Apple generation. Apple generates again after every tool result, so the number of rounds sets the pace, not the tools.

## One question, two protocols

```python exec="on" html="on" session="eval"
print(charts.protocols(results))
```

1. The model needs your city. On the classic protocol the server asks over the open connection while the generation waits.
2. On the stateless protocol the call ends with `input_required`. The client asks you and calls again with the answer.
3. The second call regenerates the turn, and the bridge replays your answer when the model asks again.

Run it yourself: `uv run --group evals python -m evals.run --repeat 10`
