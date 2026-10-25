# fm-pydantic-ai

[Pydantic AI](https://ai.pydantic.dev) on Apple's on-device Foundation Model.

```python
import apple_fm_sdk as fm
from pydantic_ai import Agent

from apple_fm_model import AppleFoundationModel

agent = Agent(AppleFoundationModel(fm.SystemLanguageModel()))
print(agent.run_sync("Say hello.").output)
```

Tools, structured output, streaming, images and token usage work as with any Pydantic AI model. Nothing leaves your Mac.
