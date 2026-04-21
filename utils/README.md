# utils

`utils` contains general helper utilities shared across experiments and scripts.

## Modules

- `plots.py`: reusable plotting utilities for LLM training, pruning, and activation analysis
- `argparser.py`: helper to initialize dataclass instances from CLI arguments

## Plot Utilities

Available plot helpers:

- `plot_training_dashboard`
- `plot_token_length_distribution`
- `plot_layer_metric_heatmap`
- `plot_layer_sparsity`
- `plot_pruning_tradeoff`
- `plot_neuron_importance`
- `plot_weight_distribution`
- `plot_activation_distribution`

### Example

```python
import numpy as np
import pandas as pd
from utils import (
    plot_training_dashboard,
    plot_pruning_tradeoff,
    plot_neuron_importance,
)

history = pd.DataFrame(
    {
        "step": [1, 2, 3, 4],
        "train_loss": [3.4, 2.8, 2.4, 2.1],
        "eval_loss": [3.6, 3.0, 2.7, 2.5],
        "perplexity": [31.1, 24.7, 20.5, 17.8],
    }
)

plot_training_dashboard(history, save_path="./plots/dashboard.png")
plot_pruning_tradeoff(
    sparsity=[0.0, 0.2, 0.4, 0.6],
    metric_values=[17.8, 18.9, 21.2, 26.1],
    metric_name="Validation Perplexity",
    save_path="./plots/pruning_tradeoff.png",
)
plot_neuron_importance(np.random.rand(256), top_k=20, save_path="./plots/top_neurons.png")
```

## Dataclass Argparser Helper

`init_dataclass_from_args` creates a dataclass from CLI flags generated from dataclass fields.

### Example

```python
from dataclasses import dataclass
from utils.argparser import init_dataclass_from_args


@dataclass
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 1e-4
    model_name: str = "baseline"


cfg = init_dataclass_from_args(TrainConfig)
print(cfg)
```

Run example:

```bash
python train.py --epochs 10 --learning-rate 5e-5 --model-name llm-v2
```
