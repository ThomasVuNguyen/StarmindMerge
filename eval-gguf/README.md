# gguf-eval

An evaluation framework for GGUF models using llama.cpp. Based of of https://github.com/kallewoof/gguf-eval

## Quick Start

**Just run the script and everything else is automatic!**

```bash
# Run all models in gguf/ folder on all tasks (default)
python eval.py

# Evaluate specific model(s)
python eval.py model.gguf

# Compare multiple models
python eval.py model1.gguf model2.gguf

# Run specific tasks only
python eval.py --tasks mmlu,hellaswag

# Evaluate and generate plot
python eval.py --plot
```

## Ultra-Simple Usage

The absolute simplest way to use this tool:

1. **Put your models in the `gguf/` folder**
2. **Run `python eval.py`**

That's it! The script will automatically:
- Find all GGUF files in the `gguf/` folder
- Run all benchmark tasks on all models
- Save results to `results/` directory
- Show you the final report

## What's Automatic

- ✅ **Model discovery**: Automatically finds all GGUF files in `gguf/` folder
- ✅ **Task selection**: Runs all available tasks by default
- ✅ **llama.cpp detection**: Automatically finds `./llama.cpp`
- ✅ **JSON results**: Automatically saved to `results/` directory
- ✅ **Plotting**: Uses JSON results by default

## JSON Results Format

Results are automatically saved in structured JSON format for easy analysis and sharing:

### Individual Model Files (`{model_name}_results.json`)
```json
{
  "model_path": "gguf/model-name.gguf",
  "model_name": "model-name",
  "tasks": {
    "MMLU": {
      "score": "78.5%",
      "execution_time_seconds": 120,
      "score_numeric": 78.5
    },
    "Hellaswag": {
      "score": "82.3%",
      "execution_time_seconds": 95,
      "score_numeric": 82.3
    }
  }
}
```

### Comprehensive Results File (`all_results.json`)
```json
{
  "metadata": {
    "timestamp": "2024-01-15T10:30:45.123456",
    "total_models": 2,
    "version": "1.0"
  },
  "models": {
    "model-name-1": {
      "model_path": "gguf/model-name-1.gguf",
      "model_name": "model-name-1",
      "tasks": {
        "MMLU": {
          "score": "78.5%",
          "execution_time_seconds": 120,
          "score_numeric": 78.5
        }
      }
    }
  }
}
```

## Advanced Usage

For more control, use the original scripts directly:

```bash
# Direct evaluation with custom options
python evaluate.py model.gguf --tasks mmlu --llama_path ./llama.cpp

# Plot with custom options  
python plot.py model.gguf --tasks mmlu

# Disable automatic JSON export
python evaluate.py model.gguf --no_json

# Use pickle archive instead of JSON
python plot.py model.gguf --use_pickle
```

## Analyzing Results

The JSON format makes it easy to analyze results programmatically:

```bash
# Quick score comparison
python -c "
import json
with open('results/all_results.json', 'r') as f:
    data = json.load(f)
    for model, results in data['models'].items():
        print(f'{model}: {results[\"tasks\"][\"MMLU\"][\"score_numeric\"]}%')
"

# Export to CSV
python -c "
import json
import csv
with open('results/all_results.json', 'r') as f:
    data = json.load(f)
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'MMLU', 'Hellaswag', 'Winogrande'])
    for model, results in data['models'].items():
        tasks = results['tasks']
        writer.writerow([
            model,
            tasks.get('MMLU', {}).get('score_numeric', 0),
            tasks.get('Hellaswag', {}).get('score_numeric', 0),
            tasks.get('Winogrande', {}).get('score_numeric', 0)
        ])
"
```

## Available Tasks

The following benchmark tasks are available:

- **MMLU** (default): Massive Multitask Language Understanding
- **Hellaswag**: Commonsense reasoning
- **Winogrande**: Pronoun resolution
- **TruthfulQA**: Truthfulness in question answering
- **ARC-Combined**: AI2 Reasoning Challenge (combined)
- **ARC-Challenge**: AI2 Reasoning Challenge (challenge set)

## Getting Started

1. Install llama.cpp in the same directory as this script.
2. Clone this repository: `git clone https://github.com/kallewoof/gguf-eval.git`
3. Change dir: `cd gguf-eval`
4. Install requirements: `pip install -r requirements.txt`
5. Place your GGUF model files in the `gguf/` folder.
6. Run evaluation: `python eval.py` (that's it!)

The script will automatically:
- Find all GGUF files in the `gguf/` folder
- Run all available benchmark tasks
- Save results to `results/` directory in JSON format

### Details

* If you are e.g. on Windows and you're not running WSL, you may need to pass `--disable_ansi`.
* Unless you installed llama.cpp so it is available from the shell, you need to pass a --llama_path to the llama.cpp directory: `python evaluate.py --llama_path ../llama.cpp model1.gguf model2.gguf ...`
* If you need to pass arguments to llama.cpp for all models you can use `--llama_args "\--arg1=x"`.
* If you need to pass arguments to llama.cpp for a specific model in your list only, you can use `--model_args`: example: `python evaluate.py llama-x.gguf nvidia-nemotron-49b.gguf GLM-4.5-Air.gguf --model_args emotron:"-ts 10/18" --model_args GLM-4.5-Air:"--n-cpu-moe 22 -ts 24/10"`
* You can select or exclude tasks using the `--tasks` argument: `python evaluate.py ... --tasks exclude:mmlu,hellaswag`

## Render Plot

After running evaluate.py on some tasks, you can plot these. The results look something like this (for `--overlay` mode):

![Plot example](docs/images/plot-examples.png "Plot example")

1. You need plotly: `pip install plotly`
2. Run `python plot.py model1.gguf model2.gguf ...`.

### Details

* You can use `--overlay` to display all models in one graph, overlayed. The default is to show each one separately in a grid.
* You can normalize the scores using `--normalization`. There are two modes, `cap` and `range`. `cap` means the models are all normalized so that the best performing model gets a 100% score, and the other models proportionately to that. E.g. if the model scores are 0.1, 0.2, and 0.3, this will be displayed as 33, 66, and 100% respectively. `range` means the models are normalized so that 0% is the worst performing model and 100% is the best performing model. The previous case would display as 0%, 50%, and 100%. The default is `none`.
* The default behavior is to generate a html file and open in your browser. You can instead use e.g. `--renderer=png` to output to a png file, although the quality of this is not great at the moment.

### Plot explanation

In the overlay mode, each model is prefixed with a number. This is the *sum* of the scores for that model, for all tasks. Models are also sorted by the scores in both grid and overlay mode.

## Troubleshooting

* `error: invalid argument: -kvu`: update your llama.cpp installation.
