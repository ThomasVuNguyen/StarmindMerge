import argparse
import json
import os
import sys

import yaml

from tools.graphs import plot_model_performance
from tools.utils import Archive, parse_score, parse_task_list


def load_json_results(results_dir: str, models: list[str], tasks: list[dict]) -> dict[str, list[float]]:
    """Load performance data from JSON results files."""
    performance_data = {}
    
    # Load the comprehensive results file
    all_results_file = os.path.join(results_dir, "all_results.json")
    if os.path.exists(all_results_file):
        with open(all_results_file, "r") as f:
            all_results = json.load(f)
        
        for model_path in models:
            # Extract model name from path
            model_name = model_path.rsplit("/")[-1].rsplit(".", 1)[0].rsplit("-00001-of-", 1)[0]
            
            if model_name in all_results["models"]:
                model_data = all_results["models"][model_name]
                scores = []
                for task in tasks:
                    task_name = task['name']
                    if task_name in model_data["tasks"]:
                        scores.append(model_data["tasks"][task_name]["score_numeric"])
                    else:
                        scores.append(0.0)  # Default score if task not found
                performance_data[model_name] = scores
            else:
                print(f"Warning: No results found for model {model_name}")
                performance_data[model_name] = [0.0] * len(tasks)
    else:
        print(f"Warning: Results file {all_results_file} not found")
        # Fallback: try individual model files
        for model_path in models:
            model_name = model_path.rsplit("/")[-1].rsplit(".", 1)[0].rsplit("-00001-of-", 1)[0]
            model_file = os.path.join(results_dir, f"{model_name}_results.json")
            if os.path.exists(model_file):
                with open(model_file, "r") as f:
                    model_data = json.load(f)
                scores = []
                for task in tasks:
                    task_name = task['name']
                    if task_name in model_data["tasks"]:
                        scores.append(model_data["tasks"][task_name]["score_numeric"])
                    else:
                        scores.append(0.0)
                performance_data[model_name] = scores
            else:
                print(f"Warning: No results found for model {model_name}")
                performance_data[model_name] = [0.0] * len(tasks)
    
    return performance_data


def main():
    parser = argparse.ArgumentParser(description="Plot model performance.")
    parser.add_argument("models", nargs="+", help="Paths to GGUF model files to plot.")
    parser.add_argument("--overlay", action="store_true", help="Overlay all models (when providing multiple models) rather than using a grid layout.")
    parser.add_argument("--renderer", default="browser", help="How to render the graph. Can be one of: png, svg, notebook, browser. Default: browser.")
    parser.add_argument("--save_to", type=str, help="Save the resulting plot to disk as the given filename.")
    parser.add_argument("--normalization", type=str, default="none", help="Normalization: none, cap, range. none means use values as is (78% means 78% on the score). cap means anchor scores so that 100% is the best performing model (100% means best performing model even if the model score was 10% as long as no other model was higher). range means use the min and max values of the given models as the 0% and 100% point (0% means worst performing model, even if that model had a 99% score on the given task).")
    parser.add_argument("--use_pickle", action="store_true", help="Load results from pickle archive instead of JSON (JSON is default).")
    parser.add_argument("--results_dir", default="results", help="Directory containing JSON results (default: results).")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (suppress output).")

    with open("tasks.yml") as f:
        tasks = yaml.load(f, Loader=yaml.FullLoader)
    task_list = ",".join(task['name'].lower() for task in tasks)
    parser.add_argument("--tasks", default="all", help=f"Tasks to run. Can be a list of {task_list}, all, or a negative list such as -{task_list}, which runs all tasks except those listed.")

    args = parser.parse_args()

    tasks = parse_task_list(args.tasks, tasks)
    if len(tasks) == 0:
        print("No tasks. Great, we're done.")
        sys.exit(0)

    if args.use_pickle:
        # Load from pickle archive (fallback behavior)
        archives = Archive(".profile_archives.pk")

        model_names = {
            model: model.rsplit("/")[-1].rsplit(".", 1)[0].rsplit("-00001-of-", 1)[0] for model in args.models
        }

        performance_data: dict[str, list[float]] = {
            model_names[model]: [parse_score(archives[model][task['name']][0]) for task in tasks]
            for model in args.models
        }
    else:
        # Load from JSON results (default behavior)
        performance_data = load_json_results(args.results_dir, args.models, tasks)
        model_names = {
            model: model.rsplit("/")[-1].rsplit(".", 1)[0].rsplit("-00001-of-", 1)[0] for model in args.models
        }

    if args.normalization != "none":
        if args.normalization != "cap" and args.normalization != "range":
            raise ValueError("--normalization must be one of none, cap, or range")
        for i in range(len(tasks)):
            scores = [performance_data[model][i] for model in performance_data.keys()]
            max_score = max(scores)
            if max_score > 0:
                if args.normalization == "cap":
                    for model in performance_data.keys():
                        performance_data[model][i] *= 100 / max_score
                else:
                    min_score = min(scores)
                    for model in performance_data.keys():
                        v = performance_data[model][i]
                        performance_data[model][i] = 100 * (v - min_score) / (max_score - min_score)

    # Sort models by aggregated performance
    model_agg_scores = {
        model: sum(performance_data[model]) for model in performance_data.keys()
    }
    sorted_models = sorted(performance_data.keys(), key=lambda m: sum(performance_data[m]), reverse=True)
    for model in sorted_models:
        score = model_agg_scores[model]
        print(f"{model}: {score:.2f} ({', '.join(f'{task['name']}: {performance_data[model][i]:.2f}' for i, task in enumerate(tasks))})")

    mode = "overlay" if args.overlay else "grid"
    fig = plot_model_performance(
        models=sorted_models,
        tasks=[task['name'] for task in tasks],
        agg_scores=model_agg_scores,
        performance_data=performance_data, # type: ignore
        mode=mode,
        title="Model Performance",
        save_path=None,
    )

    if args.renderer == "png":
        import io

        from PIL import Image

        img_bytes = fig.to_image(format="png")
        img = Image.open(io.BytesIO(img_bytes))
        img.show()
    elif args.renderer == "browser":
        import webbrowser
        fig.write_html('plot.html')
        webbrowser.open('plot.html')

    else:
        fig.show()

if __name__ == "__main__":
    main()
