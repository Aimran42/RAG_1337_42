## RAG
# Moulinette



Evaluation system for RAG submissions. Validates your search results and calculates recall metrics.



## Installation 



```bash

uv venv && uv sync

direnv allow  # or source .envrc

```


## CLI Commands



### evaluate_student_search_results



Evaluate your search results against the ground truth dataset.



```bash

uv run python -m moulinette evaluate_student_search_results \

    <student_results_path> \

    <dataset_path> \

    [--k K] \

