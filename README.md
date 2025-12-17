# Phoenix to Arize Migration Tools

Tools to export data from Phoenix and import it into Arize AX.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   Create a `.env` file with:
   ```bash
   # Phoenix (include PHOENIX_API_KEY if using Phoenix Cloud)
   PHOENIX_ENDPOINT="https://app.phoenix.arize.com/s/your-space-name"
   PHOENIX_API_KEY=your-phoenix-api-key 
   
   # Arize
   ARIZE_API_KEY=your-arize-api-key
   ARIZE_SPACE_ID=your-arize-space-id

   # Export directory where Phoenix data will be saved locally
   PHOENIX_EXPORT_DIR="phoenix_export"
   ```

## Data Types & Commands

| Data Type | Export Command | Import Command | Warnings & Limitations |
|-----------|---------------|----------------|------------------------|
| **All Types** | `python export_all_projects.py --all` | `python import_to_arize.py --all` | • Follows import order: datasets-experiments → traces (with evaluations and annotations) |
| **Datasets & Experiments** | `python export_all_projects.py --de` | `python import_to_arize.py --de` | • Experiment evaluations not yet migrated |
| **Traces** | `python export_all_projects.py --traces` | `python import_to_arize.py --traces` | • Contains traces, evaluation, and annotations |

## Important Info for Annotations

- **Wait for traces to be indexed**: After importing traces, you must wait a few minutes for them to be loaded and indexed in Arize AX before sending annotation data. The import process will prompt you to verify traces are available before proceeding with annotations.

- **31-day window**: Only annotations for traces from the past 31 days can be logged to Arize. If your traces are older than 31 days, their annotations will be skipped with a warning message.

## Quick Start

### Export from Phoenix
```bash
# Export everything
python export_all_projects.py --all

# Export specific types
python export_all_projects.py --de      # datasets and experiments
python export_all_projects.py --traces   # traces with evaluations and annotations
```

### Import to Arize
```bash
# Import everything
python import_to_arize.py --all

# Import specific types
python import_to_arize.py --de      # datasets and experiments
python import_to_arize.py --traces  # traces with evaluations and annotations
```
## Generated Files

```
phoenix_export/ # All data lives here
├── datasets/
│   ├── datasets.json
│   ├── dataset_{id}_examples.json
│   └── dataset_{id}_experiments.json
└── projects/
    └── {project_name}/
        ├── project_metadata.json
        ├── traces.json
        ├── evaluations.json
        └── annotations.json

results/  # Overview of import and export jobs
├── dataset_experiment_export_results.json
├── dataset_experiment_import_results.json
├── trace_export_results.json
├── trace_import_results.json
```
