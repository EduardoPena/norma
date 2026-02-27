# Experiments Framework - Clean Structure

This is a clean, config-driven experiment framework for Text2Constraints research.

## 📁 Structure

```
experiments_new/
├── run_classifiers.py          # Run classifier experiments
├── run_mappers.py              # Run mapper experiments
├── run_pipelines.py            # Run end-to-end pipeline experiments
│
├── shared/
│   └── experiment_base.py      # Shared base class
│
└── configs/                    # All experiment configurations
    ├── classifiers/
    │   ├── llm_variants.yaml
    │   └── rag_variants.yaml
    ├── mappers/
    │   ├── prompt_variants.yaml
    │   └── batch_size_study.yaml
    └── pipelines/
        └── norma_best.yaml
```

## 🚀 Quick Start

### 1. Run Classifier Experiments

Compare different classifier implementations (LLM, RAG, SLM):

```bash
python run_classifiers.py configs/classifiers/llm_variants.yaml
python run_classifiers.py configs/classifiers/rag_variants.yaml
```

**What it does:**
- Tests binary classification (is this sentence a rule?)
- Evaluates on: Accuracy, Precision, Recall, F1
- Saves results to: `results/<experiment_name>/`

### 2. Run Mapper Experiments

Compare different mapper configurations (rules only):

```bash
python run_mappers.py configs/mappers/prompt_variants.yaml
python run_mappers.py configs/mappers/batch_size_study.yaml
```

**What it does:**
- Tests constraint extraction on rule sentences only
- Evaluates on: Detection F1, Strict Accuracy, Jaccard, DB Execution
- Saves results to: `results/<experiment_name>/`

### 3. Run Pipeline Experiments

Test end-to-end NORMA (classifier + mapper):

```bash
python run_pipelines.py configs/pipelines/norma_best.yaml
```

**What it does:**
- Tests full pipeline on all sentences (rules + non-rules)
- Evaluates both classification and mapping stages
- Saves results to: `results/<experiment_name>/`

## 📝 Creating Your Own Configs

### Classifier Config Example

```yaml
experiment_name: "My Classifier Test"
experiment_type: "classifier"

datasets:
  - restaurant-inspections
  - tax-chu

variants:
  - name: "GPT-5 few-shot"
    implementation: "llm"
    model: "gpt-5"
    prompt_key: "few_shot"
    batch_size: 10
    use_context: false
```

### Mapper Config Example

```yaml
experiment_name: "My Mapper Test"
experiment_type: "mapper"

datasets:
  - restaurant-inspections

variants:
  - name: "GPT-5 enhanced"
    model: "gpt-5"
    prompt_key: "enhanced_with_domain_v2"
    temperature: 0.0
    batch_size: 5
```

### Pipeline Config Example

```yaml
experiment_name: "My Pipeline Test"
experiment_type: "pipeline"

datasets:
  - restaurant-inspections

variants:
  - name: "NORMA GPT-5"
    classifier:
      implementation: "llm"
      model: "gpt-5"
      prompt_key: "few_shot"
      batch_size: 5
      use_context: true
    mapper:
      model: "gpt-5"
      prompt_key: "enhanced_with_domain_v2"
      temperature: 0.0
      batch_size: 5
```

## 📊 Results

Results are saved in `results/<experiment_name>/` with:
- CSV files with all metrics
- Timestamped filenames
- Copy of config file used

## 🔧 Adding New Variants

Just edit the YAML config files! No Python code changes needed.

**To add a new classifier:**
1. Edit `configs/classifiers/llm_variants.yaml`
2. Add a new entry under `variants:`
3. Run the experiment

**To add a new mapper:**
1. Edit `configs/mappers/prompt_variants.yaml`
2. Add a new entry under `variants:`
3. Run the experiment

## 💡 Tips

- **Start small**: Test on 1 dataset first
- **Comment out variants**: Use `#` to disable variants you don't want to run
- **Check results**: Results are saved after each variant completes
- **Multiple configs**: Create as many config files as you need

## 📋 Common Experiment Patterns

### Ablation Study (Model Size)
```yaml
variants:
  - name: "Small"
    model: "gpt-5-mini"
    ...
  - name: "Large"
    model: "gpt-5"
    ...
```

### Ablation Study (Prompts)
```yaml
variants:
  - name: "Simple"
    prompt_key: "simple"
    ...
  - name: "With Context"
    prompt_key: "with_context"
    ...
```

### Ablation Study (Batch Size)
```yaml
variants:
  - name: "Batch 1"
    batch_size: 1
  - name: "Batch 5"
    batch_size: 5
  - name: "Batch 10"
    batch_size: 10
```

## 🐛 Troubleshooting

**Import Error?**
- Make sure you're in the project root directory
- Check that `src/` directory exists

**Config Not Found?**
- Check the path is correct
- Use paths relative to where you run the script

**No Results?**
- Check `results/` directory
- Look for error messages in console

## 📖 Next Steps

1. Copy this to your `text2constraints/experiments/` directory
2. Update paths if needed (currently assumes `data/` at root)
3. Create your own config files
4. Run experiments!
