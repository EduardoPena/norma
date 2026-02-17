# REDSQL-Inspired Multi-Step Constraint Mapper

Inspired by the VLDB 2025 paper: **"The Power of Constraints in Natural Language to SQL Translation"** by Ren et al.

## Overview

This implementation adapts REDSQL's multi-step refinement approach from NL2SQL to our denial constraint extraction domain. The key insight from REDSQL is that **validating predictions against the actual database content** allows for much more accurate refinement than relying solely on LLM reasoning.

## Key Concepts from REDSQL

### 1. **Multi-Step Refinement Pipeline**
Rather than one-shot generation, REDSQL breaks down the task into stages:
- Understanding the schema
- Selecting relevant operations
- Constructing initial output
- Verifying against database
- Refining based on violations

### 2. **Constraints Verification** 
REDSQL's core innovation: apply constraints to query operations and validate against the **entire database**, not just a sample. This catches semantic errors that LLMs miss without full data access.

### 3. **Context Expansion**
Provide LLMs with:
- **Schema information** (what tables/columns exist)
- **Data profiling** (types, distributions, sample values)
- **Validation results** (what violations were detected)

### 4. **Documentation/Profiling**
Build a "global data context" by profiling columns offline (type, nullability, samples, statistics). This enriches the LLM's understanding without overwhelming the prompt.

### 5. **Iterative Refinement**
Multiple refinement rounds (REDSQL uses "AB strategy": All refinement, then Bug-only refinement). Each iteration uses validation feedback to improve.

## Our Implementation

We created two mappers inspired by REDSQL:

### `REDInspiredMapper` - Full Multi-Step Pipeline

Implements the complete REDSQL-inspired workflow:

**Step 1: Classification**
- Is this sentence a data constraint?
- Can decide "not a rule" early

**Step 2: Schema Understanding**
- Identify relevant columns
- Verify columns exist in schema
- Reject if columns don't exist

**Step 3: Predicate Selection**
- Filter predicates relevant to identified columns
- Select 2-8 predicates needed
- Validate against allowed predicate set

**Step 4: Constraint Construction**
- Build initial denial constraint
- Use LLM to generate predicate list

**Step 5: Verification**
- Translate constraint to SQL
- Execute on database sample
- Check for errors and violation counts

**Step 6: Refinement** (iterative)
- If validation fails, refine constraint
- Use error messages to guide fixes
- Repeat up to max_iterations

```python
from src.mappers.redinspered_llm import REDInspiredMapper
from src.data.loader import DatasetLoader

# Load dataset (predicate_space is built automatically)
loader = DatasetLoader('data/hospital')
df, sentences, sentence2dcs_map, predicate_space = loader.load()

mapper = REDInspiredMapper(
    model_name='gpt-5-mini',
    predicate_space=predicate_space,
    df=df,
    max_refinement_iterations=2
)

results = mapper.map(sentences)
```

### `DirectRefinementMapper` - Simplified Version

Streamlined approach focusing on core refinement loop:
1. Generate initial constraint
2. Verify on database
3. Refine if errors
4. Repeat

```python
from src.mappers.redinspered_llm import DirectRefinementMapper
from src.data.loader import DatasetLoader

# Load dataset (predicate_space is built automatically)
loader = DatasetLoader('data/restaurant')
df, sentences, sentence2dcs_map, predicate_space = loader.load()

mapper = DirectRefinementMapper(
    model_name='gpt-5-mini',
    predicate_space=predicate_space,
    df=df,
    max_iterations=2
)

results = mapper.map(sentences)
```

## Comparison with REDSQL

| REDSQL (NL2SQL) | Our Implementation (Denial Constraints) |
|-----------------|----------------------------------------|
| Refine SQL queries | Refine denial constraints |
| Check SQL execution errors | Check constraint validity on data |
| Verify table/column names | Verify predicate existence |
| Test on database content | Test constraint violations |
| Schema expansion | Column profile enrichment |
| SQL validation | Predicate list validation |

## Key Differences from Agent Mapper

Our [agent.py](./agent.py) uses LangGraph with tools, while this REDSQL-inspired mapper:

1. **Structured Pipeline**: Fixed steps vs. flexible agent workflow
2. **LLM Guidance**: Each step has specific prompt vs. agent decides tool use
3. **Validation Focus**: Emphasizes database validation over exploration
4. **Simpler**: No LangGraph dependency, easier to understand
5. **Faster**: Fewer LLM calls for straightforward cases

## When to Use Each Mapper

**Use REDInspiredMapper when:**
- Need systematic validation at each step
- Want explicit schema/predicate checking
- Debugging: see exactly where pipeline fails
- Complex constraints needing careful construction

**Use DirectRefinementMapper when:**
- Want faster processing
- Constraints are relatively simple
- Trust LLM to generate reasonable initial attempts
- Primarily need validation and refinement

**Use Agent Mapper when:**
- Constraints are very complex (6+ predicates)
- Need exploratory data analysis
- Want autonomous tool selection
- Maximum flexibility needed

## Testing

Run the test script to see all mappers in action:

```bash
# Test full multi-step pipeline
python test_redinspered.py full

# Test simplified direct refinement
python test_redinspered.py direct

# Compare both approaches
python test_redinspered.py compare

# Run all tests
python test_redinspered.py
```

## Implementation Details

### Column Profiling (REDSQL's Documentation)

```python
def _build_column_profiles(self) -> Dict[str, Dict[str, Any]]:
    """Build profiles for each column (type, nulls, samples, stats)"""
    # For each column: type, null_count, unique_count, sample_values
    # For numeric: min, max, mean
    # This is the "global data context"
```

### Schema Validation (REDSQL's Schema Expansion)

```python
def _step2_understand_schema(self, sentence: str):
    """Verify mentioned columns actually exist"""
    # LLM identifies relevant columns
    # Validate against actual schema
    # Reject if columns don't exist
```

### Constraint Verification (REDSQL's Core Innovation)

```python
def _step5_verify_constraint(self, constraint: List[str]):
    """Test constraint on database"""
    # Translate to SQL
    # Execute violation query
    # Check error or violation count
    # Heuristic: if >50% violations, likely wrong
```

### Iterative Refinement (REDSQL's Refinement Strategy)

```python
for iteration in range(self.max_refinement_iterations):
    verification = self._step5_verify_constraint(constraint)
    if verification['is_valid']:
        return constraint
    constraint = self._step6_refine_constraint(constraint, verification)
```

## Future Enhancements

From REDSQL paper, we could add:

1. **Workload-Specific Constraints**: Custom validation rules per domain
2. **Violation-Level Classification**: INFO/WARNING/ERROR like REDSQL
3. **Execution Efficiency**: Track constraint performance
4. **Demonstration Selection**: Few-shot with relevant examples
5. **Model Ensemble**: Combine multiple LLMs like REDSQL shows

## Citation

If using ideas from the REDSQL paper:

```
@article{ren2025redsql,
  title={The Power of Constraints in Natural Language to SQL Translation},
  author={Ren, Tonghui and Ke, Chen and Fan, Yuankai and Jing, Yinan and He, Zhenying and Zhang, Kai and Wang, X. Sean},
  journal={PVLDB},
  volume={18},
  number={7},
  pages={2097--2111},
  year={2025}
}
```

## Summary

This implementation brings REDSQL's key insight to denial constraints: **validate with the database, not just LLM reasoning**. By checking constraints against actual data and iteratively refining based on validation results, we can catch errors that pure LLM approaches miss.

The multi-step pipeline provides transparency and control, while the simplified version offers speed. Both significantly improve over single-shot mapping by leveraging the power of constraints verification on database content.
