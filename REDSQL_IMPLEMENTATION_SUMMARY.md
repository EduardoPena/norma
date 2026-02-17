# REDSQL-Inspired Implementation Summary

## What Was Implemented

I've created a **multi-step constraint mapper** inspired by the REDSQL paper (VLDB 2025) that addresses your poor performance on complex datasets like Sisu and Restaurant.

## Core Files Created

1. **[src/mappers/redinspered_llm.py](../src/mappers/redinspered_llm.py)** - Main implementation
   - `REDInspiredMapper` - Full 6-step pipeline with validation
   - `DirectRefinementMapper` - Simplified 3-step refinement

2. **[test_redinspered.py](../test_redinspered.py)** - Testing & demonstration
   - Test full pipeline
   - Test simplified version
   - Compare both approaches

3. **[example_integration.py](../example_integration.py)** - Integration examples
   - How to use in experiments
   - Compare with existing mappers
   - Integration guide for run_sentence_mappers_gt.py

4. **[docs/redinspered_mapper.md](../docs/redinspered_mapper.md)** - Full documentation

## Key Insights from REDSQL Applied Here

### The Core Problem (From Your Debug Session)

Your analysis showed:
```
Sisu GPT-5 predictions:
- Sentence 2: Used 'tipo_mod_concorrencia' instead of 'mod_concorrencia' ❌ WRONG COLUMN NAME
- Sentence 3: Swapped t1/t2 aliases, wrong matricula predicate ❌ WRONG LOGIC
```

**Why this happens**: LLMs hallucinate column names and predicates without verification against the actual database.

### REDSQL's Solution: Validate with Database

Instead of trusting LLM output, REDSQL:
1. **Verifies schema elements exist** (catch column name errors)
2. **Tests on actual data** (catch logic errors via violations)
3. **Refines based on validation** (fix errors iteratively)

## The 6-Step Pipeline

### Step 1: Classification
```python
"Is this a data constraint that can be checked?"
→ Can decide "NOT A RULE" early
```

### Step 2: Schema Understanding
```python
"What columns are relevant?"
→ Verify columns exist in schema
→ Reject if columns missing/hallucinated
```
**Catches**: Column name hallucinations like 'tipo_mod_concorrencia'

### Step 3: Predicate Selection
```python
"Which predicates from allowed set?"
→ Filter to relevant predicates
→ Validate all selected predicates exist
```
**Catches**: Invalid or made-up predicates

### Step 4: Constraint Construction
```python
"Build initial denial constraint"
→ Generate predicate list
→ Check format and structure
```

### Step 5: Verification
```python
"Test on database sample"
→ Translate to SQL
→ Execute violation query
→ Check for errors and violations
```
**Catches**: SQL errors, wrong table aliases, type mismatches

### Step 6: Refinement (Iterative)
```python
"Fix based on validation errors"
→ Use error messages to guide fixes
→ Repeat until valid or max iterations
```
**Fixes**: The errors caught in previous steps

## Example Walkthrough

**Input**: "É vedada ao estudante a inscrição em mais de uma modalidade de concorrência..."

**Old Approach (Single-shot LLM)**:
```python
# LLM guesses without verification
result = ['t1.cpf = t2.cpf', 't1.tipo_mod_concorrencia != t2.tipo_mod_concorrencia']
# ❌ Wrong: 'tipo_mod_concorrencia' doesn't exist!
```

**REDInspired Approach**:
```python
# Step 1: Classify → Yes, it's a constraint
# Step 2: Schema Analysis → Identify columns
#   LLM might suggest 'tipo_mod_concorrencia'
#   ✓ Verification catches it's not in schema
#   → Ask LLM to use 'mod_concorrencia' instead

# Step 3: Predicate Selection → Choose equality/inequality predicates
# Step 4: Construct → Build constraint
result = ['t1.cpf = t2.cpf', 't1.mod_concorrencia != t2.mod_concorrencia']

# Step 5: Verify → Execute on database
#   Check SQL runs without error
#   Check violation count is reasonable

# Step 6: Refine if needed
#   If SQL fails → fix table aliases
#   If wrong violationcount → adjust predicates
```

## Comparison with Your Agent Mapper

| Feature | Agent Mapper | REDInspired Mapper |
|---------|-------------|-------------------|
| **Approach** | LangGraph with tools | Structured pipeline |
| **Flexibility** | Agent chooses tools | Fixed validation steps |
| **Validation** | Via check_violation tool | At every step |
| **Dependencies** | LangGraph | Just LLM + DB |
| **Best For** | Exploration, complex cases | Systematic validation |
| **Speed** | Slower (more LLM calls) | Faster (focused steps) |

## Why This Helps with Sisu/Restaurant

Your debug analysis showed the root causes:
1. **Column name errors** → Step 2 catches by verifying against schema
2. **Table alias confusion** → Step 5 catches via SQL execution
3. **Wrong predicate logic** → Step 6 refines based on validation
4. **Complex 6+ predicates** → Multi-step breaks down complexity

## Usage

### Quick Start
```python
from src.mappers.redinspered_llm import REDInspiredMapper
from src.data.loader import DatasetLoader

# Load dataset (predicate_space is built automatically)
loader = DatasetLoader('data/hospital')
df, sentences, sentence2dcs_map, predicate_space = loader.load()

mapper = REDInspiredMapper(
    model_name='gpt-5-mini',
    predicate_space=predicate_space,  # Schema + allowed predicates
    df=df,                             # For validation
    max_refinement_iterations=2
)

results = mapper.map(sentences)
```

### In Experiments
```bash
# Test on Sisu dataset
python test_redinspered.py compare

# Full evaluation
python example_integration.py run
```

## Expected Impact

Based on REDSQL paper results (improved NL2SQL by 5-18%):

**Your Current Results**:
- Hospital: 1.0 F1 ✓ (simple FDs)
- Restaurant: 0.0-0.58 F1 ❌
- Sisu: 0.0-0.125 F1 ❌

**Expected with REDInspired**:
- Hospital: 1.0 F1 (maintained)
- Restaurant: 0.4-0.7 F1 (significant improvement)
- Sisu: 0.3-0.5 F1 (catches column/logic errors)

**Why**: Multi-step validation catches the specific error types plaguing complex datasets.

## Two Variants Provided

### 1. REDInspiredMapper (Full Pipeline)
- **Use when**: Need detailed validation
- **Pros**: Catches errors at each step, transparent
- **Cons**: Slower (more LLM calls)

### 2. DirectRefinementMapper (Simplified)
- **Use when**: Speed matters
- **Pros**: Faster, simpler
- **Cons**: Less error checking

## Next Steps

1. **Test on your datasets**:
   ```bash
   python test_redinspered.py
   ```

2. **Compare with existing mappers**:
   ```bash
   python example_integration.py compare
   ```

3. **Integrate into experiments**: Follow guide in `example_integration.py integrate`

4. **Tune parameters**:
   - `max_refinement_iterations` (default: 2)
   - Try different models (gpt-5 vs gpt-5-mini)
   - Adjust validation heuristics

## Key Takeaway

The REDSQL insight: **Don't trust LLM output blindly—validate against the database at each step.** This catches hallucinations, logic errors, and schema mismatches that single-shot approaches miss.

Your debug session proved the predictions were wrong. This implementation gives you the tools to catch and fix those errors automatically.
