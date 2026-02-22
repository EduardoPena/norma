#!/usr/bin/env python3
"""
Multi-Step Constraint Refinement Mapper

Inspired by "The Power of Constraints in Natural Language to SQL Translation" (VLDB 2025)
This mapper uses a multi-step refinement process:
1. Schema Understanding: Identify relevant columns and verify they exist
2. Predicate Selection: Choose predicates from allowed set and validate
3. Constraint Construction: Build initial denial constraint
4. Verification: Test on database sample
5. Refinement: Adjust based on validation results

At any point, can decide the sentence is NOT a constraint.

Usage:
    from src.mappers.multi_step_refinement import MultiStepRefinementMapper
    from src.data.loader import DatasetLoader
    
    # Load dataset (predicate_space is built automatically)
    loader = DatasetLoader('data/hospital')
    df, sentences, sentence2dcs_map, predicate_space = loader.load()
    
    mapper = MultiStepRefinementMapper(
        model_name='gpt-4o-mini',
        predicate_space=predicate_space,
        df=df,
        max_refinement_iterations=2
    )
    
    results = mapper.map(sentences)
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from src.mappers.base import BaseMapper
from src.llm.client import get_llm_client
from src.llm.helper import PromptLoader
from src.dbms.db_interface import DBExecutor
from src.dbms.sql_translator import SQLTranslator


# ============================================================================
# Shared Utilities
# ============================================================================

def extract_json(text: str) -> str:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)

    pattern = r'\{.*\}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)

    return text


def token_similarity(s1: str, s2: str) -> float:
    """
    Compute similarity between two column names using token-based overlap.
    Splits on underscores so that days_to_next_inspection and days_to_action
    score more meaningfully than character set overlap.
    """
    t1 = set(s1.lower().split('_'))
    t2 = set(s2.lower().split('_'))
    if not t1 or not t2:
        return 0.0
    if s1.lower() == s2.lower():
        return 1.0
    if s1.lower() in s2.lower() or s2.lower() in s1.lower():
        return 0.8
    return len(t1 & t2) / len(t1 | t2)


# ============================================================================
# Multi-Step Refinement Mapper
# ============================================================================

class MultiStepRefinementMapper(BaseMapper):
    """
    Multi-step constraint mapper with iterative refinement.
    
    Key features:
    - Multi-step refinement with validation at each stage
    - Schema expansion and verification
    - Constraint verification on actual data
    - Context enrichment with data profiling
    - Iterative refinement based on violations
    """
    
    def __init__(self,
                 model_name: str = "gpt-5-mini",
                 predicate_space: Optional[Dict[str, Any]] = None,
                 df: Optional[pd.DataFrame] = None,
                 max_refinement_iterations: int = 2,
                 violation_ratio_threshold: float = 0.5,
                 prompt_file: str = 'multi_step_refinement_less_restrict',
                 chunk_size: Optional[int] = None):
        super().__init__(chunk_size=chunk_size)
        self.llm = get_llm_client(model_name)
        self.prompt_loader = PromptLoader()
        self.prompt_file = prompt_file
        self.predicate_space = predicate_space or {}
        self.df = df
        self.max_refinement_iterations = max_refinement_iterations
        self.violation_ratio_threshold = violation_ratio_threshold

        self.db_executor = None
        self.sql_translator = None
        if df is not None:
            self.db_executor = DBExecutor(df, table_name="data_table", tid_column="tid")
            self.sql_translator = SQLTranslator(self.db_executor)
        
        self.tables = predicate_space.get('tables', [])
        self.allowed_predicates = set(predicate_space.get('predicates', []))
        self.column_profiles = self._build_column_profiles()
        self.columns_with_predicates = self._extract_predicate_columns()

    def _extract_predicate_columns(self) -> set:
        """Extract unique column names that appear in the predicate space."""
        cols = set()
        for pred in self.allowed_predicates:
            matches = re.findall(r't[12]\.(\w+)', pred)
            cols.update(matches)
        return cols

    def _build_column_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Build column profiles with sample values and stats."""
        profiles = {}
        
        if self.df is None or self.df.empty:
            return profiles
        
        for col in self.df.columns:
            if col == 'tid':
                continue
            
            col_data = self.df[col].dropna()
            unique_vals = col_data.unique()
            n_unique = len(unique_vals)

            if n_unique <= 10:
                sample_values = unique_vals.tolist()
            else:
                top_frequent = col_data.value_counts().head(5).index.tolist()
                remaining = [v for v in unique_vals if v not in set(top_frequent)]
                n_random = min(5, len(remaining))
                random_sample = (
                    pd.Series(remaining).sample(n=n_random, random_state=42).tolist()
                    if n_random > 0 else []
                )
                sample_values = top_frequent + random_sample

            profile = {
                'type': str(self.df[col].dtype),
                'null_count': self.df[col].isnull().sum(),
                'unique_count': self.df[col].nunique(),
                'sample_values': sample_values
            }
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                profile.update({
                    'min': float(self.df[col].min()) if not col_data.empty else None,
                    'max': float(self.df[col].max()) if not col_data.empty else None,
                    'mean': float(self.df[col].mean()) if not col_data.empty else None
                })
            
            profiles[col] = profile
        
        return profiles

    def _format_column_profiles_for(self, columns: List[str]) -> str:
        """Format profiles only for relevant columns to avoid token waste."""
        if not self.column_profiles or not columns:
            return ""
        
        lines = ["\nColumn profiles (use these to verify string constants and casing):"]
        for col in columns:
            if col in self.column_profiles:
                profile = self.column_profiles[col]
                samples = profile.get('sample_values', [])
                samples_str = ', '.join([
                    f"'{s}'" if isinstance(s, str) else str(s)
                    for s in samples[:8]
                ])
                null_info = f", {profile['null_count']} nulls" if profile['null_count'] > 0 else ""
                lines.append(f"  - {col} ({profile['type']}{null_info}): [{samples_str}]")
        
        return '\n'.join(lines) if len(lines) > 1 else ""

    def _format_column_profiles(self) -> str:
        """Format all column profiles for schema understanding step."""
        if not self.column_profiles:
            return ""
        
        lines = ["\nColumn profiles (sample data context):"]
        for col, profile in self.column_profiles.items():
            samples = profile.get('sample_values', [])
            samples_str = ', '.join([
                f"'{s}'" if isinstance(s, str) else str(s)
                for s in samples[:3]
            ])
            null_info = f", {profile['null_count']} nulls" if profile['null_count'] > 0 else ""
            lines.append(f"  - {col} ({profile['type']}{null_info}): [{samples_str}]")
        
        return '\n'.join(lines)

    def _map_sentences(self,
                       sentences: List[str],
                       context: Dict[str, Any]) -> List[Optional[List[str]]]:
        """Process sentences through multi-step refinement pipeline."""
        results = []
        for sentence in sentences:
            try:
                constraint = self._process_single_sentence(sentence, context)
                results.append(constraint)
            except Exception as e:
                print(f"Error processing sentence: {e}")
                results.append(None)
        return results

    def _process_single_sentence(self,
                              sentence: str,
                              context: Dict[str, Any]) -> Optional[List[str]]:
        print(f"\n{'='*60}")
        print(f"Processing: {sentence[:80]}...")
        
        # Step 1: Schema Understanding (schema-grounded gate)
        schema_analysis = self._step2_understand_schema(sentence)
        if not schema_analysis['is_valid']:
            print(f"-> Schema validation failed: {schema_analysis.get('reason', 'unknown')}")
            return None
        
        # Step 2: Predicate Selection
        predicate_selection = self._step3_select_predicates(sentence, schema_analysis)
        if not predicate_selection['is_valid']:
            print(f"-> Predicate selection failed: {predicate_selection['reason']}")
            return None
        
        # Step 3: Initial Construction
        initial_constraint = self._step4_construct_constraint(
            sentence, schema_analysis, predicate_selection
        )
        if not initial_constraint:
            print("-> Could not construct initial constraint")
            return None
        
        # Steps 4 & 5: Verification and Refinement Loop
        constraint = initial_constraint
        for iteration in range(self.max_refinement_iterations):
            print(f"\n-> Verification iteration {iteration + 1}/{self.max_refinement_iterations}")
            
            verification = self._step5_verify_constraint(constraint, sentence)
            
            if verification['is_valid']:
                print(f"-> Verification successful")
                return constraint
            
            print(f"-> Refining - {verification['message']}")
            refined = self._step6_refine_constraint(
                constraint, sentence, verification, schema_analysis, predicate_selection
            )
            
            if refined == constraint:
                print("-> No refinement possible")
                return constraint
            
            constraint = refined
        
        print(f"-> Max iterations reached")
        return constraint

    def _step2_understand_schema(self, sentence: str) -> Dict[str, Any]:
        """Step 2: Understand relevant columns and verify they exist.
        
        Also acts as schema-grounded gate: rejects if no sentence concepts
        map to columns that have predicates defined.
        """
        schema_desc = self._format_schema_for_llm()
        column_profiles_desc = self._format_column_profiles()
        predicate_columns = '\n'.join([f"  - {c}" for c in sorted(self.columns_with_predicates)])

        prompt_template = self.prompt_loader.load(self.prompt_file, 'understand_schema')
        prompt = prompt_template.format(
            sentence=sentence,
            schema_desc=schema_desc,
            column_profiles_desc=column_profiles_desc,
            predicate_columns=predicate_columns
        )

        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(extract_json(response))

            # Merge direct + implied into relevant_columns
            direct = parsed.get('direct_columns', [])
            implied = parsed.get('implied_columns', [])
            parsed['relevant_columns'] = list(dict.fromkeys(direct + implied))

            # Validate all columns actually exist in schema
            relevant = parsed['relevant_columns']
            available_columns = {
                col for table in self.tables
                for col in [c['name'] for c in table['columns']]
            }
            available_list = list(available_columns)

            missing = [col for col in relevant if col not in available_columns]
            if missing:
                suggestions = {}
                for missing_col in missing:
                    similar = self._find_similar_columns(missing_col, available_list, top_n=3)
                    suggestions[missing_col] = [col for col, score in similar if score > 0.3]

                parsed['is_valid'] = False
                parsed['reason'] = f"Columns not found in schema: {missing}"
                if any(suggestions.values()):
                    parsed['suggestions'] = suggestions
                    print(f"-> Step 2: Columns not found: {missing}")
                    for col, sug in suggestions.items():
                        if sug:
                            print(f"   Suggestions: {sug}")

            if parsed.get('is_valid'):
                print(f"-> Step 2: Identified columns {relevant}")
            elif 'suggestions' not in parsed:
                print(f"-> Step 2 failed: {parsed.get('reason', 'unknown')}")

            return parsed

        except Exception as e:
            print(f"-> Step 2 parsing error: {e}")
            return {'is_valid': False, 'reason': str(e), 'relevant_columns': []}

    def _step3_select_predicates(self,
                                  sentence: str,
                                  schema_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Select predicates from allowed set."""
        relevant_columns = schema_analysis.get('relevant_columns', [])
        
        candidate_predicates = self._filter_predicates_by_columns(relevant_columns)
        
        if not candidate_predicates:
            return {
                'is_valid': False,
                'reason': 'No valid predicates found for these columns',
                'selected_predicates': []
            }
        
        predicates_desc = '\n'.join([f"  - {p}" for p in sorted(candidate_predicates)])
        column_profiles_desc = self._format_column_profiles_for(relevant_columns)
        
        prompt_template = self.prompt_loader.load(self.prompt_file, 'select_predicates')
        prompt = prompt_template.format(
            sentence=sentence,
            relevant_columns=', '.join(relevant_columns),
            predicates_desc=predicates_desc,
            column_profiles_desc=column_profiles_desc
        )
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(extract_json(response))
            
            selected = parsed.get('selected_predicates', [])
            invalid_preds = [p for p in selected if p not in self.allowed_predicates]
            
            if invalid_preds:
                parsed['is_valid'] = False
                parsed['reason'] = f"Invalid predicates (not in allowed set): {invalid_preds}"
            
            if parsed.get('is_valid'):
                print(f"-> Step 3: Selected {len(selected)} predicates: {selected}")
            else:
                print(f"-> Step 3 failed: {parsed.get('reason', 'unknown')}")
            
            return parsed
        except Exception as e:
            return {'is_valid': False, 'reason': f'Parse error: {e}', 'selected_predicates': []}

    def _step4_construct_constraint(self,
                                     sentence: str,
                                     schema_analysis: Dict[str, Any],
                                     predicate_selection: Dict[str, Any]) -> Optional[List[str]]:
        """Step 4: Construct initial denial constraint."""
        relevant_columns = schema_analysis.get('relevant_columns', [])
        selected_predicates = predicate_selection.get('selected_predicates', [])
        profiles_desc = self._format_column_profiles_for(relevant_columns)

        print(f"-> Step 4: Constructing constraint with {len(selected_predicates)} predicates and {len(relevant_columns)} columns")
        
        prompt_template = self.prompt_loader.load(self.prompt_file, 'construct_constraint')
        prompt = prompt_template.format(
            sentence=sentence,
            relevant_columns=', '.join(relevant_columns),
            selected_predicates=', '.join(selected_predicates),
            column_profiles_desc=profiles_desc
        )
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(extract_json(response))
            constraint = parsed.get('constraint')
            constraint_type = parsed.get('constraint_type', 'unknown')
            
            if constraint and isinstance(constraint, list) and len(constraint) > 0:
                print(f"-> Step 4: Constructed {constraint_type} with {len(constraint)} predicates")
                print(f"   {constraint}")
                print(f"   {parsed.get('constraint_interpretation', '')}")
                return constraint
            
            print(f"-> Step 4 failed: {parsed.get('reasoning', 'no reason given')}")
            return None
        except Exception as e:
            print(f"-> Step 4 error: {e}")
            return None

    def _step5_verify_constraint(self,
                                  constraint: List[str],
                                  sentence: str) -> Dict[str, Any]:
        """Step 5: Verify constraint on database sample."""
        if not self.sql_translator:
            return {'is_valid': True, 'message': 'No database available for verification'}
        
        dead = self._check_predicate_satisfiability(constraint)
        if dead:
            print(f"-> Step 5: Found invented constants (dead predicates): {dead}")
            return {
                'is_valid': False,
                'message': f"Invented constant — no rows satisfy: {dead}",
                'dead_predicates': dead,
                'constraint': constraint,
                'is_dead_predicate': True   # flag para NÃO tentar refinar
            }
        
        try:
            result = self.sql_translator.execute_violation(constraint)
            
            if not result['success']:
                print(f"-> Step 5: SQL execution failed - {result.get('error', 'unknown')}")
                return {
                    'is_valid': False,
                    'message': f"SQL execution failed: {result.get('error', 'unknown')}",
                    'sql_error': result.get('error'),
                    'sql': result.get('sql'),
                    'constraint': constraint
                }
            
            violation_count = result['row_count']
            print(f"-> Step 5: Found {violation_count} violations")

            # Only apply ratio heuristic for cross-row constraints.
            # Note: a high violation count does not necessarily mean the constraint
            # is wrong — some rules genuinely flag many rows. This check is a loose
            # signal and the threshold is configurable via violation_ratio_threshold.
            is_single_row = not any('t2.' in p for p in constraint)
            if not is_single_row and self.df is not None:
                total_pairs = len(self.df) * (len(self.df) - 1) // 2
                if total_pairs > 0 and violation_count > total_pairs * self.violation_ratio_threshold:
                    return {
                        'is_valid': False,
                        'message': (
                            f"Too many violations ({violation_count} of {total_pairs} pairs), "
                            f"constraint may be too loose or logically inverted"
                        ),
                        'violation_count': violation_count,
                        'constraint': constraint
                    }
            
            return {
                'is_valid': True,
                'message': f"Constraint valid, found {violation_count} violations",
                'violation_count': violation_count,
                'constraint': constraint
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'message': f"Verification error: {e}",
                'error': str(e),
                'constraint': constraint
            }
        
    def _check_predicate_satisfiability(self, constraint: List[str]) -> List[str]:
        """
        Para cada predicado com constante literal (=, LIKE, IN),
        verifica se ao menos 1 tupla o satisfaz na tabela.
        Retorna lista de predicados 'mortos' (valor inexistente nos dados).
        """
        if self.db_executor is None:
            return []

        import re
        dead = []

        # Padrões que implicam busca por valor — se COUNT=0, valor foi inventado
        checkable = re.compile(
            r"t1\.(\w+)\s*(?:=\s*'[^']+'|LIKE\s*'[^']+'|IN\s*\([^)]+\))"
        )

        for pred in constraint:
            # Ignorar: cross-row, IS NULL, IS NOT NULL, != (negação não precisa de match)
            if 't2.' in pred:
                continue
            if any(op in pred for op in ['IS NULL', 'IS NOT NULL', '!=']):
                continue
            if not checkable.search(pred):
                continue

            # Remove alias t1. para rodar direto na tabela
            clean = pred.replace('t1.', '')
            sql = f"SELECT COUNT(*) FROM data_table WHERE {clean}"

            try:
                result = self.db_executor.execute(sql)
                count = result[0][0] if result else 0
                if count == 0:
                    dead.append(pred)
            except Exception:
                pass  # erro de SQL não descarta — pode ser tipo errado, não valor inventado

        return dead

    def _step6_refine_constraint(self,
                                  constraint: List[str],
                                  sentence: str,
                                  verification: Dict[str, Any],
                                  schema_analysis: Dict[str, Any],
                                  predicate_selection: Dict[str, Any]) -> List[str]:
        """Step 6: Refine constraint based on verification results."""
        error_msg = verification.get('message', '')
        sql_error = verification.get('sql_error', '')
        sql_error_section = f"SQL error: {sql_error}" if sql_error else ""

        relevant_columns = schema_analysis.get('relevant_columns', [])
        profiles_desc = self._format_column_profiles_for(relevant_columns)

        # Pass all candidate predicates for these columns, not just previously selected
        candidate_predicates = self._filter_predicates_by_columns(relevant_columns)
        available_predicates_str = '\n'.join([f"  - {p}" for p in sorted(candidate_predicates)])

        prompt_template = self.prompt_loader.load(self.prompt_file, 'refine_constraint')
        prompt = prompt_template.format(
            sentence=sentence,
            constraint=json.dumps(constraint),
            error_msg=error_msg,
            sql_error_section=sql_error_section,
            available_columns=', '.join(relevant_columns),
            available_predicates=available_predicates_str,
            column_profiles_desc=profiles_desc
        )
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(extract_json(response))
            refined = parsed.get('refined_constraint', constraint)
            
            if refined and isinstance(refined, list):
                print(f"   Refined to: {refined}")
                return refined
            
            return constraint
        except Exception as e:
            print(f"-> Refinement error: {e}")
            return constraint

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _format_schema_for_llm(self) -> str:
        """Format schema information for LLM prompts."""
        if not self.tables:
            return "Schema: No table information available"
        
        lines = ["Available schema:"]
        for table in self.tables:
            lines.append(f"\nTable: {table['name']}")
            for col in table['columns']:
                lines.append(f"  - {col['name']} ({col['type']})")
        
        return '\n'.join(lines)

    def _filter_predicates_by_columns(self, columns: List[str]) -> List[str]:
        """Filter allowed predicates to those relevant for given columns."""
        if not columns or not self.allowed_predicates:
            return list(self.allowed_predicates)
        
        relevant = [
            pred for pred in self.allowed_predicates
            if any(col in pred for col in columns)
        ]
        
        return relevant if relevant else list(self.allowed_predicates)

    def _find_similar_columns(self, target: str, available: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """Find columns similar to target using token-based similarity."""
        similarities = [(col, token_similarity(target, col)) for col in available]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def close(self):
        """Cleanup resources."""
        if self.db_executor:
            self.db_executor.close()


# ============================================================================
# Simplified Variant: Direct Refinement Mapper
# ============================================================================

class DirectRefinementMapper(BaseMapper):
    """
    Simplified mapper that focuses on iterative refinement.
    
    Skips multi-step analysis and goes straight to:
    1. Generate initial constraint
    2. Verify on data
    3. Refine if errors detected
    """
    
    def __init__(self,
                 model_name: str = "gpt-4o-mini",
                 predicate_space: Optional[Dict[str, Any]] = None,
                 df: Optional[pd.DataFrame] = None,
                 max_iterations: int = 2,
                 prompt_file: str = 'multi_step_refinement',
                 chunk_size: Optional[int] = None):
        super().__init__(chunk_size=chunk_size)
        self.llm = get_llm_client(model_name)
        self.prompt_loader = PromptLoader()
        self.prompt_file = prompt_file
        self.predicate_space = predicate_space or {}
        self.df = df
        self.max_iterations = max_iterations
        
        self.db_executor = None
        self.sql_translator = None
        if df is not None:
            self.db_executor = DBExecutor(df, table_name="data_table", tid_column="tid")
            self.sql_translator = SQLTranslator(self.db_executor)
        
        self.allowed_predicates = set(predicate_space.get('predicates', []))

    def _map_sentences(self,
                       sentences: List[str],
                       context: Dict[str, Any]) -> List[Optional[List[str]]]:
        """Simple refinement loop for each sentence."""
        results = []
        for sentence in sentences:
            print(f"\n{'='*60}")
            print(f"Processing: {sentence[:80]}...")

            constraint = self._generate_initial(sentence, context)
            
            if not constraint:
                print("-> Could not generate initial constraint")
                results.append(None)
                continue

            print(f"-> Initial constraint: {constraint}")

            for iteration in range(self.max_iterations):
                print(f"-> Verification iteration {iteration + 1}/{self.max_iterations}")
                verification = self._verify(constraint)
                
                if verification['is_valid']:
                    print(f"-> Verification successful ({verification.get('violation_count', '?')} violations)")
                    break

                print(f"-> Refining - {verification.get('error', 'validation failed')}")
                refined = self._refine(sentence, constraint, verification, context)

                if refined == constraint:
                    print("-> No refinement possible")
                    break
                
                print(f"-> Refined to: {refined}")
                constraint = refined
            
            results.append(constraint)
        
        return results

    def _generate_initial(self, sentence: str, context: Dict[str, Any]) -> Optional[List[str]]:
        """Generate initial constraint attempt."""
        predicates_list = '\n'.join([f"  - {p}" for p in sorted(self.allowed_predicates)])
        
        prompt_template = self.prompt_loader.load(self.prompt_file, 'generate_initial')
        prompt = prompt_template.format(
            sentence=sentence,
            predicates_list=predicates_list
        )
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(extract_json(response))
            return parsed.get('constraint')
        except Exception as e:
            print(f"-> Parse error: {e}")
            return None

    def _verify(self, constraint: List[str]) -> Dict[str, Any]:
        """Verify constraint on database."""
        if not self.sql_translator:
            return {'is_valid': True}
        
        try:
            result = self.sql_translator.execute_violation(constraint)

            if not result['success']:
                return {'is_valid': False, 'error': result.get('error')}

            return {'is_valid': True, 'violation_count': result['row_count']}
        except Exception as e:
            return {'is_valid': False, 'error': str(e)}

    def _refine(self, sentence: str, constraint: List[str],
                verification: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Refine constraint based on verification."""
        error = verification.get('error', 'validation failed')
        predicates_list = '\n'.join([f"  - {p}" for p in sorted(self.allowed_predicates)])
        
        prompt_template = self.prompt_loader.load(self.prompt_file, 'refine_simple')
        prompt = prompt_template.format(
            sentence=sentence,
            constraint=json.dumps(constraint),
            error=error,
            predicates_list=predicates_list
        )
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(extract_json(response))
            return parsed.get('refined', constraint)
        except Exception as e:
            print(f"-> Refinement parse error: {e}")
            return constraint

    def close(self):
        if self.db_executor:
            self.db_executor.close()