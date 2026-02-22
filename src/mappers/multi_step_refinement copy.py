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
        model_name='gpt-5-mini',
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
                 prompt_file: str = 'multi_step_refinement',
                 chunk_size: Optional[int] = None):
        """
        Args:
            model_name: LLM model for all reasoning steps
            predicate_space: Schema and predicate definitions (use DatasetLoader.load() to get this)
            df: DataFrame for constraint validation
            max_refinement_iterations: Max refinement cycles 
            prompt_file: Name of prompt YAML file (without .yaml extension)
            chunk_size: Process in batches
        """
        super().__init__(chunk_size=chunk_size)
        self.llm = get_llm_client(model_name)
        self.prompt_loader = PromptLoader()
        self.prompt_file = prompt_file
        self.predicate_space = predicate_space or {}
        self.df = df
        self.max_refinement_iterations = max_refinement_iterations
        
        # Initialize DB components if DataFrame provided
        self.db_executor = None
        self.sql_translator = None
        if df is not None:
            self.db_executor = DBExecutor(df, table_name="data_table", tid_column="tid")
            self.sql_translator = SQLTranslator(self.db_executor)
        
        # Extract schema information
        self.tables = predicate_space.get('tables', [])
        self.allowed_predicates = set(predicate_space.get('predicates', []))
        self.column_profiles = self._build_column_profiles()
    
    def _build_column_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Build column profiles 
        
        Returns:
            Dict mapping column names to their profiles (type, sample values, etc.)
        """
        profiles = {}
        
        if self.df is None or self.df.empty:
            return profiles
        
        for col in self.df.columns:
            if col == 'tid':  # Skip internal columns
                continue
            
            col_data = self.df[col].dropna()

            unique_vals = col_data.unique()
            n_unique = len(unique_vals)

            if n_unique <= 10:
                # Too few distinct values: just use all of them
                sample_values = unique_vals.tolist()
            else:
                # Top 5 most frequent unique values
                top_frequent = (
                    col_data.value_counts()
                    .head(5)
                    .index
                    .tolist()
                )

                # 5 random unique values, excluding already selected ones
                remaining = [v for v in unique_vals if v not in set(top_frequent)]
                n_random = min(5, len(remaining))
                random_sample = (
                    pd.Series(remaining)
                    .sample(n=n_random, random_state=42)
                    .tolist()
                    if n_random > 0 else []
                )

                # Combine, preserving order: frequent first, then random
                sample_values = top_frequent + random_sample


            profile = {
                'type': str(self.df[col].dtype),
                'null_count': self.df[col].isnull().sum(),
                'unique_count': self.df[col].nunique(),
                # 'sample_values': col_data.head(5).tolist() if not col_data.empty else []
                'sample_values': sample_values
            }
            
            # Add numeric stats if applicable
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
        
        lines = ["\nColumn profiles (use these to verify string constants):"]
        for col in columns:
            if col in self.column_profiles:
                profile = self.column_profiles[col]
                samples = profile.get('sample_values', [])
                samples_str = ', '.join([
                    f"'{s}'" if isinstance(s, str) else str(s)
                    for s in samples[:8]
                ])
                lines.append(f"  - {col}: [{samples_str}]")
        
        return '\n'.join(lines) if len(lines) > 1 else ""
    
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
        """
        Multi-step refinement for a single sentence.
        
        Steps:
        1. Classification: Is this a constraint?
        2. Schema Understanding: What columns are relevant?
        3. Predicate Selection: What predicates to use?
        4. Constraint Construction: Build initial constraint
        5. Verification: Test on data
        6. Refinement: Fix errors (iterative)
        
        Returns:
            List of predicates or None if not a constraint
        """
        print(f"\n{'='*60}")
        print(f"Processing: {sentence[:80]}...")
        
        # Step 1: Check Mappability
        is_mappable = self._step1_check_mappability(sentence)
        if not is_mappable:
            print("-> Sentence is not mappable to a denial constraint")
            return None
        
        # Step 2: Schema Understanding
        schema_analysis = self._step2_understand_schema(sentence)
        if not schema_analysis['is_valid']:
            print(f"-> Schema validation failed: {schema_analysis['reason']}")
            return None
        
        # Step 3: Predicate Selection
        predicate_selection = self._step3_select_predicates(sentence, schema_analysis)
        if not predicate_selection['is_valid']:
            print(f"-> Predicate selection failed: {predicate_selection['reason']}")
            return None
        
        # Step 4: Initial Construction
        initial_constraint = self._step4_construct_constraint(
            sentence, schema_analysis, predicate_selection
        )
        if not initial_constraint:
            print("-> Could not construct initial constraint")
            return None
        
        # Step 5 & 6: Verification and Refinement Loop
        constraint = initial_constraint
        for iteration in range(self.max_refinement_iterations):
            print(f"\n-> Verification iteration {iteration + 1}/{self.max_refinement_iterations}")
            
            verification = self._step5_verify_constraint(constraint, sentence)
            
            if verification['is_valid']:
                print(f"-> Verification successful")
                return constraint
            
            # Step 6: Refinement
            print(f"-> Step 6: Refining - {verification['message']}")
            refined = self._step6_refine_constraint(
                constraint, sentence, verification, schema_analysis, predicate_selection
            )
            
            if refined == constraint:
                print("-> No refinement possible")
                return constraint
            
            constraint = refined
        
        print(f"-> Max iterations reached")
        return constraint
    
    def _step1_check_mappability(self, sentence: str) -> bool:
        """
        Step 1: Check if sentence is mappable to a denial constraint.
        
        Since sentences come from a classifier, we just check if they contain
        references to columns, values, or predicates (vs. procedural/meta content).
        """
        prompt_template = self.prompt_loader.load(self.prompt_file, 'check_mappability')

        escaped_sentence = sentence.replace('"', '\\"').replace('\n', '\\n')

        prompt = prompt_template.format(sentence=escaped_sentence)
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(self._extract_json(response))
            is_mappable = parsed.get('is_mappable', False)
            if not is_mappable:
                print(f"-> Step 1: NOT mappable - {parsed.get('reasoning', 'unknown')}")
            return is_mappable
        except Exception as e:
            print(f"-> Step 1 parsing error: {e}")
            return True
    
    def _step2_understand_schema(self, sentence: str) -> Dict[str, Any]:
        """
        Step 2: Understand relevant columns and verify they exist.
        
        Schema expansion - identifies and validates schema elements.
        """
        # Build schema context
        schema_desc = self._format_schema_for_llm()
        column_profiles_desc = self._format_column_profiles()
        
        prompt_template = self.prompt_loader.load(self.prompt_file, 'understand_schema')
        prompt = prompt_template.format(
            sentence=sentence,
            schema_desc=schema_desc,
            column_profiles_desc=column_profiles_desc
        )
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(self._extract_json(response))
            
            # Validate columns actually exist
            relevant = parsed.get('relevant_columns', [])
            available_columns = {col for table in self.tables for col in [c['name'] for c in table['columns']]}
            available_list = list(available_columns)
            
            missing = [col for col in relevant if col not in available_columns]
            if missing:
                # Suggest similar columns
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
            return {
                'is_valid': False,
                'reason': f'Failed to parse LLM response: {e}',
                'relevant_columns': []
            }
    
    def _step3_select_predicates(self,
                                 sentence: str,
                                 schema_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Select predicates from allowed set.
        
        Predicate validation - ensures only valid operations are used.
        """
        relevant_columns = schema_analysis.get('relevant_columns', [])
        
        # Filter predicates relevant to these columns
        candidate_predicates = self._filter_predicates_by_columns(relevant_columns)
        
        if not candidate_predicates:
            return {
                'is_valid': False,
                'reason': 'No valid predicates found for these columns',
                'selected_predicates': []
            }
        
        predicates_desc = '\n'.join([f"  - {p}" for p in sorted(candidate_predicates)])
        column_profiles_desc = self._format_column_profiles()  
        
        prompt_template = self.prompt_loader.load(self.prompt_file, 'select_predicates')
        prompt = prompt_template.format(
            sentence=sentence,
            relevant_columns=', '.join(relevant_columns),
            predicates_desc=predicates_desc,
            column_profiles_desc=column_profiles_desc
        )
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(self._extract_json(response))
            
            # Validate all selected predicates are in allowed set
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
            return {
                'is_valid': False,
                'reason': f'Parse error: {e}',
                'selected_predicates': []
            }
    
    def _step4_construct_constraint(self,
                                    sentence: str,
                                    schema_analysis: Dict[str, Any],
                                    predicate_selection: Dict[str, Any]) -> Optional[List[str]]:
        """
        Step 4: Construct initial denial constraint.
        
        Uses LLM to generate the constraint formula.
        """
        relevant_columns = schema_analysis.get('relevant_columns', [])
        selected_predicates = predicate_selection.get('selected_predicates', [])

        
        profiles_desc = self._format_column_profiles_for(relevant_columns)

        print(f"-> Step 4: Constructing constraint with {len(selected_predicates)} predicates and {len(relevant_columns)} columns")
        # print(f"   Relevant columns: {relevant_columns}")
        # print(f"   Selected predicates: {selected_predicates}")
        # print(f"   Column profiles: {profiles_desc}")
        
        prompt_template = self.prompt_loader.load(self.prompt_file, 'construct_constraint')
        prompt = prompt_template.format(
            sentence=sentence,
            relevant_columns=', '.join(relevant_columns),
            selected_predicates=', '.join(selected_predicates),
            column_profiles_desc=profiles_desc
        )
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(self._extract_json(response))
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
        """
        Step 5: Verify constraint on database sample.
        
        Constraints verification - tests on actual data.
        """
        if not self.sql_translator:
            # Can't verify without database
            return {'is_valid': True, 'message': 'No database available for verification'}
        
        # Translate to SQL and execute
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
            
            # Heuristic: if too many violations (> 50% of data), constraint likely wrong
            if self.df is not None:
                total_pairs = len(self.df) * (len(self.df) - 1) // 2  # Approximate
                if total_pairs > 0 and violation_count > total_pairs * 0.5:
                    return {
                        'is_valid': False,
                        'message': f"Too many violations ({violation_count}), constraint may be incorrect",
                        'violation_count': violation_count,
                        'constraint': constraint
                    }
            
            # Success - but inform about violation count
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
    
    def _step6_refine_constraint(self,
                                 constraint: List[str],
                                 sentence: str,
                                 verification: Dict[str, Any],
                                 schema_analysis: Dict[str, Any],
                                 predicate_selection: Dict[str, Any]) -> List[str]:
        """
        Step 6: Refine constraint based on verification results.
        
        SQL Refinement - uses violation reports to fix errors.
        """
        error_msg = verification.get('message', '')
        sql_error = verification.get('sql_error', '')
        sql_error_section = f"SQL error: {sql_error}" if sql_error else ""
        
        prompt_template = self.prompt_loader.load(self.prompt_file, 'refine_constraint')
        prompt = prompt_template.format(
            sentence=sentence,
            constraint=json.dumps(constraint),
            error_msg=error_msg,
            sql_error_section=sql_error_section,
            available_columns=', '.join(schema_analysis.get('relevant_columns', [])),
            available_predicates=', '.join(predicate_selection.get('selected_predicates', []))
        )
        
        response = self.llm.complete(prompt)
        try:
            parsed = json.loads(self._extract_json(response))
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
    
    def _format_column_profiles(self) -> str:
        """Format column profiles (data context) for LLM."""
        if not self.column_profiles:
            return ""
        
        lines = ["\nColumn profiles (sample data context):"]
        for col, profile in list(self.column_profiles.items())[:10]:  # Limit to avoid token overflow
            samples = profile.get('sample_values', [])
            samples_str = ', '.join([str(s) for s in samples[:3]])
            lines.append(f"  - {col}: {profile['type']}, samples: [{samples_str}]")
        
        return '\n'.join(lines)
    
    def _filter_predicates_by_columns(self, columns: List[str]) -> List[str]:
        """Filter allowed predicates to those relevant for given columns."""
        if not columns or not self.allowed_predicates:
            return list(self.allowed_predicates)
        
        # For now, return all predicates that mention any of the columns
        # More sophisticated filtering could check predicate structure
        relevant = []
        for pred in self.allowed_predicates:
            # Check if any column appears in this predicate template
            if any(col in pred for col in columns):
                relevant.append(pred)
        
        # If no matches, return all (better to over-include than under-include)
        return relevant if relevant else list(self.allowed_predicates)
    
    def _find_similar_columns(self, target: str, available: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Find columns similar to target using simple string similarity.
        TODO: Replace with more sophisticated embedding-based similarity if needed.
        
        Args:
            target: Column name to find matches for
            available: List of available column names
            top_n: Number of top matches to return
            
        Returns:
            List of (column_name, similarity_score) tuples
        """
        def simple_similarity(s1: str, s2: str) -> float:
            """Simple similarity: case-insensitive substring matching and character overlap."""
            s1_lower = s1.lower()
            s2_lower = s2.lower()
            
            # Exact match
            if s1_lower == s2_lower:
                return 1.0
            
            # One is substring of other
            if s1_lower in s2_lower or s2_lower in s1_lower:
                return 0.8
            
            # Character overlap (Jaccard on character set)
            set1 = set(s1_lower)
            set2 = set(s2_lower)
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return 0.5 * (intersection / union)
        
        similarities = [(col, simple_similarity(target, col)) for col in available]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        # Try to find JSON in code blocks
        pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find raw JSON
        pattern = r'\{.*?\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        
        return text
    
    def close(self):
        """Cleanup resources."""
        if self.db_executor:
            self.db_executor.close()


# ============================================================================
# Simplified Variant: Direct Refinement Mapper
# ============================================================================

class DirectRefinementMapper(BaseMapper):
    """
    Simplified\ mapper that focuses on iterative refinement.
    
    Skips multi-step analysis and goes straight to:
    1. Generate initial constraint
    2. Verify on data
    3. Refine if errors detected
    """
    
    def __init__(self,
                 model_name: str = "gpt-5-mini",
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
        
        # DB setup
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
            # Generate initial constraint
            constraint = self._generate_initial(sentence, context)
            
            if not constraint:
                results.append(None)
                continue
            
            # Refinement loop
            for iteration in range(self.max_iterations):
                verification = self._verify(constraint)
                
                if verification['is_valid']:
                    break
                
                # Refine based on error
                refined = self._refine(sentence, constraint, verification, context)
                if refined == constraint:
                    break  # No improvement possible
                
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
            parsed = json.loads(self._extract_json(response))
            return parsed.get('constraint')
        except:
            return None
    
    def _verify(self, constraint: List[str]) -> Dict[str, Any]:
        """Verify constraint on database."""
        if not self.sql_translator:
            return {'is_valid': True}
        
        try:
            sql = self.sql_translator.translate(constraint)
            result = self.db_executor.execute_violation_query(sql)
            
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
            parsed = json.loads(self._extract_json(response))
            return parsed.get('refined', constraint)
        except:
            return constraint
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from response."""
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0) if match else text
    
    def close(self):
        if self.db_executor:
            self.db_executor.close()
