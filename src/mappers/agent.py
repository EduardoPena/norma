#!/usr/bin/env python3
"""
LangGraph-based Agent Mapper for Denial Constraints

Uses an agentic approach with tools to iteratively explore schema,
sample data, and refine constraint generation.
"""

import json
import re
from typing import List, Dict, Any, TypedDict, Annotated, Literal
from operator import add
import pandas as pd

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.llm.client import get_llm_client
from src.dbms.db_interface import DBExecutor
from src.dbms.sql_translator import SQLTranslator


# ============================================================================
# Agent State
# ============================================================================

class AgentState(TypedDict):
    """State for the agent workflow"""
    # Input
    sentence: str
    predicate_space: Dict[str, Any]
    df: pd.DataFrame
    
    # Agent working memory
    schema_info: str
    sample_data: Dict[str, List[Any]]
    candidate_predicates: List[str]
    reasoning: Annotated[List[str], add]  # Accumulated reasoning steps
    
    # Attempts and refinement
    iteration: int
    max_iterations: int
    
    # Output
    final_constraint: List[str]
    constraint_valid: bool
    violation_count: int


# ============================================================================
# Tools for the Agent
# ============================================================================

class ConstraintTools:
    """Tools available to the agent for constraint generation"""
    
    def __init__(self, df: pd.DataFrame, predicate_space: Dict[str, Any], 
                 llm_model: str = "gpt-5-mini"):
        self.df = df
        self.predicate_space = predicate_space
        self.llm_client = get_llm_client(llm_model)
        
        # Initialize DB executor and translator
        self.db_executor = DBExecutor(df, table_name="data_table", tid_column="tid")
        self.sql_translator = SQLTranslator(self.db_executor)
        
        # Extract schema info
        self.tables = predicate_space.get('tables', [])
        self.allowed_predicates = set(predicate_space.get('predicates', []))
    
    def query_schema(self, table_name: str = None) -> str:
        """
        Get schema information for tables.
        
        Args:
            table_name: Optional specific table name, otherwise returns all
            
        Returns:
            Human-readable schema description
        """
        if not self.tables:
            return "No schema information available"
        
        result = []
        for table in self.tables:
            if table_name and table['name'] != table_name:
                continue
                
            result.append(f"\n=== Table: {table['name']} ===")
            for col in table['columns']:
                ops = ', '.join(col['allowed_ops'])
                result.append(f"  - {col['name']}: {col['type']} (ops: {ops})")
        
        return '\n'.join(result)
    
    def query_predicates(self, sentence: str, top_k: int = 10) -> List[str]:
        """
        Use LLM to suggest most relevant predicates for a sentence.
        
        Args:
            sentence: The constraint sentence
            top_k: Number of predicates to return
            
        Returns:
            List of suggested predicate strings
        """
        schema_info = self.query_schema()
        
        prompt = f"""Given this constraint sentence, identify the {top_k} most relevant predicates.

Sentence: "{sentence}"

Available Schema:
{schema_info}

Available Predicates (select from these):
{chr(10).join(list(self.allowed_predicates)[:50])}  # Show first 50

Task: Select the {top_k} most relevant predicates that would be needed to express this constraint.

Output format: Return ONLY a JSON array of predicate strings, nothing else.
Example: ["t1.field1 = t2.field1", "t1.field2 != t2.field2"]

Output:"""
        
        try:
            response = self.llm_client.complete(prompt)
            # Extract JSON array
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                predicates = json.loads(match.group(0))
                # Validate against allowed predicates
                valid = [p for p in predicates if p in self.allowed_predicates]
                return valid[:top_k]
            return []
        except Exception as e:
            print(f"Warning: query_predicates failed: {e}")
            return []
    
    def sample_column(self, column_name: str, n_samples: int = 5, 
                     distinct: bool = True) -> List[Any]:
        """
        Get sample values from a column.
        
        Args:
            column_name: Column to sample
            n_samples: Number of samples
            distinct: Whether to get distinct values
            
        Returns:
            List of sample values
        """
        if column_name not in self.df.columns:
            return []
        
        if distinct:
            samples = self.df[column_name].dropna().unique()[:n_samples].tolist()
        else:
            samples = self.df[column_name].dropna().head(n_samples).tolist()
        
        return samples
    
    def check_violation(self, predicates: List[str]) -> Dict[str, Any]:
        """
        Test if a constraint (list of predicates) finds violations.
        
        Args:
            predicates: List of predicate strings
            
        Returns:
            Dict with success, violation_count, sample_violations
        """
        try:
            result = self.sql_translator.execute_violation(predicates)
            
            if not result['success']:
                return {
                    'success': False,
                    'violation_count': 0,
                    'error': result.get('error', 'Unknown error'),
                    'sample_violations': []
                }
            
            return {
                'success': True,
                'violation_count': result['row_count'],
                'constraint_violated': result['constraint_violated'],
                'sample_violations': result['rows'][:3] if result['rows'] else []
            }
        except Exception as e:
            return {
                'success': False,
                'violation_count': 0,
                'error': str(e),
                'sample_violations': []
            }
    
    def test_constraint(self, predicates: List[str]) -> Dict[str, Any]:
        """
        Comprehensive test of a constraint.
        
        Args:
            predicates: List of predicate strings
            
        Returns:
            Dict with validity, violation info, and feedback
        """
        # Test translation
        translation = self.sql_translator.translate(predicates)
        if not translation['success']:
            return {
                'valid': False,
                'error': f"Translation failed: {translation.get('error')}",
                'feedback': 'Predicates could not be translated to SQL. Check syntax and column names.'
            }
        
        # Test execution
        result = self.check_violation(predicates)
        if not result['success']:
            return {
                'valid': False,
                'error': result['error'],
                'feedback': 'Constraint execution failed. Check column names and types.'
            }
        
        feedback = []
        feedback.append(f"Constraint type: {translation['constraint_type']}")
        feedback.append(f"Violations found: {result['violation_count']}")
        
        if result['violation_count'] == 0:
            feedback.append("Warning: No violations found. Constraint may be too strict or incorrect.")
        elif result['violation_count'] > len(self.df) * 0.5:
            feedback.append("Warning: High violation count. Constraint may be too loose.")
        else:
            feedback.append("Constraint appears reasonable.")
        
        return {
            'valid': True,
            'violation_count': result['violation_count'],
            'constraint_violated': result['constraint_violated'],
            'sql': translation['sql'],
            'feedback': ' '.join(feedback)
        }
    
    def close(self):
        """Clean up resources"""
        self.db_executor.close()


# ============================================================================
# Agent Nodes
# ============================================================================

def should_continue(state: AgentState) -> Literal["refine", "end"]:
    """Determine if agent should continue or end"""
    if state['iteration'] >= state['max_iterations']:
        return "end"
    
    if state.get('constraint_valid') and state.get('violation_count', 0) > 0:
        return "end"
    
    return "refine"


def understand_sentence_node(state: AgentState, tools: ConstraintTools) -> AgentState:
    """Initial understanding of the sentence and schema"""
    sentence = state['sentence']
    
    # Get relevant schema info
    schema_info = tools.query_schema()
    
    # Query for relevant predicates
    candidate_predicates = tools.query_predicates(sentence, top_k=15)
    
    reasoning = [
        f"Analyzing sentence: '{sentence}'",
        f"Found {len(candidate_predicates)} candidate predicates from schema"
    ]
    
    return {
        **state,
        'schema_info': schema_info,
        'candidate_predicates': candidate_predicates,
        'reasoning': reasoning,
        'iteration': 0
    }


def generate_constraint_node(state: AgentState, tools: ConstraintTools) -> AgentState:
    """Generate constraint using LLM with available context"""
    sentence = state['sentence']
    schema_info = state['schema_info']
    candidate_predicates = state['candidate_predicates']
    iteration = state['iteration']
    
    # Build prompt with all available information
    prompt = f"""Task: Convert this natural language sentence into a denial constraint.

Sentence: "{sentence}"

Schema Information:
{schema_info}

Relevant Candidate Predicates (use these if applicable):
{chr(10).join(candidate_predicates[:10])}

Instructions:
1. A denial constraint is a list of predicates that CANNOT all be true simultaneously
2. Use t1. and t2. as tuple identifiers for cross-tuple constraints
3. For single-tuple constraints, use t1. prefix
4. Extract concrete values from the sentence
5. Map sentence terms to actual schema column names

{f"Previous attempt feedback: This is iteration {iteration}. Previous attempts had issues. Try a different approach." if iteration > 0 else ""}

Output format: Return ONLY a JSON array of predicate strings, nothing else.
Example: ["t1.field1 = t2.field1", "t1.field2 != t2.field2"]

Output:"""
    
    try:
        response = tools.llm_client.complete(prompt)
        
        # Extract JSON array
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            predicates = json.loads(match.group(0))
            reasoning = state['reasoning'] + [f"Generated {len(predicates)} predicates"]
            
            # Test the constraint
            test_result = tools.test_constraint(predicates)
            
            reasoning.append(f"Test result: {test_result.get('feedback', 'Unknown')}")
            
            return {
                **state,
                'final_constraint': predicates if test_result['valid'] else [],
                'constraint_valid': test_result['valid'],
                'violation_count': test_result.get('violation_count', 0),
                'reasoning': reasoning,
                'iteration': iteration + 1
            }
        else:
            reasoning = state['reasoning'] + ["Failed to extract predicates from LLM response"]
            return {
                **state,
                'final_constraint': [],
                'constraint_valid': False,
                'reasoning': reasoning,
                'iteration': iteration + 1
            }
            
    except Exception as e:
        reasoning = state['reasoning'] + [f"Error: {str(e)}"]
        return {
            **state,
            'final_constraint': [],
            'constraint_valid': False,
            'reasoning': reasoning,
            'iteration': iteration + 1
        }


def refine_constraint_node(state: AgentState, tools: ConstraintTools) -> AgentState:
    """Refine constraint based on test results"""
    # Sample some columns to better understand the data
    sample_data = {}
    for col in list(state['df'].columns)[:5]:
        samples = tools.sample_column(col, n_samples=3)
        if samples:
            sample_data[col] = samples
    
    reasoning = state['reasoning'] + [
        f"Sampled data from {len(sample_data)} columns for refinement"
    ]
    
    # Proceed to generate again with more context
    return generate_constraint_node({**state, 'sample_data': sample_data, 'reasoning': reasoning}, tools)


# ============================================================================
# Agent Workflow
# ============================================================================

class AgentMapper:
    """LangGraph-based mapper with agentic constraint generation"""
    
    def __init__(self, df: pd.DataFrame, predicate_space: Dict[str, Any],
                 llm_model: str = "gpt-5-mini", max_iterations: int = 3):
        self.df = df
        self.predicate_space = predicate_space
        self.llm_model = llm_model
        self.max_iterations = max_iterations
        
        # Initialize tools
        self.tools = ConstraintTools(df, predicate_space, llm_model)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("understand", lambda state: understand_sentence_node(state, self.tools))
        workflow.add_node("generate", lambda state: generate_constraint_node(state, self.tools))
        workflow.add_node("refine", lambda state: refine_constraint_node(state, self.tools))
        
        # Define edges
        workflow.set_entry_point("understand")
        workflow.add_edge("understand", "generate")
        
        # Conditional edge based on should_continue
        workflow.add_conditional_edges(
            "generate",
            should_continue,
            {
                "refine": "refine",
                "end": END
            }
        )
        workflow.add_edge("refine", "generate")
        
        return workflow.compile()
    
    def map_sentence(self, sentence: str, verbose: bool = False) -> List[str]:
        """
        Map a sentence to a denial constraint.
        
        Args:
            sentence: Natural language constraint sentence
            verbose: Print reasoning steps
            
        Returns:
            List of predicate strings forming the constraint, or empty list if failed
        """
        initial_state = {
            'sentence': sentence,
            'predicate_space': self.predicate_space,
            'df': self.df,
            'schema_info': '',
            'sample_data': {},
            'candidate_predicates': [],
            'reasoning': [],
            'iteration': 0,
            'max_iterations': self.max_iterations,
            'final_constraint': [],
            'constraint_valid': False,
            'violation_count': 0
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        if verbose:
            print("\n=== Agent Reasoning ===")
            for step in final_state['reasoning']:
                print(f"  - {step}")
            print(f"  - Final: {'Valid' if final_state['constraint_valid'] else 'Invalid'} constraint")
            print(f"  - Violations found: {final_state['violation_count']}")
        
        return final_state['final_constraint']
    
    def close(self):
        """Clean up resources"""
        self.tools.close()


# ============================================================================
# Main Function for Testing
# ============================================================================

def main():
    """Test the agent mapper with example sentences"""
    print("=" * 80)
    print("Agent-Based Constraint Mapper - Testing")
    print("=" * 80)
    
    # Load hospital dataset
    print("\n1. Loading hospital dataset...")
    df = pd.read_csv('data/hospital/hospital.csv')
    
    with open('data/hospital/hospital_predicates.json', 'r') as f:
        predicate_space = json.load(f)
    
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Predicate space: {len(predicate_space['predicates'])} predicates")
    
    # Initialize agent
    print("\n2. Initializing agent mapper with GPT-5-mini...")
    agent = AgentMapper(df, predicate_space, llm_model="gpt-5-mini", max_iterations=3)
    
    # Test sentences
    test_sentences = [
        "A provider number uniquely identifies a hospital name.",
        "A hospital name uniquely determines its provider number.",
        "A hospital name uniquely determines its ZIP code.",
    ]
    
    print("\n3. Testing constraint generation...")
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n--- Test {i} ---")
        print(f"Sentence: {sentence}")
        
        constraint = agent.map_sentence(sentence, verbose=True)
        
        if constraint:
            print(f"\n✓ Generated Constraint:")
            for pred in constraint:
                print(f"    {pred}")
        else:
            print(f"\n✗ Failed to generate valid constraint")
    
    # Clean up
    agent.close()
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    main()
