#!/usr/bin/env python3
"""
Test the Multi-Step Refinement Mapper

Demonstrates the multi-step refinement process.
"""

import pandas as pd
from src.mappers.multi_step_refinement import MultiStepRefinementMapper, DirectRefinementMapper
from src.data.loader import DatasetLoader


def test_full_pipeline():
    """Test the complete MultiStepRefinementMapper on dataset."""
    print("="*80)
    print("Testing MultiStepRefinementMapper (Full Multi-Step Pipeline)")
    print("="*80)
    
    # Load dataset (predicate_space is already built by loader)
    loader = DatasetLoader('data/sisu')
    df, sentences, sentence2dcs_map, predicate_space = loader.load()
    
    # Create mapper
    mapper = MultiStepRefinementMapper(
        model_name='gpt-5-mini',
        predicate_space=predicate_space,
        df=df,
        max_refinement_iterations=2
    )
    
    # Test sentences
    test_sentences = list(sentence2dcs_map.keys())[:2]
    
    non_constraints =  [
        # "Os estudantes gostam de pizza",
        "A maioria dos alunos estuda à noite",
        "Estudantes portugueses são muito dedicados"
    ]

    test_sentences.extend(non_constraints)



    print(f"\nTesting {len(test_sentences)} sentences...")
    
    results = mapper.map(test_sentences)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for sentence, constraint in results:
        print(f"\nSentence: {sentence}")
        if constraint:
            print(f"Constraint: {constraint}")
        else:
            print("Result: NOT A CONSTRAINT")
    
    mapper.close()


def test_direct_refinement():
    """Test the simplified Direct Refinement mapper."""
    print("\n" + "="*80)
    print("Testing DirectRefinementMapper (Simplified Version)")
    print("="*80)
    
    # Load dataset (predicate_space is already built by loader)
    loader = DatasetLoader('data/restaurant')
    df, sentences, sentence2dcs_map, predicate_space = loader.load()
    
    # Create mapper
    mapper = DirectRefinementMapper(
        model_name='gpt-5-mini',
        predicate_space=predicate_space,
        df=df,
        max_iterations=2
    )
    
    # Test a few sentences
    test_sentences = list(sentence2dcs_map.keys())[:3]
    
    print(f"\nTesting {len(test_sentences)} sentences...")
    
    results = mapper.map(test_sentences)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for sentence, constraint in results:
        print(f"\nSentence: {sentence[:80]}...")
        ground_truth = sentence2dcs_map.get(sentence)
        
        print(f"Predicted:  {constraint}")
        print(f"Ground Truth: {ground_truth}")
        
        if constraint and ground_truth:
            match = set(constraint) == set(ground_truth)
            print(f"Match: {'✓' if match else '✗'}")
    
    mapper.close()


def test_comparison():
    """Compare MultiStepRefinement vs DirectRefinement on same data."""
    print("\n" + "="*80)
    print("Comparing MultiStepRefinement vs DirectRefinement")
    print("="*80)
    
    # Load dataset (predicate_space is already built by loader)
    loader = DatasetLoader('data/person')
    df, sentences, sentence2dcs_map, predicate_space = loader.load()
    
    test_sentence = list(sentence2dcs_map.keys())[0]
    print(f"\nTest sentence: {test_sentence}")
    print(f"Ground truth: {sentence2dcs_map[test_sentence]}")
    
    # Test MultiStepRefinement
    print("\n" + "-"*80)
    print("MultiStepRefinement (Full Multi-Step):")
    print("-"*80)
    mapper1 = MultiStepRefinementMapper(
        model_name='gpt-5-mini',
        predicate_space=predicate_space,
        df=df,
        max_refinement_iterations=2
    )
    result1 = mapper1.map([test_sentence])[0][1]
    print(f"Result: {result1}")
    mapper1.close()
    
    # Test DirectRefinement
    print("\n" + "-"*80)
    print("DirectRefinement (Simplified):")
    print("-"*80)
    mapper2 = DirectRefinementMapper(
        model_name='gpt-5-mini',
        predicate_space=predicate_space,
        df=df,
        max_iterations=2
    )
    result2 = mapper2.map([test_sentence])[0][1]
    print(f"Result: {result2}")
    mapper2.close()
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print("Ground Truth:       {}".format(sentence2dcs_map[test_sentence]))
    print("MultiStepRefinement: {}".format(result1))
    print("DirectRefinement:    {}".format(result2))
    
    if result1 and sentence2dcs_map[test_sentence]:
        match1 = set(result1) == set(sentence2dcs_map[test_sentence])
        print(f"\nMultiStepRefinement Match: {'Y' if match1 else 'N'}")
    
    if result2 and sentence2dcs_map[test_sentence]:
        match2 = set(result2) == set(sentence2dcs_map[test_sentence])
        print(f"DirectRefinement Match:    {'Y' if match2 else 'N'}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        
        if test_type == 'full':
            test_full_pipeline()
        elif test_type == 'direct':
            test_direct_refinement()
        elif test_type == 'compare':
            test_comparison()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python test_redinspered.py [full|direct|compare]")
    else:
        # Run all tests
        test_full_pipeline()
        print("\n\n")
        # test_direct_refinement()
        # print("\n\n")
        # test_comparison()
