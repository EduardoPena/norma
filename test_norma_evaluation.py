# test.py - Quick migration testing


DATA_PATH = 'data/person/'  # Adjust as needed


def test_classification_evaluation():
    """Test and evaluate LLM-based classifier with all available prompts."""
    from src.classifiers.llm import LLMRuleClassifier
    from src.evaluation.classification import ClassificationEvaluator
    from src.data import DatasetLoader
    from src.llm.helper import PromptLoader
    from src.utils.table_context import TableContextBuilder
    import os

    print("\nTesting LLM Classifier with Evaluation")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    # Load data
    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(f"Loaded {len(sentences)} sentences")
    print(
        f"Ground truth rules: {sum(1 for dc in sentence2dcs_map.values() if dc)}")

    # Build table context
    context_builder = TableContextBuilder(sample_strategy="head", num_rows=3)
    context = context_builder.build(df)
    print(f"\nTable context preview:\n{context['table_context'][:200]}...\n")

    # Get all available classification prompts
    prompt_loader = PromptLoader()
    prompt_names = prompt_loader.list_prompts("classification")
    print(f"Available prompts: {prompt_names}\n")

    # Test each prompt
    for prompt_name in prompt_names:
        print(f"\n{'=' * 60}")
        print(f"Testing Prompt: {prompt_name}")
        print('=' * 60)

        # Load prompt template
        prompt_template = prompt_loader.load("classification", prompt_name)

        # Create classifier
        classifier = LLMRuleClassifier(
            model_name="mistral-small-2506",
            prompt_template=prompt_template
        )

        # Classify all sentences
        print(f"Classifying {len(sentences)} sentences...")
        predictions = classifier.classify(sentences, context=context)

        # Evaluate predictions
        evaluator = ClassificationEvaluator(ground_truth=sentence2dcs_map)
        metrics, sentence_results = evaluator.evaluate(predictions)

        # Display metrics
        print(f"\nMetrics:")
        print(f"  Accuracy:  {metrics.accuracy:.3f}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall:    {metrics.recall:.3f}")
        print(f"  F1 Score:  {metrics.f1_score:.3f}")
        print(f"\nConfusion Matrix:")
        print(
            f"  TP: {metrics.true_positives:3d}  |  FP: {metrics.false_positives:3d}")
        print(
            f"  FN: {metrics.false_negatives:3d}  |  TN: {metrics.true_negatives:3d}")

        # Show some examples of each category
        print(f"\nExample Results:")

        # Show first TP, FP, FN (if any)
        for status in ['TP', 'FP', 'FN']:
            examples = [r for r in sentence_results if r.status == status][:2]
            if examples:
                print(f"\n{status}:")
                for ex in examples:
                    print(f"  {ex.sentence[:80]}...")
                    print(
                        f"    Gold: {ex.gold_is_rule}, Pred: {ex.pred_is_rule}")

    print("\n" + "=" * 60)
    print("Classification evaluation complete")

def test_norma_document_evaluation():
    """Test NORMA DOCUMENT pipeline with aggregate execution metrics"""
    from src.pipelines.norma_pipeline import NormaPipeline, ProcessingMode
    from src.mappers.llm_document_mapper import LLMDocumentMapper
    from src.evaluation.mapping import MappingEvaluator
    from src.data import DatasetLoader
    from src.utils.table_context import TableContextBuilder
    from src.llm.helper import PromptLoader
    import os

    print("\nTesting NORMA DOCUMENT Pipeline with Evaluation")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    # Load data
    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(f"Loaded {len(sentences)} sentences, {len(df)} rows")
    gt_dcs = [dc for dc in sentence2dcs_map.values() if dc]
    print(f"Ground truth constraints: {len(gt_dcs)}")

    # Build table context
    context_builder = TableContextBuilder(sample_strategy="head", num_rows=3)
    context = context_builder.build(df)
    
    # Add predicates list to context
    predicates_list_str = "\n".join(predicate_space['predicates'])
    context['predicates_list'] = predicates_list_str

    # Get document mapping prompts
    prompt_loader = PromptLoader()
    document_prompt_names = prompt_loader.list_prompts("document_mapping")
    document_prompt = document_prompt_names[0]
    
    print(f"\nUsing document prompt: {document_prompt}")

    # Create document mapper
    document_mapper = LLMDocumentMapper(
        model_name="mistral-small-2506",
        prompt_name=document_prompt
    )

    # Create pipeline
    pipeline = NormaPipeline(
        mode=ProcessingMode.DOCUMENT,
        document_mapper=document_mapper
    )

    # Process entire document
    print(f"\nProcessing document with {len(sentences)} sentences...")
    predicted_rules = pipeline.process(sentences, context=context)
    print(f"Extracted {len(predicted_rules)} denial constraints")

    # Display extracted constraints
    print("\n" + "=" * 80)
    print(f"EXTRACTED CONSTRAINTS (first 5 of {len(predicted_rules)})")
    print("=" * 80)
    
    for i, dc in enumerate(predicted_rules[:5], 1):
        print(f"\n[{i}] {dc}")
        print(f"    Predicates: {len(dc)}")
    
    if len(predicted_rules) > 5:
        print(f"\n... {len(predicted_rules) - 5} more")

    # Evaluate using MappingEvaluator
    print("\n" + "=" * 80)
    print("EVALUATION (AGGREGATE EXECUTION METRICS)")
    print("=" * 80)
    print("\nNote: Document mode produces rules without sentence mapping.")
    print("Only aggregate execution metrics can be computed.")
    
    evaluator = MappingEvaluator(ground_truth=sentence2dcs_map, df=df)
    metrics = evaluator.evaluate_document_level(
        predicted_rules=predicted_rules,
        gold_rules=gt_dcs
    )

    # Display metrics
    print("\nGlobal Execution Metrics (Union-Based)")
    print(f"  Union Precision:    {metrics['union_precision']:.3f}")
    print(f"  Union Recall:       {metrics['union_recall']:.3f}")
    print(f"  Union F1:           {metrics['union_f1']:.3f} [PRIMARY]")
    print(f"  Gold violations:    {metrics['total_gold_violations']}")
    print(f"  Pred violations:    {metrics['total_pred_violations']}")
    print(f"  Overlap violations: {metrics['total_overlap_violations']}")
    
    # Coverage analysis
    print("\nCoverage Analysis")
    coverage = metrics['union_recall']
    precision = metrics['union_precision']
    print(f"  Coverage (recall):  {coverage:.3f}")
    print(f"  Precision:          {precision:.3f}")
    
    # Constraint count analysis
    ratio = metrics['num_pred_rules'] / metrics['num_gold_rules'] if metrics['num_gold_rules'] > 0 else 0.0
    print(f"\nConstraint Count")
    print(f"  Ground truth:  {metrics['num_gold_rules']}")
    print(f"  Predicted:     {metrics['num_pred_rules']}")
    print(f"  Ratio:         {ratio:.2f}")
    
    if ratio < 0.5:
        print(f"  Warning: Predicted far fewer rules than ground truth")
    elif ratio > 2.0:
        print(f"  Warning: Predicted far more rules than ground truth")
    else:
        print(f"  Constraint count in reasonable range")

    # Close DB
    evaluator.close()

    print("\n" + "=" * 80)
    print("NORMA DOCUMENT evaluation complete")
    print("=" * 80)
    print(f"\nKey Metric:")
    print(f"  Union F1: {metrics['union_f1']:.3f}")
    
    # Comparison insight
    print("\nNote:")
    print("  Document-level approach trades per-sentence accuracy for")
    print("  potentially better global coverage of violations.")


def test_norma_sentence_filtered_evaluation():
    """Test NORMA SENTENCE_FILTERED pipeline with full evaluation"""
    from src.pipelines.norma_pipeline import NormaPipeline, ProcessingMode
    from src.classifiers.llm import LLMRuleClassifier
    from src.mappers.llm import LLMMapper
    from src.evaluation.mapping import MappingEvaluator
    from src.data import DatasetLoader
    from src.utils.table_context import TableContextBuilder
    from src.llm.helper import PromptLoader
    import os

    print("\nTesting NORMA SENTENCE_FILTERED Pipeline with Evaluation")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    # Load data
    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(f"Loaded {len(sentences)} sentences, {len(df)} rows")
    print(f"Ground truth rules: {sum(1 for dc in sentence2dcs_map.values() if dc)}")

    # Build table context
    context_builder = TableContextBuilder(sample_strategy="head", num_rows=3)
    context = context_builder.build(df)

    # Get prompts
    prompt_loader = PromptLoader()
    classification_prompt_names = prompt_loader.list_prompts("classification")
    mapping_prompt_names = prompt_loader.list_prompts("mapping")

    # Use first prompts
    classification_prompt = classification_prompt_names[0]
    mapping_prompt = mapping_prompt_names[0]

    print(f"\nUsing classification prompt: {classification_prompt}")
    print(f"Using mapping prompt: {mapping_prompt}")

    # Load prompt template for classifier
    classification_prompt_template = prompt_loader.load("classification", classification_prompt)
    
    # Create components
    classifier = LLMRuleClassifier(
        model_name="mistral-small-2506",
        prompt_template=classification_prompt_template
    )
    
    mapper = LLMMapper(
        model_name="mistral-small-2506",
        prompt_name=mapping_prompt,
        allowed_predicates=predicate_space['predicates'],
        chunk_size=10
    )

    # Create pipeline
    pipeline = NormaPipeline(
        mode=ProcessingMode.SENTENCE_FILTERED,
        classifier=classifier,
        mapper=mapper
    )

    # Process sentences
    print(f"\nProcessing {len(sentences)} sentences through SENTENCE_FILTERED pipeline...")
    results = pipeline.process(sentences, context=context)
    print("Pipeline processing complete")

    # Evaluate with DB execution
    print("\n" + "=" * 80)
    print("EVALUATION WITH DB EXECUTION")
    print("=" * 80)
    
    # Convert pipeline results format (sentence, is_rule, preds) to evaluator format (sentence, preds)
    predictions = [(sent, preds) for sent, is_rule, preds in results]
    
    evaluator = MappingEvaluator(ground_truth=sentence2dcs_map, df=df)
    metrics, sentence_results = evaluator.evaluate(predictions)

    # Display metrics
    print("\nDetection Metrics (All Sentences)")
    print(f"  Precision: {metrics.detection_precision:.3f}")
    print(f"  Recall:    {metrics.detection_recall:.3f}")
    print(f"  F1:        {metrics.detection_f1:.3f}")
    print(f"  Confusion: TP={metrics.detection_tp}, FP={metrics.detection_fp}, "
          f"FN={metrics.detection_fn}, TN={metrics.detection_tn}")

    print(f"\nSyntactic Correctness (Gold Rules Only, n={metrics.strict_total})")
    print(f"  Exact Match:       {metrics.strict_accuracy:.3f} ({metrics.strict_correct}/{metrics.strict_total})")
    print(f"  Predicate Jaccard: {metrics.avg_predicate_jaccard:.3f}")
    print(f"  Soft Jaccard:      {metrics.avg_soft_jaccard:.3f}")

    print("\nExecution Correctness - Per-Rule (Gold Rules Only)")
    print(f"  Exact Match:       {metrics.execution_accuracy:.3f} ({metrics.execution_correct}/{metrics.execution_total}) [PRIMARY]")
    print(f"  Execution Jaccard: {metrics.avg_execution_jaccard:.3f} [PRIMARY]")
    print(f"  Avg Precision:     {metrics.avg_execution_precision:.3f}")
    print(f"  Avg Recall:        {metrics.avg_execution_recall:.3f}")
    print(f"  Avg F1:            {metrics.avg_execution_f1:.3f}")

    print("\nGlobal Execution Metrics (Union-Based)")
    print(f"  Union Precision:    {metrics.union_precision:.3f}")
    print(f"  Union Recall:       {metrics.union_recall:.3f}")
    print(f"  Union F1:           {metrics.union_f1:.3f}")
    print(f"  Gold violations:    {metrics.total_gold_violations}")
    print(f"  Pred violations:    {metrics.total_pred_violations}")
    print(f"  Overlap violations: {metrics.total_overlap_violations}")

    # Show gap between syntactic and execution accuracy
    print("\nInsight: Syntactic vs Execution Gap")
    gap = metrics.execution_accuracy - metrics.strict_accuracy
    print(f"  Execution Accuracy - Exact Match = {gap:.3f}")
    if gap > 0:
        print(f"  {gap*100:.1f}% of rules are semantically correct despite syntactic differences")
    
    # Show example results
    print("\n" + "=" * 80)
    print("EXAMPLE RESULTS (First 5 Gold Rules)")
    print("=" * 80)
    
    gold_results = [r for r in sentence_results if r.gold_is_rule][:5]
    for i, r in enumerate(gold_results, 1):
        print(f"\n[{i}] {r.sentence[:70]}...")
        print(f"    Gold:      {r.gold_rule}")
        print(f"    Predicted: {r.pred_rule}")
        print(f"    Detection: {r.detection_status}")
        print(f"    Strict:    {r.strict_correct}")
        print(f"    Jaccard:   {r.predicate_jaccard:.3f}")
        if r.execution_exact is not None:
            print(f"    Exec Exact: {r.execution_exact}")
            print(f"    Exec Jacc:  {r.execution_jaccard:.3f}")
            print(f"    Violations: Gold={len(r.gold_violations)}, Pred={len(r.pred_violations)}, "
                  f"Overlap={len(r.gold_violations & r.pred_violations)}")

    # Show examples by detection status
    print("\n" + "=" * 80)
    print("EXAMPLES BY DETECTION STATUS")
    print("=" * 80)
    
    for status in ['TP', 'FP', 'FN', 'TN']:
        examples = [r for r in sentence_results if r.detection_status == status]
        if examples:
            print(f"\n{status}: {len(examples)} total (first 2 shown)")
            for r in examples[:2]:
                print(f"  {r.sentence[:60]}...")
                print(f"    Gold: {r.gold_is_rule}, Pred: {r.pred_is_rule}")

    # Close DB connection
    evaluator.close()

    print("\n" + "=" * 80)
    print("NORMA SENTENCE_FILTERED evaluation complete")
    print("=" * 80)
    print(f"\nKey Metrics Summary:")
    print(f"  Detection F1:        {metrics.detection_f1:.3f}")
    print(f"  Execution Accuracy:  {metrics.execution_accuracy:.3f}")
    print(f"  Execution Jaccard:   {metrics.avg_execution_jaccard:.3f}")
    print(f"  Union F1:            {metrics.union_f1:.3f}")

if __name__ == "__main__":

    test_classification_evaluation()


    # test_norma_sentence_filtered_evaluation()


    test_norma_document_evaluation()
