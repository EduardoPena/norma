# test.py - Quick migration testing


DATA_PATH = 'data/person/'  # Adjust as needed


def test_load():
    """Test predicates + data loading"""
    from src.predicates import Predicate, PredicateSpace
    from src.data import DatasetLoader

    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(df.head())

    print(f"\nLoaded {len(sentences)} sentences and {len(df)} rows")
    print(
        f"Predicate space has {len(predicate_space['predicates'])} predicates")
    print(f"\nFirst 10 predicates:")
    for pred in predicate_space['predicates'][:10]:
        print(f"  {pred}")

    print(f"\nFirst 5 sentences:")
    for s in sentences[:5]:
        print(s)

    print(f"\nGround truth mappings:")
    for k, v in sentence2dcs_map.items():
        print(f"Sentence: {k}\nGround Truth: {v}\n")


def test_llm_clients():
    """Test each supported LLM with a simple prompt"""
    from src.llm.client import get_llm_client
    import os

    print("Testing LLM Clients")

    models = [
        ("gpt-5", "OPENAI_API_KEY"),
        ("claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY"),
        ("mistral-small-2506", "MISTRAL_API_KEY"),
        ("llama-3.3-70b-versatile", "GROQ_API_KEY"),
    ]

    for model_name, env_var in models:
        print(f"\nTesting {model_name}...")

        if not os.getenv(env_var):
            print(f"    Skipping - {env_var} not set")
            continue

        try:
            client = get_llm_client(model_name)
            response = client.complete("hi")
            print(f"    Response: {response[:100]}...")
        except Exception as e:
            print(f"    Error: {e}")


def test_sql():
    """Test SQL translation + execution for a few ground-truth denial constraints."""
    from src.data import DatasetLoader
    from src.dbms import DBExecutor
    from src.dbms.sql_translator import SQLTranslator

    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    db = DBExecutor(df)  # default table_name="data_table"
    tr = SQLTranslator(db)  # uses db.table_name + db.tid_column

    # Quick smoke test: DB works
    t = db.table_name
    print(db.execute_sql(f"SELECT COUNT(*) FROM {t}"))

    print("\n--- Translation + violation execution (first 5 GT rules) ---")

    tested = 0
    for sentence, rule in sentence2dcs_map.items():
        # Skip non-rules / empty
        if not rule:
            continue

        tested += 1
        print(f"\nSentence: {sentence}")
        print(f"Rule: {rule}")

        translated = tr.translate(rule)
        assert translated["success"], f"Translation failed: {translated.get('error')}"
        sql = translated["sql"]
        print("SQL:", sql)

        result = db.execute_violation_query(sql)
        assert result["success"], f"SQL execution failed: {result.get('error')}"
        print("violated:", result["constraint_violated"],
              "row_count:", result["row_count"])

        # show up to first 5 violations
        if result["rows"]:
            print("first_rows:", result["rows"][:5])

        if tested >= 5:
            break

    db.close()
    print(f"\nOK: tested {tested} rule(s)")


def test_classifier():
    """Test LLM-based classifier with all available prompts."""
    from src.classifiers.llm import LLMRuleClassifier
    from src.data import DatasetLoader
    from src.llm.helper import PromptLoader
    from src.utils.table_context import TableContextBuilder
    import os

    print("\nTesting LLM Classifier")

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(f"Loaded {len(sentences)} sentences")

    # Get all available classification prompts
    prompt_loader = PromptLoader()
    prompt_names = prompt_loader.list_prompts("classification")
    print(f"Testing {len(prompt_names)} prompts: {prompt_names}")

    test_sentences = sentences[:5]

    # Build table context using helper
    context_builder = TableContextBuilder(sample_strategy="head", num_rows=3)
    context = context_builder.build(df)

    # Show truncated context
    print(f"\nTable context for LLM:\n{context['table_context'][:500]}...")

    for prompt_name in prompt_names:
        print(f"\n--- Prompt: {prompt_name} ---")
        prompt_template = prompt_loader.load("classification", prompt_name)
        classifier = LLMRuleClassifier(
            model_name="mistral-small-2506", prompt_template=prompt_template)

        results = classifier.classify(test_sentences, context=context)

        for sentence, is_rule in results:
            label = "RULE" if is_rule else "NOT"
            print(f"[{label}] {sentence[:60]}...")

    print("\nOK")


def test_mapper():
    """Test LLM-based mapper with ground truth sentences."""
    from src.mappers.llm import LLMMapper
    from src.data import DatasetLoader
    from src.llm.helper import PromptLoader
    from src.utils.table_context import TableContextBuilder
    import os

    print("\nTesting LLM Mapper")
    print("-" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    # Load data
    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(f"Loaded {len(sentences)} sentences, {len(df)} rows")
    print(f"Predicate space: {len(predicate_space['predicates'])} predicates")

    # Build table context
    context_builder = TableContextBuilder(sample_strategy="head", num_rows=3)
    context = context_builder.build(df)
    print(f"\nTable context preview:\n{context['table_context']}...\n")

    # Get all available mapping prompts
    prompt_loader = PromptLoader()
    prompt_names = prompt_loader.list_prompts("mapping")
    print(f"Available prompts: {prompt_names}\n")

    # prompt_names = prompt_names[:1]  # Test first prompt only for brevity

    # Filter to only sentences with ground truth rules
    rule_sentences = [s for s, dc in sentence2dcs_map.items() if dc]
    print(f"Found {len(rule_sentences)} sentences with ground truth rules")

    # Test on first 5 rule sentences
    test_sentences = rule_sentences[:5]

    # Test each prompt
    for prompt_name in prompt_names:
        print(f"\n{'=' * 60}")
        print(f"Testing Prompt: {prompt_name}")
        print('=' * 60)

        # Create mapper
        mapper = LLMMapper(
            model_name="gpt-4o-mini",
            prompt_name=prompt_name,
            allowed_predicates=predicate_space['predicates'],
            chunk_size=10
        )

        # Map sentences
        results = mapper.map(test_sentences, context=context)

        # Display results
        for sentence, predicted_dc in results:
            ground_truth_dc = sentence2dcs_map.get(sentence, None)

            print(f"\nSentence: {sentence}")
            print(f"Ground Truth: {ground_truth_dc}")
            print(f"Predicted:    {predicted_dc}")

            # Check if prediction matches ground truth structure
            if predicted_dc is None:
                print("Status: ✗ Returned null (not a rule)")
            elif ground_truth_dc and len(predicted_dc) > 0:
                print(f"Status: ✓ Mapped to {len(predicted_dc)} predicates")
            else:
                print("Status: ⚠ Mapped but unexpected structure")

    print("\n" + "=" * 60)
    print("Mapper test complete!")


def test_mapper_all():
    """Test LLM-based mapper with ground truth sentences."""
    from src.mappers.llm import LLMMapper
    from src.data import DatasetLoader
    from src.llm.helper import PromptLoader
    from src.utils.table_context import TableContextBuilder
    import os

    print("\nTesting LLM Mapper on all sentences")
    print("-" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    # Load data
    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(f"Loaded {len(sentences)} sentences, {len(df)} rows")
    print(f"Predicate space: {len(predicate_space['predicates'])} predicates")

    # Build table context
    context_builder = TableContextBuilder(sample_strategy="head", num_rows=3)
    context = context_builder.build(df)
    print(f"\nTable context preview:\n{context['table_context']}...\n")

    # Get all available mapping prompts
    prompt_loader = PromptLoader()
    prompt_names = prompt_loader.list_prompts("mapping")
    print(f"Available prompts: {prompt_names}\n")

    prompt_names = prompt_names[:1]  # Test first prompt only for brevity

    # Filter to only sentences with ground truth rules
    rule_sentences = [s for s, dc in sentence2dcs_map.items() if dc]

    non_rule_test_sentences = [
        sentence for sentence in sentences if sentence not in sentence2dcs_map]

    # Test on first 5 rule sentences
    test_sentences = rule_sentences[:5] + non_rule_test_sentences[:5]

    # Test each prompt
    for prompt_name in prompt_names:
        print(f"\n{'=' * 60}")
        print(f"Testing Prompt: {prompt_name}")
        print('=' * 60)

        # Create mapper
        mapper = LLMMapper(
            # model_name="gpt-5-mini",
            model_name="mistral-small-2506",
            prompt_name=prompt_name,
            allowed_predicates=predicate_space['predicates'],
            chunk_size=10
        )

        # Map sentences
        results = mapper.map(test_sentences, context=context)

        # Display results
        for sentence, predicted_dc in results:
            ground_truth_dc = sentence2dcs_map.get(
                sentence, 'Non rule sentence')

            print(f"\nSentence: {sentence}")
            print(f"Ground Truth: {ground_truth_dc}")
            print(f"Predicted:    {predicted_dc}")

            # Check if prediction matches ground truth structure
            if predicted_dc is None:
                print("Status: ✗ Returned null (not a rule)")
            elif ground_truth_dc and len(predicted_dc) > 0:
                print(f"Status: ✓ Mapped to {len(predicted_dc)} predicates")
            else:
                print("Status: ⚠ Mapped but unexpected structure")

    print("\n" + "=" * 60)
    print("Mapper test complete!")


def test_norma_sentence_direct():
    """Test the full Norma pipeline in SENTENCE_DIRECT mode."""
    from src.pipelines.norma_pipeline import NormaPipeline, ProcessingMode
    from src.mappers.llm import LLMMapper
    from src.data import DatasetLoader
    from src.utils.table_context import TableContextBuilder
    import os
    from src.llm.helper import PromptLoader

    print("\nTesting Norma Pipeline - SENTENCE_DIRECT mode")
    print("-" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    # Load data
    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(f"Loaded {len(sentences)} sentences, {len(df)} rows")
    print(f"Predicate space: {len(predicate_space['predicates'])} predicates")

    # Build table context
    context_builder = TableContextBuilder(sample_strategy="head", num_rows=3)
    context = context_builder.build(df)
    print(f"\nTable context preview:\n{context['table_context']}...\n")

    prompt_loader = PromptLoader()
    mapping_prompt_names = prompt_loader.list_prompts("mapping")
    # Test first prompt only for brevity
    mapping_prompt = mapping_prompt_names[0]
    print(
        f"Available prompts: {mapping_prompt_names}, but using {mapping_prompt}\n")

    mapper = LLMMapper(
        # model_name="gpt-5-mini",
        model_name="mistral-small-2506",
        prompt_name=mapping_prompt,
        allowed_predicates=predicate_space['predicates'],
        chunk_size=10
    )
    # Initialize Norma pipeline in SENTENCE_DIRECT mode
    pipeline = NormaPipeline(
        mode=ProcessingMode.SENTENCE_DIRECT, mapper=mapper)

    # Process sentences through the pipeline
    results = pipeline.process(sentences, context=context)

    # Save results
    results_dir = "results/test"
    os.makedirs(results_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(DATA_PATH))
    result_file = os.path.join(results_dir, f"test_norma_sentence_direct_results_{dataset_name}.csv")

    print("\nSaving results to results/test...")
    pipeline.save(results, output_file=result_file, ground_truth=sentence2dcs_map)
    print(f"✓ Results saved to {result_file}")

    # Display results for first 5 sentences
    print("\nFirst 5 results:")
    for sentence, is_rule, predicates in results[:5]:
        gt_dc = sentence2dcs_map.get(sentence, 'Non rule sentence')
        print(f"\nSentence: {sentence}")
        print(f"Ground Truth: {gt_dc}")
        print(f"Is Rule: {is_rule}")
        print(f"Predicates: {predicates}")

    print("\nNorma Pipeline test complete!")

def test_norma_sentence_filtered():
    """Test the full Norma pipeline in SENTENCE_FILTERED mode."""
    from src.pipelines.norma_pipeline import NormaPipeline, ProcessingMode
    from src.classifiers.llm import LLMRuleClassifier
    from src.mappers.llm import LLMMapper
    from src.data import DatasetLoader
    from src.utils.table_context import TableContextBuilder
    import os
    from src.llm.helper import PromptLoader

    print("\nTesting Norma Pipeline - SENTENCE_FILTERED mode")
    print("-" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    # Load data
    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(f"Loaded {len(sentences)} sentences, {len(df)} rows")
    print(f"Predicate space: {len(predicate_space['predicates'])} predicates")

    # Build table context
    context_builder = TableContextBuilder(sample_strategy="head", num_rows=3)
    context = context_builder.build(df)
    print(f"\nTable context preview:\n{context['table_context']}...\n")

    prompt_loader = PromptLoader()
    classification_prompt_names = prompt_loader.list_prompts("classification")
    mapping_prompt_names = prompt_loader.list_prompts("mapping")

    # Test first prompt only for brevity
    classification_prompt = classification_prompt_names[0]
    mapping_prompt = mapping_prompt_names[0]

    print(
        f"Available classification prompts: {classification_prompt_names}, but using {classification_prompt}\n")
    print(
        f"Available mapping prompts: {mapping_prompt_names}, but using {mapping_prompt}\n")

    # Load the actual prompt template for classifier
    classification_prompt_template = prompt_loader.load("classification", classification_prompt)
    
    classifier = LLMRuleClassifier(
        model_name="mistral-small-2506", prompt_template=classification_prompt_template)
    
    mapper = LLMMapper(
        model_name="mistral-small-2506",
        prompt_name=mapping_prompt,
        allowed_predicates=predicate_space['predicates'],
        chunk_size=10
    )

    # Initialize Norma pipeline in SENTENCE_FILTERED mode
    pipeline = NormaPipeline(
        mode=ProcessingMode.SENTENCE_FILTERED, classifier=classifier, mapper=mapper)

    # Process sentences through the pipeline
    results = pipeline.process(sentences, context=context)

    # Save results
    results_dir = "results/test"
    os.makedirs(results_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(DATA_PATH))
    result_file = os.path.join(results_dir, f"test_norma_sentence_filtered_results_{dataset_name}.csv")


    print("\nSaving results to results/test...")
    pipeline.save(results, output_file=result_file, ground_truth=sentence2dcs_map)
    print(f"✓ Results saved to {result_file}")


def test_norma_document_mapping():
    """Test the full Norma pipeline in DOCUMENT mode."""
    from src.pipelines.norma_pipeline import NormaPipeline, ProcessingMode
    from src.mappers.llm_document_mapper import LLMDocumentMapper
    from src.data import DatasetLoader
    from src.utils.table_context import TableContextBuilder
    import os
    from src.llm.helper import PromptLoader

    print("\nTesting Norma Pipeline - DOCUMENT mode")
    print("-" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping - OPENAI_API_KEY not set")
        return

    # Load data
    loader = DatasetLoader(DATA_PATH)
    df, sentences, sentence2dcs_map, predicate_space = loader.load()

    print(f"Loaded {len(sentences)} sentences, {len(df)} rows")
    print(f"Predicate space: {len(predicate_space['predicates'])} predicates")

    # Build table context
    context_builder = TableContextBuilder(sample_strategy="head", num_rows=3)
    context = context_builder.build(df)
    
    # Add predicates list to context (REQUIRED for document mapping)
    predicates_list_str = "\n".join(predicate_space['predicates'])
    context['predicates_list'] = predicates_list_str
    
    # print(f"\nTable context preview:\n{context['table_context'][:200]}...\n")
    # print(f"Predicates list preview:\n{predicates_list_str[:200]}...\n")

    # Get available document mapping prompts
    prompt_loader = PromptLoader()
    
    try:
        document_prompt_names = prompt_loader.list_prompts("document_mapping")
        print(f"Available document prompts: {document_prompt_names}\n")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        document_prompt_names = ["basic"]
    
    # Test with first prompt
    document_prompt = document_prompt_names[0]
    print(f"Using prompt: {document_prompt}\n")

    # Initialize document mapper
    document_mapper = LLMDocumentMapper(
        # model_name="mistral-small-2506",
        model_name="gpt-5-mini",
        prompt_name=document_prompt
    )

    # Initialize Norma pipeline in DOCUMENT mode
    pipeline = NormaPipeline(
        mode=ProcessingMode.DOCUMENT,
        document_mapper=document_mapper
    )

    # Process entire document (all sentences together)
    print(f"Processing document with {len(sentences)} sentences in single call...")
    results = pipeline.process(sentences, context=context)

    # Display extraction results
    print(f"\n{'=' * 60}")
    print(f"Extracted {len(results)} denial constraints from document")
    print('=' * 60)

    for i, dc in enumerate(results, 1):
        print(f"\nConstraint {i}:")
        print(f"  Predicates: {dc}")
        print(f"  Predicate count: {len(dc)}")

    # Compare with ground truth
    print(f"\n{'=' * 60}")
    print("Ground Truth Comparison")
    print('=' * 60)
    
    gt_dcs = [dc for dc in sentence2dcs_map.values() if dc]
    print(f"Ground truth constraints: {len(gt_dcs)}")
    print(f"Extracted constraints: {len(results)}")
    
    if gt_dcs:
        print("\nGround truth DCs (first 5):")
        for i, dc in enumerate(gt_dcs[:5], 1):
            print(f"  {i}. {dc}")

    # Save results
    results_dir = "results/test"
    os.makedirs(results_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(DATA_PATH))
    result_file = os.path.join(
        results_dir, 
        f"test_norma_document_{document_prompt}_{dataset_name}"
    )

    print(f"\nSaving results to {result_file}...")
    pipeline.save(results, output_file=result_file, ground_truth=sentence2dcs_map)
    print(f"✓ Results saved")

    print("\n" + "=" * 60)
    print("Document mapping test complete!")
    print(f"Total constraints extracted: {len(results)}")
    print(f"Sentences processed: {len(sentences)}")
    if len(sentences) > 0:
        print(f"Ratio: {len(results)/len(sentences):.2f} constraints per sentence")

if __name__ == "__main__":
    # test_load()
    # test_llm_clients()
    # test_sql()
    # test_classifier()
    # test_mapper()
    # test_mapper_all()
    # test_norma_sentence_direct()

    # test_norma_sentence_filtered()
    test_norma_document_mapping()