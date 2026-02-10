# test.py - Quick migration testing

DATA_PATH = 'data/tax/'  # Adjust as needed


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


if __name__ == "__main__":
    # test_load()
    # test_llm_clients()
    test_sql()
