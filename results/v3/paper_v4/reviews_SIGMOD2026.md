# Reviews — Paper 473 (SIGMOD 2026, Research Round 2)

**Paper Title:** Text-to-Constraints: Extracting Data Quality Constraints from Regulatory Text
**Track:** Research Round 2

## Resumo das avaliações

| Reviewer | Overall Evaluation | Novelty | Importance | Top-3 críticos |
|----------|--------------------|---------|------------|----------------|
| **R1** | Reject | DI&DS: New data challenges from real-world applications or another field | SIGMOD attendees will learn something interesting | S1; O5; O7 |
| **R2** | Major Revision (accept pós-revisão não é certo) | Novel solution | Likely to influence other research | O2, O3, O5 |
| **R3** | Reject | Novel problem | No impact expected | O1–O7 (todos) |

Artefatos: disponíveis (confirmado pelos 3 revisores).

---

# Reviewer #1 — Regular

**1. Overall Evaluation:** Reject
**3. Novelty:** DI&DS: New data challenges from real-world applications or another field
**4. Importance:** SIGMOD attendees will learn something interesting from the paper

## 5. Summary of contribution
The paper presents Norma that identifies requirements from regulatory documents and maps them to denial constraints based on a predicate vocabulary. While sharing similarity to text-to-SQL work, the challenges of translating text to declarative constraints requires identifying passages that contain relevant conditions, context and specific thresholds or exception clauses. While the paper has some nice ideas and tackles an important and relevant problem, the technical depth is lacking and raises questions about research novelty.

## 6. Strong points
- **S1:** Extracting rule-based conditions from text is a practical and relevant problem, requiring context-based solutions to identify the right criteria.
- **S2:** Annotated datasets and constraints for reproducibility.
- **S3:** Paper is generally easy to understand with clear examples.

## 7. Opportunities for improvement
- **O1:** The introduction should place the work more deeply with respect to the state-of-the-art beyond just text-to-SQL approaches. How are existing declarative rules such as REEs related which use ML-predicates? How can existing DC-discovery methods be extended? There are no specific discussion in the intro with related work references.
- **O2:** The 3rd challenge in the intro was not clear and does not describe a technical limitation, especially if the work is already using denial constraints - what other constraint language?
- **O3:** How are these constraints re-usable in light of dynamic data?
- **O4:** The argument that LLM-based text-to-constraint extraction is not traceable is not clear to me. Why can't there be an iterative procedure that summarizes the relevant and similar sentences from text, and maps these to a DC? I don't understand the technical limitations of using an LLM to at least perform the preprocessing extraction and then possibly have a more fine-grained summarization that interprets similar sentences before mapping to rules?
- **O5:** While the paper is generally easy to understand, it lacks technical depth in several areas as challenges, novelty over related work, and methodology are described at a high level raising questions about research novelty, generalizability.
- **O6:** Why use SQL as the validation method when there are DC methods to test constraint satisfaction?
- **O7:** LLMs are used for sentence extraction in Section 4 but there were arguments against this in earlier sections due to lack of traceability, so how is the latter handled? If table schema and training examples are optional, how is context interpreted during the extraction? The classification methodology is rather light on details, and it's not clear how generalizable this method is beyond the presented example.
- **O8:** The brief discussion on cross-tuple constraints introduces the possibility of a different class of constraints such as differential dependences but it is not clear how such conditions that hold across all or majority of tuples are found, and how validation occurs in the framework.
- **O9:** Given one constraint per sentence, how does a user judge utility given the possibly large return set?
- **O10:** Evaluated datasets are rather small with respect to the number of sentences. There is no description of sentence similarity, redundancy among sentences, and how ground truth constraints are computed. The paper states the SISU data has ordering constraints and conditional selection rules, what types of constraints are these? CFDs, ODs?
- **O11:** The paper may be better suited in a data science or applied track that uses existing methods and models to tackle a real-world problem. For example, Figure 6 (evaluation across FMs) provides opportunities to discuss comparative performance, and model sensitivity.
- **O12:** If norma maps to a broader set of constraints, as referenced to ordering or conditional selection rules, then a broader set of baselines should be evaluated beyond just DC algorithms.

**8. Top-3 críticos:** S1; O5; O7
**9. Questions to authors:** N/A
**11. Required changes for revision:** NA
**12. Artifacts:** Yes

---

# Reviewer #2 — Regular

**1. Overall Evaluation:** Major Revision - Many revisions required and an "accept" post revision is not certain
**3. Novelty:** Novel solution
**4. Importance:** The paper is likely to influence other research in the community

## 5. Summary of contribution
The paper aims to solve the challenge of extracting executable data quality constraints from natural-language regulatory documents and grounding them in a relational dataset schema. It proposes Norma, a framework that segments documents at the sentence level, uses an LLM-based classifier to identify normative sentences, and maps them to denial constraints through a multi-step pipeline covering schema linking, predicate selection, rule filtering, and constraint construction over a schema-derived predicate space. The framework is evaluated on four datasets along three dimensions, constraint classification, constraint extraction, and violation detection, and is compared against denial-constraint discovery and a Text-to-SQL baseline, showing that it recovers regulatory rules that data-driven methods miss.

## 6. Strong points
- **S1:** The paper aims to address an interesting problem of extracting data quality constraints from regulatory text, which is relevant to compliance checking and data validation in real-world settings where rules are documented only in natural language.
- **S2:** It proposes a method that integrates sentence-level classification, schema-grounded predicate selection, and multi-step constraint construction, with a controlled predicate vocabulary that avoids hallucinated columns and unsupported operators.
- **S3:** Experimental results on four datasets across six LLM backbones and three complementary metrics demonstrate consistent gains over data profiling and Text-to-SQL baselines.

## 7. Opportunities for improvement
- **O1:** The paper introduces the Brazilian SISU case in Section 1 and provides a SQL query that checks Article 19, but does not report how many tuples in the released SISU data actually violate this rule. Given that the entire framing rests on the claim that published datasets silently violate their governing regulations, an empirical number for the motivating example itself would have made the motivation more credible and is a missed opportunity.
- **O2:** The pipeline is thoughtfully assembled, though the core contribution currently leans heavily on prompt design. To articulate the technical novelty of the paper, it is beneficial to explicitly dedicate a subsection to discuss the methodological novelty. Highlighting specific structural or theoretical innovations beyond the LLM integration would make the contribution much clearer to the reader.
- **O3:** While the Text-to-Constraints formulation is new in framing constraints as declarative invariants rather than queries, the underlying technical machinery including schema linking, predicate selection decomposition, controlled vocabulary generation and post-construction validation overlaps heavily with established Text-to-SQL practice (RESDSQL [22], DIN-SQL [36]), and the contribution list would benefit from explicitly delineating which components are inherited adaptations versus genuinely novel mechanisms.
- **O4:** While the six predicate categories in Section 5.1 form a solid foundation, the paper lacks an analysis demonstrating their sufficiency for real-world regulatory text. The exclusion of aggregations and temporal arithmetic creates a practical bottleneck, as evidenced by the need to manually derive a "days-until-next-inspection" column for the Restaurant dataset. Adding a coverage analysis against the gold constraints and explicitly discussing the expressivity limits of this vocabulary would significantly strengthen the methodology.
- **O5:** The four-stage mapper (schema linking, predicate selection, rule filter, constraint building) is reasonable but not fully justified. Why four stages? The Single-Prompt vs Multi-Step comparison is useful, but a finer ablation that removes or reorders stages would clarify whether the gains come from the decomposition or just from more LLM calls.

### Experiment
- **O6.1:** The experimental datasets do not fully reflect the scalability of the solution. Hospital and Tax rely on synthetically generated text, and the Restaurant dataset is quite brief at only 33 sentences. Consequently, SISU is the only dataset that truly resembles the target real-world setting. Acknowledging this limitation and discussing how the framework might scale to typical legal documents with hundreds of sentences and nested clauses would significantly strengthen the generalizability of the claims.
- **O6.2:** Tax and Hospital ground-truth constraints are turned into natural language by an LLM and surrounded with LLM-generated distractor sentences (Section 6.1), and although Claude 4.5 Sonnet is used to avoid evaluating on the same model, all six evaluated backbones share the LLM family's stylistic and conceptual priors with the generator, so high scores on these datasets (Hospital F1=1.0) likely overstate real regulatory NLP difficulty. The more telling results are on SISU and Restaurant, where all methods struggle around F1 ≈ 0.5, and the paper would benefit from making this asymmetry explicit.

**8. Top-3 críticos:** O2, O3, O5 (texto original: "O2.3.5")
**9. Questions to authors:** "Please kindly all."
**11. Required changes for revision:** "Please kindly all."
**12. Artifacts:** Yes

---

# Reviewer #3 — Regular

**1. Overall Evaluation:** Reject
**3. Novelty:** Novel problem
**4. Importance:** No impact expected

## 5. Summary of contribution
The authors propose methods to extract rules (written in natural language) from text documents that are then converted into denial constraints that must hold on an input table. This is definitely an interesting and novel problem, however, the entire solution is based on LLM prompting which greatly narrows the technical contribution.

## 6. Strong points
- **S1:** Good and timely problem: discovering denial constraints from text documents is a novel problem.
- **S2:** The authors showed real-world examples of tuples violating rules that are mentioned in regularly text in natural language.

## 7. Opportunities for improvement
- **O1:** This statement is not entirely true: *"However, a fundamental assumption underlying this line of work is that constraints must be inferred from data, and thus, they are limited to discovering constraints that reflect what the data happens to satisfy, not what it is required to satisfy"*. Those are expert-provided rules, and there has been a lot of work on guiding domain experts to discover and refine those rules. Now, it's not true that existing methods only discover rules that are satisfied: many approaches discover rules that approximately hold on the data, e.g., *Discovery of Approximate (and Exact) Denial Constraints. VLDB 2019*.
- **O2:** While the authors did a good job motivating for the problem, the underlying techniques are simply LLM prompts to address different stages of the pipeline. This greatly limits the rigor of the technique as the LLM is doing all the heavy lifting which makes it unclear what the technical contribution of the paper is.
- **O3:** The evaluation setup has a few problems: for the Tax dataset, the authors use the a paper that used this dataset as the "regulatory text input". Same goes for the Hospital dataset. The problem here is that the accompanying papers already state the (best) rules that are expected in those datasets. Therefore, this setup is quite contrived. The other two datasets have much more realistic table-text pairs.
- **O4:** The evaluation results show that the classifier (the LLM) can do a decent job even without context or examples. For instance, for GPT, the improvements brought by the inclusion of context and examples are marginal for many of the datasets. The result analysis did not shed light onto why that is the case. I am assuming it's because GPT already picked up knowledge about these datasets and the regulatory text in its pre-training.
- **O5:** Many parts of the text read poorly using vague and non-technical language. For example: *"Because Norma operates over text instead of searching for regularities in the data"*: what does "regularities" mean here? I am assuming it means "consistency with respect to the rules", but this has to be spelled out precisely to the reader. Another example where the authors refer to a component that hadn't been defined yet: *"False positives can still be ruled out by the mapper, which applies a more expressive classification strategy internally."* The reader at this point doesn't know what the mapper is and what it does. Therefore, saying that it applies a "more expressive classification strategy internally" is confusing for someone who doesn't know what the mapper is. Perhaps a better way to state this is: *"False positives can still be ruled out by the mapper (which we define next)."*
- **O6:** The authors state: *"A third challenge is the output formalism, and there is an unavoidable compromise. General SQL is expressive, but it is a moving target for extraction, making it difficult to standardize checks across documents."* I am not sure I understand what this passage means, why is SQL a "moving target" for extraction? This is also confusing, because we can express any denial constraint using SQL, so is the output not in SQL? (this is contradicted later on in the paper where the output is a denial constraint which can be written as a SQL query).
- **O7:** The authors claim that denial constraints don't support the predicates they are looking for: *"Our setting requires a richer predicate space. Regulatory text frequently refers to explicit literals, enumerations, ranges, nullability conditions, and string formats that are not covered by traditional DC discovery spaces."*, but this is not true: one can check for nulls using denial constraints, same goes for referring to explicit literals/constants. For ranges/sets, it's indeed true that one needs to extend the original predicate space to support them.

**8. Top-3 críticos:** O1–O7 (texto original: "O1-O7" — todos)
**9. Questions to authors:** N/A
**11. Required changes for revision:** N/A
**12. Artifacts:** Code and data available

---

## Mapeamento reviews → notas `\altic{}` já no texto

| Objeção | Onde está endereçada (`\altic`) |
|---------|--------------------------------|
| R1-O1 | `intro.tex:72`, `related_work.tex:6` |
| R1-O2 | `intro.tex:66` |
| R1-O3 | `conclusion.tex:10` |
| R1-O4 | `overview.tex:11` |
| R1-O5 | (geral — coberta por R2-O2/O3 em `mapping.tex:18`) |
| R1-O6 | `overview.tex:46` |
| R1-O7 | `evaluation_classification.tex:22` + `overview.tex:11` |
| R1-O8 | `mapping.tex:119` |
| R1-O9 | `evaluation_end2end.tex:101` |
| R1-O10 | `evaluation_base.tex:58` |
| R1-O11 | `evaluation_mapping.tex:48` |
| R1-O12 | `evaluation_base.tex:92` |
| R2-O1 | `intro.tex:44` |
| R2-O2, O3 | `mapping.tex:18` (⚠️ crítico, 3 revisores) |
| R2-O4 | `mapping.tex:45` |
| R2-O5 | `mapping.tex:141` |
| R2-O6.1 | `conclusion.tex:9` |
| R2-O6.2 | `evaluation_base.tex:85`, `evaluation_classification.tex:22` |
| R3-O1 | `intro.tex:9`, `related_work.tex:6` |
| R3-O2 | `mapping.tex:18` |
| R3-O3 | `evaluation_base.tex:85` |
| R3-O4 | `evaluation_classification.tex:22` |
| R3-O5 | `intro.tex:83`, `overview.tex:35` |
| R3-O6 | `intro.tex:66` |
| R3-O7 | `mapping.tex:32` |
