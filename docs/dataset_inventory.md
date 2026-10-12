# Dataset Inventory — Amazon ESCI Shopping Queries

**Project:** Context-Aware Agentic Search for Improved Product Discovery in E-Commerce  
**Purpose:** Document the dataset source, filtering logic, prepared query sample, and annotation artifacts.  
**Inventory status:** Initial preparation complete; manual annotation pilot pending review.

## 1. Dataset Source

The project uses the Amazon Shopping Queries Dataset, also known as the ESCI dataset. It contains shopping queries, associated products, and query–product relevance labels.

The relevance labels are:

- **E — Exact:** Exact product match
- **S — Substitute:** Substitute product
- **C — Complement:** Complementary product
- **I — Irrelevant:** Irrelevant product

These labels support product relevance evaluation. They are not query chunk labels or query intent labels.

## 2. Raw Files

| File | Reported size | Rows | Main purpose |
|---|---:|---:|---|
| `shopping_queries_dataset_examples.parquet` | ~51 MB | 2,621,288 | Query–product pairs and relevance judgements |
| `shopping_queries_dataset_products.parquet` | ~1.108 GB | 1,814,924 | Product metadata |
| `shopping_queries_dataset_sources.csv` | ~1.8 MB | Not recorded | Query source metadata |

### Examples file fields

`example_id`, `query`, `query_id`, `product_id`, `product_locale`, `esci_label`, `small_version`, `large_version`, `split`, `__index_level_0__`

### Products file fields

`product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, `product_locale`

## 3. Initial Dataset Counts

The examples file contains 2,621,288 query–product judgement rows.

### Locale distribution

| Locale | Rows |
|---|---:|
| US (`us`) | 1,818,825 |
| Japan (`jp`) | 446,053 |
| Spain (`es`) | 356,410 |
| **Total** | **2,621,288** |

### Split distribution

| Split | Rows |
|---|---:|
| Train | 1,983,272 |
| Test | 638,016 |
| **Total** | **2,621,288** |

These counts represent query–product rows, not unique queries.

## 4. Filtering and Query Preparation

The preparation script is `src/data/prepare_queries.py`.

The current process:

1. Reads the examples Parquet file.
2. Selects `query`, `query_id`, `product_locale`, and `split`.
3. Filters to `product_locale == "us"`.
4. Filters to `split == "train"`.
5. Checks whether a query ID maps to multiple query texts.
6. Deduplicates to one record per query ID.
7. Calculates query length for basic descriptive analysis.
8. Creates a reproducible sample using random seed `42`.

### Filtered dataset counts

| Stage | Rows or records |
|---|---:|
| All query–product rows | 2,621,288 |
| US-locale rows | 1,818,825 |
| US training query–product rows | 1,393,063 |
| Query IDs mapping to multiple query texts | 0 |
| Unique US training queries | 74,888 |
| Initial manual annotation sample | 150 |

The 1,393,063 figure is the number of query–product rows after filtering. The 74,888 figure is the number of unique training queries after deduplication.

## 5. Query Length Statistics

Query length was calculated for the 74,888 unique US training queries.

| Statistic | Value |
|---|---:|
| Count | 74,888 |
| Mean | 3.53 |
| Standard deviation | 1.63 |
| Minimum | 1 |
| 25th percentile | 2 |
| Median | 3 |
| 75th percentile | 4 |
| Maximum | 29 |

The query length unit and calculation method should be recorded in the preparation script. If length is measured in whitespace-separated words, state that explicitly.

Initial inspection suggests that most queries are short keyword searches, while a smaller number contain longer natural-language requirements. The latter are particularly relevant to testing context and constraint extraction.

## 6. Prepared Artifacts

| Artifact | Purpose |
|---|---|
| `data/processed/unique_queries_us_train.csv` | Deduplicated US training queries |
| `data/annotations/query_sample.csv` | Reproducible sample of 150 queries |
| `data/annotations/intent_annotations.csv` | Primary intent labels and review notes |
| `data/annotations/chunk_annotations.csv` | Query spans and chunk labels |

The sample contains 150 unique query IDs and no missing queries, according to the preparation run.

## 7. Project-Created Labels

The current annotation schema is project-defined.

### Chunk labels

`PRODUCT`, `ATTRIBUTE`, `PURPOSE`, `OCCASION`, `CONTEXT`, `CONSTRAINT`

### Intent labels

`PRODUCT_SEARCH`, `ATTRIBUTE_SEARCH`, `USE_CASE_SEARCH`, `OCCASION_SEARCH`, `CONTEXT_CONSTRAINT_SEARCH`

These labels are not supplied by the original ESCI dataset. They will be created through manual annotation and used for Blocks 1 and 2.

Record the annotation guideline version and any changes made after the pilot.

## 8. Data Quality and Limitations

- The current query sample is derived from the US training split only.
- Query–product relevance labels must not be mistaken for query intent labels.
- Query deduplication is performed by `query_id`; the preparation run found no query IDs associated with multiple query texts.
- The initial 150-query sample is intended for annotation design and pipeline development, not for a definitive performance claim.
- The distribution of intent labels is not yet established.
- The product file has been counted, but its full join coverage against the filtered examples has not yet been measured.
- Any additional context-rich or synthetic challenge queries must be identified separately from the original ESCI queries.
- The test split should remain separate from training and model-selection data when retrieval or classification evaluation is implemented.

## 9. Next Actions

1. Review 20–30 queries and finalize annotation guidelines.
2. Manually annotate a small pilot for both chunk spans and primary intent.
3. Record ambiguous cases and update the guideline version.
4. Implement the initial chunking and intent-classification baselines.
5. Compare raw-query, predicted-chunk, and gold-chunk experiments.
6. Record the annotation class distribution and the final train/validation/test splitting strategy.
7. Verify the literature gap before claiming novelty.

## 10. References

- Reddy, C. K., et al. (2022). *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search*. arXiv:2206.06588.
- Amazon Science. *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search*.
