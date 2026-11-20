# LLM-Based Automated Answer Script Evaluator

**Rubric-Based Automated Grading of Subjective Answers Using Large Language Models**

An AI-powered subjective answer evaluation system that uses Large Language Models (LLMs) to assess student answers against predefined marking rubrics and compares automated scores with human evaluations.

---

## Project Overview

Evaluating subjective examination answers manually can be time-consuming and may involve differences in grading between evaluators.

This project explores the use of Large Language Models to automate subjective answer evaluation using question-specific rubrics, reference answers, and structured grading instructions.

The system uses **DeepSeek-R1:8B through Ollama** to evaluate student answers, assign marks, and generate feedback.

The research also compares LLM-generated scores with human-assigned scores to study the agreement between automated and manual evaluation.

## Objectives

- Develop an LLM-based subjective answer grading pipeline.
- Evaluate student answers using predefined marking rubrics.
- Generate marks and feedback automatically.
- Compare LLM scores with human evaluator scores.
- Measure grading performance using statistical evaluation metrics.
- Explore handwritten answer recognition and a teacher-facing grading dashboard.

## Project Workflow

```text
Question + Student Answer
          |
          v
    Question-Specific Rubric
          |
          v
   Reference Answer
          |
          v
     DeepSeek-R1:8B
       (Ollama)
          |
          v
   Automated Evaluation
          |
     +----+----+
     |         |
     v         v
   Marks    Feedback
     |
     v
Compare with Human Scores
     |
     v
Evaluation Metrics & Graphs
```

## Dataset

The project uses a labelled dataset containing approximately **500 records based on 100 questions**, with different answer-quality examples and marks.

Important dataset fields include:

| Column | Description |
|---|---|
| question_id | Unique question identifier |
| subject | Subject or topic |
| question | Examination question |
| max_score | Maximum marks |
| rubric | Question-specific marking criteria |
| reference_answer | Expected answer |
| student_answer | Student's response |
| quality_tier | Answer-quality category |
| grader1_score | First human evaluator's marks |
| grader2_score | Second human evaluator's marks |
| avg_human_score | Average human-assigned score |
| llm_score | LLM-generated marks |
| llm_feedback | LLM-generated feedback |

## Model and Grading Methodology

The project initially explored Google's Gemini API. Due to authentication, quota, and availability limitations, the implementation moved to a locally hosted model.

**Selected model: DeepSeek-R1:8B**

The grading pipeline uses:

1. Question and maximum marks.
2. Question-specific rubric.
3. Reference answer.
4. Student answer.
5. Structured prompting for automated evaluation.
6. JSON-based score and feedback extraction.
7. Score validation against the maximum marks.

The model generates two primary outputs:

- `llm_score`
- `llm_feedback`

The grading process uses a temperature setting of 0 for more consistent outputs.

## Evaluation and Experimental Analysis

The automated grading results are compared against the average human evaluator scores.

The project uses the following evaluation metrics:

- Mean
