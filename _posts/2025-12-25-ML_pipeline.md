---
title: Structure of a Small Machine Learning Pipeline
subtitle: A practical, step-by-step guide to building a small-scale machine learning pipeline—from problem framing and data preparation to baseline models, tuning, evaluation, and deployment considerations.
layout: default
date: 2025-12-25
keywords: machine learning, business, economics
published: true
---

# Foreword

In this post, I outline the typical stages of a small-scale machine learning (ML) pipeline. I present a rigorous,
step-by-step methodology for applying core ML techniques, and I discuss how each stage can be translated into actionable
business insights.

# 1. Introduction

Over the past decade, the field of *data science*, and *artificial intelligence* (AI) have attracted enormous attention
due to the progressive shift of our society towards a numerical era. OpenAI's public release of the large-language model 
(LLM) GPT 3.5. in November 2022 further propagated the sentiment that humanity might be at the dawn of a new industrial revolution.
Today, it is still unclear how, and where AI influence will really materialize in society. The manufacturing, information-technology,
finance, and healthcare sectors might be evident early adopters (McElheran et al., 2024), but the forthecoming decades will probably be
marked by the diffusion of AI technologies through the entire economy.

With the aim of making a modest contribution to this broader development, this report outlines a streamlined machine ML tailored to small-scale projects and practical business applications. To illustrate the procedure, I will draw on examples from my practical projects throughout.

# 2. The Machine Learning Pipeline

In their examination of machine learning workflows at varying scales, Biswas et al. (2022) outline how pipeline structures evolve according to operational requirements. In a professional, production-oriented setting, a fully integrated pipeline typically follows the structure shown below:

<div class='figure'>
    <img src="{{ '/images/DS_pipeline01.png' | relative_url }}" alt="Data pipeline"
         style="width: 100%; display: block; margin: 0 auto;" id = "fig1"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Concepts in a data science pipeline. The sub-tasks are listed below each stage. The stages are connected with feedback loops denoted with arrows. Solid arrows are always present in the lifecycle, while the dashed arrows are optional. Distant feedback loops (e.g.,from deployment to data acquisition) are also possible through intermediate stage(s).
    </div>
</div>

Naturally, the structure of a pipeline depends on the scope, constraints, and objectives of the project. Hence, I adapt my approach to this high-level structure. 

## 2.1. From Business Case to Success Metrics

[What is the business probem? For which metrics should we aim?]

```text
ML task (classification/regression/forecasting).
+ Define constraints (latency, interpretability, cost of errors, compliance).
+ Choose metrics aligned with business (precision/recall trade-off, MAE/RMSE, calibration).
+ Best practices + challenges: Prevent “metric gaming”, data leakage, misaligned objectives.
```

## 2.2. Data Collection & Understanding

[...]

## 2.3. Data Cleaning & Preprocessing

[...]

## 2.4. Exploratory Data Analysis (EDA) & Features Engineering

[...]

## 2.5. Baseline Models

[...]

## 2.6. Error Analysis, Hyperparamters Tuning

[...]

## 2.7. Final Evaluation, Deployment, & Supervision


# 3. Conclusion








# References

- Biswas, S., Wardat, M., & Rajan, H. (2022). The art and practice of data science pipelines: A comprehensive study of data science pipelines in theory, in-the-small, and in-the-large. In *Proceedings of the 44th International Conference on Software Engineering* (pp. 2091-2103).

- McElheran, K., Li, J. F., Brynjolfsson, E., Kroff, Z., Dinlersoz, E., Foster, L., & Zolas, N. (2024). AI adoption in America: Who, what, and where. *Journal of Economics & Management Strategy, 33(2)*, 375-415.