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

With the aim of making a modest contribution to this broader development, this report outlines the structure of a minimalistic ML pipeline tailored to small-scale projects and practical business applications. To illustrate the procedure, I will draw on examples from my projects throughout.

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

In their paper *CRISP-DM: Towards a Standard Process Model for Data Mining*, Wirth and Hipp (2000) attempt to lay the foundations of a standardized data mining (DM) pipeline useful across all types of industries: the *Cross Industry Standard Process for Data Mining* project. While not directly focused on ML, their framework is extremely well aligned with the scope of this report. Indeed, they reduce the pipeline to six main "phases": ($i$) business understanding, ($ii$) data understanding, ($iii$) data preparation,
($iv$) modelling, ($v$) evaluation, and ($vi$) deployment.

This initial phase, *business understanding*, is the backbone of the majority of ML projects in the industry. This first
step "focuses on understanding the project objectives and requirements from a
business perspective, and then converting this knowledge into a data mining problem
definition" (Whirth & Hipp, 2000, p. 5).
Defining precisely the goals and objectives of the ML project is a detrimental step to its success since the performance metrics used for training and tuning the models rely on those early guidelines.

For instance, in a [project](https://github.com/YVYX04/ML_First_Steps/blob/main/02_real_estate.ipynb) I realized based on Aurélien Géron's book *Hands on Machine Learning*, the task at hand was to forecast the median price in different blocks (aggregate of houses) in California to outperform the outdated techniques of real estate agents. In this specific case, we clearly identify the project as being a classic *regression* problem and the one of the gold standard typically selected as a performance metric is the *mean squared error* (MSE).

$$
\begin{equation}
\mathrm{MSE}(y^{(i)}, \hat{y}^{(i)}) = \frac{1}{n} \sum_{i = 1}^{n} (y^{(i)} - \hat{y}^{(i)})^2
\end{equation}
$$

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

- Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media, Inc.

- McElheran, K., Li, J. F., Brynjolfsson, E., Kroff, Z., Dinlersoz, E., Foster, L., & Zolas, N. (2024). AI adoption in America: Who, what, and where. *Journal of Economics & Management Strategy, 33(2)*, 375-415.

- Wirth, R., & Hipp, J. (2000, April). CRISP-DM: Towards a standard process model for data mining. In *Proceedings of the 4th international conference on the practical applications of knowledge discovery and data mining* (Vol. 1, pp. 29-39).