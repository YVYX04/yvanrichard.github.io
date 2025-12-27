---
title: Structure of a Small Machine Learning Pipeline
subtitle: A conceptual guide to building a small-scale machine learning pipeline—from problem framing and data preparation to baseline models, tuning, evaluation, and deployment considerations.
layout: default
date: 2025-12-27
keywords: machine learning, business, economics
published: true
---

# Foreword

In this post, I outline the typical stages of a small-scale machine learning (ML) pipeline. I present a conceptual,
step-by-step methodology for applying core ML techniques.

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

### Setting the Stage

In *CRISP-DM: Towards a Standard Process Model for Data Mining*, Wirth and Hipp (2000) propose a standardized, industry-agnostic workflow for data mining projects. Although not formulated specifically for modern ML, it aligns closely with our structure. The project's lifecycle is decomposed into six phases: (i) business understanding, (ii) data understanding, (iii) data preparation, (iv) modelling, (v) evaluation, and (vi) deployment.

The *business understanding* phase is typically the decisive step in applied ML. Its purpose is to translate a managerial objective into a well-posed prediction problem by specifying: (i) the decision to be supported, (ii) the unit of prediction (what is predicted, for whom, and when), (iii) the information set available at prediction time (principle of *ex-ante observability: features must be available at decision/prediction time*), and (iv) success criteria expressed as measurable metrics. Wirth and Hipp (2000, p. 5) describe this step as focusing on "understanding objectives and requirements from a business perspective, and then converting this knowledge into a data mining problem definition".

### Practical Examples

In a [project](https://github.com/YVYX04/ML_First_Steps/blob/main/02_real_estate.ipynb) inspired by Géron’s *Hands-On Machine Learning* (Géron, 2022), I predict the median house value for California census blocks in order to improve upon heuristic pricing rules used by real estate agents. This objective maps naturally to a *regression* task. A common metric for model selection in this setting is the mean squared error (MSE), which penalizes larger deviations more strongly than smaller ones:

$$
\mathrm{MSE}(y^{(i)}, \hat{y}^{(i)}) = \frac{1}{n} \sum_{i = 1}^{n} (y^{(i)} - \hat{y}^{(i)})^2
$$

where we seek to minimize the square deviation between the true value $y^{(i)}$ and the estimate $\hat{y}^{(i)}$.

In a second, ongoing [project](https://github.com/YVYX04/Swiss_Train_ML), the business objective is to flag delayed SBB (Schweizerische Bundesbahnen) trains on the Swiss rail network. SBB defines a train as *delayed* when its arrival delay exceeds $3$ minutes, which induces a *binary classification* task. Because delays are relatively rare, accuracy can be misleading: a model that predicts “on time” for most observations may achieve high accuracy while failing to detect the events of interest. Instead, performance should be assessed through the precision–recall trade-off, and a standard summary metric is the $F_1$ score.

### Desired Output

The first step of an ML pipeline is therefore not “choosing an algorithm,” but *formalizing the decision problem*. Concretely, the output of this phase should be a short specification that states the prediction target, the unit and timing of prediction, the permissible features, and a primary metric reflecting business value and risk.

## 2.2. Data

### Data Acquisition, Exploratory Data Analyis, & Cleaning

With the prediction task and success criteria defined, the focus shifts to constructing and understanding a dataset that faithfully represents the information environment of the decision.

The ML literature treats data acquisition and management as a first-order modelling concern rather than a peripheral engineering task: beyond extracting raw records, the objective is to build a reproducible data asset with explicit schemas, quality checks, and traceable transformations that can be executed consistently in both training and production. In their tutorial on production ML pipelines, Polyzotis et al. (2017) emphasize that “preparing data for an ML pipeline requires effort and care,” and explicitly warn that invalid inputs can propagate into operational failures, motivating systematic validation, monitoring, and correction mechanisms as part of the data layer. This perspective aligns with Sculley et al.’s caution that, without disciplined design, “pipeline jungles often appear in data preparation,” as feature pipelines accrete ad hoc joins, scrapes, and resampling steps that are difficult to test and maintain. In my SBB project I developed a minimal [data pipeline](https://github.com/YVYX04/Swiss_Train_ML/blob/main/src/data/clean_data_ist.py) but it looks like a small "jungle" (the author will make his best efforts to improve in the future...).

Once the data has been made available for retrieval, an ideal first step is to perform a *minimal exploratory data analysis* (EDA). This stage helps us gain a clearer understanding of the dataset’s structure and content. As Behrens (1997, p. 132) states, “the goal of EDA is to discover patterns in data […] until a plausible story of the data is apparent.” In the context of a ML project, this is precisely the purpose EDA serves. By examining distributions, interactions, and variability within the data, we build knowledge that shapes the steps that follow. Data cleaning becomes more targeted, allowing decisions about handling missing values to be made with greater justification (building data imputation pipelines). Similarly, insights uncovered during EDA inform feature engineering, enabling the creation of features that better account for variance in the target. Ultimately, the strength of our predictions is closely tied to the depth of understanding achieved during this exploratory phase and our ability to translate those findings into meaningful model improvements.

### Feature Engineering

Feature engineering is best viewed as the deliberate construction of a *data representation* that makes the learning task statistically and operationally tractable, subject to the constraint that every feature must be computable from information available at prediction time. As stated by Domingos (2012, p.):

<div style="
  max-width: 50vw;
  margin: 1.25rem auto;
  padding: 1rem 1.25rem;
  text-align: left;
  font-style: italic;
  line-height: 1.6;
  border-left: 4px solid #bbb;
  background: #eeeeeeff;
">
  <p style="margin: 0;">
    "Often, the
        raw data is not in a form that is amenable to learning, but you can construct features from it that are. This
        is typically where most of the effort in
        a machine learning project goes. It is
        often also one of the most interesting
        parts, where intuition, creativity and
        “black art” are as important as the
        technical stuff."
  </p>
</div>

In the SBB delay setting, this implies engineering temporally valid predictors (e.g., scheduled-time indicators, calendar effects, rolling historical delay statistics at the station–hour level, and time-aligned weather covariates), while controlling for leakage (no post-arrival information) and ensuring the transformations are reproducible and consistent between training and deployment.

## 2.5. Modelling and Evaluation

### Dummy Baselines

Before comparing sophisticated algorithms, it is good practice to establish *simple baselines* that quantify the minimum acceptable performance (e.g., a majority-class classifier, or a mean/median predictor). Baselines serve two roles: they provide an interpretable benchmark, and they act as a diagnostic tool.

### Candidate Models and Selection Criterion

Algorithm selection should be framed as a constrained choice among a small set of model families that match the task and the operational requirements (e.g., interpretability or robustness to non-linearities and interactions). Comparison should rely on a *consistent evaluation protocol* (e.g. identical temporal split in my SBB project) and a single primary metric aligned with the business objective (e.g. the MSE in the housing prices project).

### Error analysis and hyperparameter optimisation

Model improvement is typically driven by *error analysis* (where does the model fail, and why?) and by controlled hyperparameter optimisation. In practice, this means inspecting class-conditional errors (confusion matrices, precision–recall trade-offs), stratifying performance by regimes (e.g., peak vs. off-peak, dense vs. sparse stations), and using learning curves to separate underfitting from overfitting. For hyperparameter tuning, Bergstra and Bengio (2012) argue that "randomly chosen trials are more efficient for hyper-parameter optimization than trials on a grid". On this same topic Cawley and Talbot (2010) caution that “the effects of this form of over-fitting are often of comparable magnitude to differences in performance between learning algorithms,” motivating careful separation between tuning and final evaluation. A useful approach for validating the stability (low variance) of a model is $k$-fold cross-validation. 

### Final evaluation

The final evaluation should be performed *once*, on a test set that has not been used for feature design, model selection, or hyperparameter tuning, in order to obtain an unbiased estimate of generalization performance. When cross-validation is used for model selection, it should be nested (or accompanied by a truly held-out test set) to mitigate selection bias. If the performances meet the expected benchmarks, the ML pipeline can be moved to production.

## 2.7. Deployment & Supervision

Even small projects should anticipate that deployment is where systems fail. A central operational insight is that ML systems accumulate “technical debt”: they may be quick to build but expensive to maintain (Sculley et al., 2015). To prevent those issues, a minimal deployment plan could include:

+ Reproducible training script (single command, pinned dependencies).
+ A saved artefact containing preprocessing + model.
+ A clearly versioned dataset snapshot and feature code.
+ Monitoring for input schema changes, performance drift, and data distribution shift.

*Concept drift* (changes in the relationship between inputs and targets) is common in operational environments and must be handled via monitoring and retraining policies. Gama et al. (2014) provide an in-depth analysis of the tools available to mitigate concept drifts. A Google's research team also craft 28 specific tests for monitoring running ML systems (Breck et al., 2017).

# 3. Conclusion

This post presented a compact, practice-oriented structure for small-scale ML projects, from problem framing and data preparation to feature engineering, model selection, and evaluation. The central takeaway is that robust performance rarely comes from algorithm choice alone, but from disciplined problem specification, reproducible data pipelines, and careful validation procedures that prevent leakage and selection bias.

# References

- Biswas, S., Wardat, M., & Rajan, H. (2022). The art and practice of data science pipelines: A comprehensive study of data science pipelines in theory, in-the-small, and in-the-large. In *Proceedings of the 44th International Conference on Software Engineering* (pp. 2091-2103).

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305.

- Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017, December). The ML test score: A rubric for ML production readiness and technical debt reduction. In 2017 *IEEE international conference on big data (big data)* (pp. 1123-1132). IEEE.

- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107.

- Domingos, P. (2012). A few useful things to know about machine learning. *Communications of the ACM, 55(10)*, 78-87.

- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM computing surveys (CSUR), 46(4)*, 1-37.

- Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media, Inc.

- McElheran, K., Li, J. F., Brynjolfsson, E., Kroff, Z., Dinlersoz, E., Foster, L., & Zolas, N. (2024). AI adoption in America: Who, what, and where. *Journal of Economics & Management Strategy, 33(2)*, 375-415.

- Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2017, May). Data management challenges in production machine learning. In *Proceedings of the 2017 ACM international conference on management of data* (pp. 1723-1726).

- Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., & Dennison, D. (2015). Hidden technical debt in machine learning systems. Advances in *neural information processing systems*, 28.

- Wirth, R., & Hipp, J. (2000, April). CRISP-DM: Towards a standard process model for data mining. In *Proceedings of the 4th international conference on the practical applications of knowledge discovery and data mining* (Vol. 1, pp. 29-39).
