# Summary: TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation

## 1. PAPER SNAPSHOT (multi-agent parallel extraction)

### Abstract
We introduce a comprehensive benchmark for travel planning that unifies fine-grained criteria into a single reward, enabling direct comparison of plan quality and seamless integration with reinforcement learning (RL). Across base models, RL generally improves itinerary feasibility over prompt-only and supervised baselines, yielding higher unified reward scores.

### Introduction
Planning is widely regarded as one of the most sophisticated cognitive skills in humans. However, benchmarks such as meeting scheduling and graph coloring (Jimenez et al., 2024; Zheng et al., 2024; Stechly et al., 2025) reveal that planning remains a challenging domain to solve comprehensively. Travel planning, in particular, has gained significant attention due to its real-world applicability and inherent complexity (Shao et al., 2024a; Chen et al., 2024). TravelPlanner (Xie et al., 2024) revealed that even state-of-the-art models achieved only a 4.4% pass rate when evaluating LLMs solely on their planning skills. Recognizing the limitations of TravelPlanner’s simplicity, more sophisticated benchmarks such as ChinaTravel (Shao et al., 2024a) and TripTailor (Wang et al., 2025) were developed. To address the travel planning problem, various approaches have been proposed to enhance the planning capabilities of LLMs. While these methods have shown promise in passing constraints, they rely heavily on test-time computation, which may not be practical in real-world applications where users eagerly await immediate results. Recently, reinforcement learning (DeepSeek-AI et al., 2025; Hao et al., 2023) (RL) has emerged as a key paradigm for scaling LLM reasoning capability and shows strong potential for planning. In this work, we propose a comprehensive framework that evaluates itinerary quality along fine-grained criteria and aggregates the results into a single reward score. Leveraging this reward, our benchmark enables the integration with RL, and our experiments accordingly assess the resulting improvements in effectiveness and performance. Across base models, reinforcement learning (GRPO) generally improves itinerary quality over prompt-only and supervised baselines, resulting in higher reward scores.

### Method
1. Researchers have developed diverse approaches to address travel planning challenges.
2. These include solver-based optimization methods Ju et al. (2024); Hao et al. (2025) and test-time compute methods Gui et al. (2025); Shao et al. (2024a); Kambhampati et4 al. (2024); Yang et al. (2025b).
3. While solver-based methods achieve high success rates by adhering to rule-based constraints, they often struggle to capture nuanced user preferences.
4. Conversely, test-time compute methods, although potentially more flexible, face limitations in real-time deployment due to their test time demands.
5. The key to advancing travel planning while maintaining user-friendly response times lies in enhancing the reasoning capabilities of Large Language Models (LLMs).
6. Recently, Reinforcement Learning (RL) has emerged as a promising paradigm for improving LLMs’ reasoning and planning abilities (Hao et al., 2023; Shao et al., 2024b; DeepSeek-AI et al., 2025).
7. Building on these advancements, our work investigates the integration of RL techniques into travel planning framework, aiming to strike an optimal balance between itinerary quality and computational efficiency.
8. Although LLMs have made significant strides in reasoning and planning capabilities, evaluation benchmarks remain far from perfect.
9. Recently, several travel planning benchmarks have been proposed to assess the itinerary quality.
10. TravelPlanner (Xie et al., 2024) introduces commonsense and hard constraints to evaluate the feasibility of itineraries.
11. ChinaTravel (Shao et al., 2024a) and ITINERA (Tang et al., 2024) incorporate the soft constraint to evaluate the aspects that cannot be addressed as boolean constraint satisfaction problems.
12. TripTailor (Wang et al., 2025) and RealTravel (Shao et al., 2025) further advances the field by integrating LLM-based evaluation to assess the rationality and personalization of travel itineraries.
13. Our work introduces a comprehensive benchmark that unifies multifaceted evaluation criteria into a single reward, enabling more direct comparisons of plan quality.
14. Furthermore, we incorporate real-world large-scale user queries to assess the model’s planning capabilities in complex and unpredictable scenarios.
15. We introduce TripScore, a comprehensive benchmark for evaluating LLMs’ capabilities in complex, multi-constraint travel planning scenarios.
16. To focus on assessing the model’s core reasoning abilities, we streamline the evaluation process by directly providing all relevant information, thus eliminating extraneous tool usage complexities.
17. This approach aligns with the sole-planning setting described in Xie et al. (2024).
18. Our benchmark encompasses a total of 4,870 queries, categorized into 3 splits: 3,493 training samples, 158 validation samples and 1,219 test samples.
19. The test set contains 1,219 queries from two sources: 1,000 synthetic queries constructed from usage statistics (combinations of popular destinations and frequent durations, with randomized preferences) and 219 real-world free-form user requests.
20. The dataset spans 897 cities, 9,376 hotels and 10,997 attractions.
21. To comprehensively evaluate the plan quality, we have incorporated four distinct types of constraints in TripScore, as outlined in Table 2.
22. The format constraints assess whether the plan adheres to the specified format guidelines.
23. The commonsense constraints evaluate the logical feasibility of the plan.
24. The soft and preference constraints measure the alignment of the plan with plan quality standards and user preferences.
25. Format constraints ensure the structural integrity and accuracy of the generated plan.
26. Commonsense constraints rigorously test the model’s ability to apply real-world logic and practical considerations to travel planning.
27. Soft constraints, while not strictly mandatory, play a crucial role in assessing the quality and practicality of an itinerary.
28. While adherence to these constraints is not obligatory, the degree to which a plan complies correlates directly with its reward score, indicating enhanced practicality.
29. Personal preference constraints evaluate the model’s ability to tailor the itinerary to individual user preferences.
30. We construct synthetic queries from user interaction logs by sampling the top 900 most requested destinations.
31. Moreover, to capture diverse user needs, we augment queries with randomized user preferences, selected from a predefined set outlined in the personal preference constraints in Table 2.
32. To enhance the benchmark’s authenticity, we incorporate real user requests from production travel applications under appropriate permissions.
33. Users submit free-form text describing their needs, and our system will return a tailored itinerary for them.
34. In this way, these logs capture genuine planning intent rather than contrived prompts.
35. To ensure comprehensive and accurate retrieval, we rely on production-grade industry data curated and maintained by professional operators, with the benchmark snapshot frozen and last updated in August 2025.
36. However, to streamline the evaluation process in this benchmark, we bypass the tool-call stage and directly provide the planning model with the information collected by the agent.
37. After collecting the user request and corresponding information, we implement a rigorous quality assurance process to remove the invalid request, leveraging both LLMs and human evaluation.
38. Following previous work (Xie et al., 2024), we evaluate travel plans using Delivery Rate and Commonsense Constraint Pass Rate.
39. The delivery rate calculates the ratio of plans that successfully meet all format constraints, reflecting the model’s ability to understand and follow structural requirements.
40. The commonsense constraint pass rate calculates the ratio of plans that pass all commonsense constraints among tested plans, ensuring that the generated itineraries exhibit logical consistency and real-world practicality.
41. Furthermore, we introduce the Reward for the quality of the itinerary, which integrates all constraints into a single metric.
42. The final reward $R$ is calculated as follow:
    $$R(S; \theta) = S_{\text{format}} + S_{\text{com}} + w_3 \frac{\sum_j w_{1,j} S_{\text{soft},j}}{\sum_j w_{1,j}} + w_4 \frac{\sum_k w_{2,k} S_{\text{pref},k}}{\sum_k w_{2,k}} \qquad (2)$$
43. We adopt a lexicographic gate for hard constraints to prevent pathologically invalid plans from being rewarded: format must pass first, then commonsense.
44. To avoid hard-penalty dominance and score compression, we bound the penalties to small constant ($S_{\text{format}} \in \{-3, +1\}, S_{\text{com}} \in \{-1, +1\}$), normalize all soft and preference sub-scores to $[0,1]$, and combine them with weights that sum to 1.
45. To validate our reward score, we recruited 203 travel experts to rank in pairs of travel plans.
46. The result show that inter-annotator agreement was moderate (Cohen’s $\kappa = 0.5421$ for pairwise; Fleiss’s $\kappa = 0.5039$ across three raters), with mean pairwise agreement of 71.69% and overall three-rater agreement of 59.00%.
47. We optimize the weight used in the Equation 2 based on expert annotations to compute the reward.
48. The annotation dataset was given the final label by majority voting.
49. Given route pairs and the ground-truth labels, we learn a scoring function $R(p)$ that maximizes agreement with labels.
50. The selected weights align well with human annotations, achieving a cross-validated validation accuracy of $0.6075 \pm 0.0275$ (mean $\pm$ std) and a bootstrap accuracy of $0.6138$ with a 95% CI of $[0.5967, 0.6383]$.
51. This suggests the reward captures many of the factors experts use when judging itineraries.
52. In this section, we compare our evaluation framework with LLM-as-judge (Zheng et al., 2023; Singh et al., 2024) under multiple backbones (Gemini-2.5-flash/pro, GPT-4o, GPT-4.1-Mini, DeepSeek-V3-0324).
53. Across the backbones where our method is applied, it consistently attains the highest agreement (e.g., Gemini-2.5-pro: 62.62%), and exhibits stronger ordinal alignment with human preferences.
54. By contrast, our framework achieves higher agreement with lower evaluation overhead.
55. Our comprehensive evaluation encompassed a diverse range of state-of-the-art models, including both proprietary and open-source LLMs.
56. We evaluated GPT-4.1-Mini, GPT-4o (OpenAI, 2023), DeepSeek-V3-0324 (DeepSeek-AI et al., 2025), Qwen3-8B, Qwen3-14B, Qwen3-32B (Yang et al., 2025a).
57. We adopt the metrics of Delivery rate (DR), Commonsense constraint Pass Rate (CPR), Reward score, and the generation time to evaluate the different methods comprehensively.
58. Because DR is a hard feasibility gate, a high correlation between DR and Reward is expected.
59. We examined four categories of planning approaches.
60. First, we evaluated direct methods, which involve the straightforward application of LLMs to the planning task.
61. Second, we explore test-time compute methods, including ZS-CoT (Wei et al., 2022), LLM-Modulo (Kambhampati et al., 2024), and HyperTree (Gui et al., 2025), which enhance model performance by increasing inference time.
62. Third, we investigate the neural-symbolic methods 1, including TTG (Ju et al., 2024) and NESY (Shao et al., 2024a), which integrate LLM and symbolic method.
63. We also assess fine-tuning techniques, including Supervised Fine-Tuning (SFT), Rejection Sampling Fine-Tuning (RFT) (Yuan et al., 2023), and GRPO (Shao et al., 2024b), which enhance the planning capability of models by parameter optimization.
64. In this section, we discuss the performance of various methods and LLMs in Table 3.
65. Within the same base model, approaches such as LLM-Modulo, and HyperTree consistently raise DR/CPR and reward relative to Direct prompting on both synthetic and real-world settings, but they incur substantially larger inference time (e.g., GPT-4o, real-world: Reward 1.61 to 2.69, Time 21.36s to 62.39s).
66. The extra reasoning steps executed at test time are effective for LLMs planning tasks.
67. Neuro-symbolic approaches (TTG/NESY) significantly increase CPR (often near 100%), yet do not translate into higher Reward because DR remains low.
68. We found that the low DR is due to strict hard constraints leading to no feasible solutions for travel plan generation, resulting in consistent failures.
69. Across backbones, fine-tuned variants (SFT/RFT/GRPO) simultaneously lift DR, CPR, and reward, while keeping inference time comparable to Direct.
70. On a fixed Qwen3-8B SFT backbone, GRPO further improves the reward over RFT/SFT (e.g., real-world: 1.82/1.53 to 2.04) and keeps CondR broadly comparable, with a small gain on Qwen3-8B synthetic (2.88/2.90 to 2.91) and a slight decrease on Qwen3-14B real-world (3.56/3.52 to 3.52), indicating that RL helps enhance plan feasibility without sacrificing plan quality.
71. By contrast, fine-tuned variants (SFT/RFT/GRPO) markedly raise DR and CPR and translate these gains into higher Reward with moderate time overhead, indicating that capacity-limited models benefit far more from parameter adaptation than from additional test-time reasoning.
72. To analyze the pass rate, Figure 2 breaks down violations of format and commonsense constraints on the real-world set.
73. And we found consistent dominant errors across all methods are format: information accuracy and response format, and commonsense: operating hours and chronological order.
74. Both SFT and GRPO training methodologies demonstrate significant efficacy in mitigating this type of hallucinations.
75. As the trip duration extends from 1 to 5 days, we observe a notable degradation in performance across vanilla baselines.
76. In contrast, GRPO (14B), which denotes the GRPO method applied to the Qwen3-14B model, generally achieves the highest DR and CPR scores while exhibiting the smallest performance decline as trip duration increases, indicating superior stability in long-horizon planning scenarios.
77. This suggests that GRPO improves the model’s reasoning ability for complex itineraries: it learns to maintain the consistency and constraint satisfaction (e.g., operating hours and chronological order) while preserving valid output structure, leading to stronger performance when scheduling becomes more challenging.
78. To examine how itinerary quality is assessed, we present several head-to-head cases in Figure 4.
79. This underscores the need for fine-grained comparative evaluation rather than relying solely on rules or single-pass LLM judgments.
80. Reliable evaluation must account for structural validity, spatio-temporal coherence, semantic value (iconicity and diversity), and user preferences.
81. A practical evaluator should fuse rule-based diagnostics with preference signals, quantify uncertainty, and return actionable feedback.

### Conclusion
1. In this work, we introduce a comprehensive evaluation framework that aggregates fine-grained criteria into a single reward for itinerary quality, achieving 60.75% agreement with expert annotations and surpassing LLM-as-judge baselines.
2. Finally, we conduct extensive comparative experiments that underscore the effectiveness of our framework and demonstrate the promise of fine-tuning techniques for improving itinerary quality while maintaining low inference time.

---

## 2. 150-WORD SUMMARY

This paper introduces TripScore, a comprehensive benchmark and evaluation framework designed to address the challenges of assessing travel planning capabilities in large language models. While prior benchmarks evaluate feasibility through binary constraint satisfaction, TripScore unifies fine-grained criteria—including format accuracy, commonsense logic, soft constraints, and user preferences—into a single reward score. To construct a highly realistic evaluation setting, the authors release a large-scale dataset of 4,870 queries, incorporating 219 free-form, authentic user requests from production travel applications. Through extensive experiments, the paper evaluates direct prompting, test-time computation, neuro-symbolic, and parameter-tuning methods. The findings show that reinforcement learning (specifically via GRPO) significantly enhances itinerary quality and planning feasibility over supervised baselines while keeping inference overhead extremely low. Ultimately, the TripScore evaluator exhibits moderate agreement (60.75%) with human travel experts, outperforming multiple LLM-as-judge baselines and providing a robust, RL-integrable pathway for complex travel planning optimization.

---

## 3. NOVELTY & CONTEXT

### Key Prior Works
- **TravelPlanner (Xie et al., 2024)**: Evaluated LLMs on planning capabilities under commonsense and hard constraints in sole-planning environments, demonstrating that state-of-the-art models achieved only a low 4.4% success rate.
- **ChinaTravel (Shao et al., 2024a)**: Introduced a benchmark evaluating travel planning against survey-based soft constraints, going beyond simple boolean rules to model quality attributes.
- **TripTailor (Wang et al., 2025)**: Evaluated LLMs' travel plans by integrating LLM-based judgments to assess rationality, consistency, and alignment with detailed user preferences.

### Core Contribution & Differentiation
- **Unified Reward Scoring Formula**: TripScore translates four main constraint types (format, commonsense, soft constraints, and user preferences) into a single, continuous, mathematically bounded reward score. This makes the evaluation metric directly compatible with reinforcement learning framework (e.g., GRPO), moving beyond binary pass/fail grades.
- **Authentic User-Log Queries**: Instead of synthetically generated or questionnaire-based requests, the benchmark test set includes 219 free-form, real-world queries collected from production travel logs, reflecting messy, authentic user intents.
- **Reinforcement Learning Validation**: Demonstrates for the first time that training models with GRPO (Reinforcement Learning) directly optimizes planning feasibility and quality metrics without adding the massive latency overhead of test-time search algorithms.

### Accessibility
- **None / Not Accessible**: (As of current search findings, no dedicated active public repository or official dataset download URL has been officially released for this paper.)

---

## 4. FINDINGS GROUNDED IN EXAMPLES

### Finding 1: Mitigation of Hallucinations and Infeasible Itineraries
- **Claim**: Reinforcement learning (GRPO) training demonstrates significant efficacy in mitigating spacing, accuracy, and chronological errors, especially as plans scale in temporal length.
- **Example**: The authors analyze formatting and commonsense violations across trip durations ranging from 1 to 5 days on the real-world query set. Vanilla baselines show sharp drops in Delivery Rate (DR) and Commonsense Pass Rate (CPR) as duration grows. However, the GRPO-trained Qwen3-14B model exhibits the smallest performance decline, maintaining high consistency in formatting and chronological activity ordering.
- ** Tangibility**: This makes the claim tangible by showing that parameter adaptation via RL explicitly trains the model to respect complex temporal structures (like matching attraction operating hours or sequencing departure/arrival locations) over long planning horizons without generating impossible timetables.

### Finding 2: Alignment of mathematical reward with Expert Judgments
- **Claim**: The designed mathematical reward function captures the key criteria and trade-offs used by human domain experts when evaluating itineraries.
- **Example**: The paper validates its scoring function using annotations from 203 travel experts who ranked pairwise plan itineraries. The experts reached a moderate pairwise agreement of 71.69% (with a three-rater agreement of 59.00%). By optimizing the weight terms of the reward scoring function (Equation 2) against the human choices, the learned TripScore evaluator achieved a cross-validation agreement accuracy of 60.75% with the experts' majority votes.
- **Tangibility**: This confirms that the continuous reward score is a reliable proxy for human preferences, reaching roughly 82.8% of the human ceiling (which is bounded by the experts' own internal consensus of 71.69%).

### Finding 3: Computational Efficiency of Fine-Tuning vs. Test-Time Compute
- **Claim**: Parameter tuning (SFT/RFT/GRPO) improves planning success and reward values with negligible runtime overhead, making them far more practical for deployment than test-time search methods.
- **Example**: The authors compare different methods on GPT-4o and Qwen3. Test-time search algorithms like HyperTree and LLM-Modulo achieve higher plan quality but significantly inflate response times (e.g., GPT-4o's average response time on real-world queries jumps from 21.36 seconds to 62.39 seconds). In contrast, Qwen3-8B trained with GRPO achieves comparable reward score improvements while maintaining a fast, near-instantaneous inference duration.
- **Tangibility**: This illustrates that reinforcement learning embeds the necessary reasoning and constraint compliance directly into the model's weights, bypassing the need for slow, iterative search loops during real-time user requests.
