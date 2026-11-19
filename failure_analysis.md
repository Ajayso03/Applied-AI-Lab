# SupportIQ: Comprehensive Failure Analysis

This failure analysis inspects genuine prediction discrepancies produced by the fine-tuned LoRA model on the held-out test split (`SupportIQ/data/test.jsonl`). None of these examples are fabricated; all are extracted directly from `SupportIQ/results/predictions.csv`.

---

## Evaluation Overview

Across the 137 held-out test examples:
- **JSON Parsing Success Rate:** 100.00%
- **Category Accuracy:** 94.89% (130 / 137 correct)
- **Sentiment Accuracy:** 91.24% (125 / 137 correct)
- **Priority Accuracy:** 68.61% (94 / 137 correct)
- **Exact 3-Field Match:** 59.12% (81 / 137 perfect matches)

As the statistics demonstrate, category and sentiment recognition achieved exceptional performance (>91%), while priority classification exhibited the most ambiguity. The three genuine failure cases below illustrate the primary failure modes.

---

## Failure Example 1: Priority Escalation via Imperative Phrasing

### Metadata
- **Sample ID:** `supp_1137`
- **Category:** Shipping
- **Status:** Priority Mismatch (`low` predicted as `medium`)

### Input Text
```text
Can someone explain this? Would be nice if the courier sent an SMS reminder in addition to the email notification. Fix this asap.
```

### Classification Output
| Field | Expected Ground Truth | Actual Model Output | Match Status |
| :--- | :--- | :--- | :--- |
| **Category** | `shipping` | `shipping` | Correct |
| **Priority** | `low` | `medium` | **Incorrect (Escalated)** |
| **Sentiment** | `negative` | `negative` | Correct |

### Detailed Diagnosis
1. **What Went Wrong:** The model accurately identified that the message belongs to `shipping` and expresses `negative` sentiment, but elevated the priority from `low` to `medium`.
2. **Root Cause Analysis:** The core operational request is a cosmetic feature suggestion ("Would be nice if the courier sent an SMS reminder"). However, the customer framed this preference with aggressive, high-urgency idioms: *"Can someone explain this?"* and *"Fix this asap."* The attention heads in the transformer layers placed heavy semantic weight on the imperative token sequence `"Fix this asap"`, overriding the low-stakes nature of the primary topic.
3. **Actionable Improvement:** Incorporate priority-specific decision rules during dataset curation and training. Specifically, train the model to distinguish between *functional blockages* (which warrant high/medium priority) versus *stylistic venting on cosmetic requests* (which should remain low priority despite demanding language).

---

## Failure Example 2: Conflating Problem Severity with Negative Sentiment

### Metadata
- **Sample ID:** `supp_1313`
- **Category:** Shipping
- **Status:** Sentiment Mismatch (`neutral` predicted as `negative`)

### Input Text
```text
Freight consignment #71709 required a liftgate at delivery but the driver arrived in a standard truck and refused delivery. Reschedule priority freight with liftgate.
```

### Classification Output
| Field | Expected Ground Truth | Actual Model Output | Match Status |
| :--- | :--- | :--- | :--- |
| **Category** | `shipping` | `shipping` | Correct |
| **Priority** | `high` | `high` | Correct |
| **Sentiment** | `neutral` | `negative` | **Incorrect (Over-penalized)** |

### Detailed Diagnosis
1. **What Went Wrong:** The model successfully identified that the issue is a `high`-priority `shipping` failure (failed delivery requiring expedited re-dispatch), but marked the sentiment as `negative` instead of `neutral`.
2. **Root Cause Analysis:** Words like *"refused delivery"* and *"required ... but"* describe an adverse, problematic event. In pre-trained causal language models, representations of failure scenarios are strongly correlated with negative sentiment polarity. However, the author maintained a strictly factual, business-oriented, and objective tone without emotional modifiers, abusive words, or exasperation punctuation. The model failed to decouple the *adverse nature of the incident* from the *emotional demeanor of the speaker*.
3. **Actionable Improvement:** Enrich the training set with contrastive pairs of objective incident reports (neutral sentiment, high operational severity) versus emotionally charged incident reports (negative sentiment, high operational severity).

---

## Failure Example 3: Politeness Masking High-Priority Security Risk

### Metadata
- **Sample ID:** `supp_0919`
- **Category:** Account
- **Status:** Priority Underestimation (`high` predicted as `medium`)

### Input Text
```text
Hey there! Really appreciate your fast account security alerts! I received an alert and want to quickly reset my master 2FA keys with your help. Thanks so much!
```

### Classification Output
| Field | Expected Ground Truth | Actual Model Output | Match Status |
| :--- | :--- | :--- | :--- |
| **Category** | `account` | `account` | Correct |
| **Priority** | `high` | `medium` | **Incorrect (Underestimated)** |
| **Sentiment** | `positive` | `positive` | Correct |

### Detailed Diagnosis
1. **What Went Wrong:** The model correctly identified the `account` domain and the customer's `positive` sentiment, but downgraded the priority from `high` to `medium`.
2. **Root Cause Analysis:** The customer utilized cheerful, highly polite phrasing (*"Hey there!"*, *"Really appreciate..."*, *"Thanks so much!"*). The underlying task is a critical account security matter: a security alert was triggered, and the user must reset their master 2FA keys. In enterprise support routing, identity/2FA recovery is strictly `high` priority because access and security vulnerabilities are at stake. However, the polite, calm tone led the model to infer that the user was not in acute distress, downgrading the urgency.
3. **Actionable Improvement:** Implement domain-specific priority overrides or multi-stage classification where account security actions (2FA resets, unauthorized access alerts, credential lockouts) are mapped to a minimum priority tier (`high`) regardless of emotional polarity.

---

## Engineering Takeaways & Architectural Lessons

1. **Category and Format Precision:** LoRA fine-tuning with rank 8 successfully converted an open-ended conversational LLM into a deterministic JSON generator with **100% valid JSON syntax** and **94.89% category accuracy**.
2. **The "Tone vs. Severity" Orthogonality Challenge:** In human communication, customer sentiment (affect) and operational priority (urgency) are frequently orthogonal. A user can be furious about a minor $1 typo (negative sentiment, low priority) or cheerful while reporting a major security lockout (positive sentiment, high priority). Small language models require explicit contrastive training examples to disentangle tone from business impact.

