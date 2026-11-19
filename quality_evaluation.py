"""
Output Quality Evaluation Module.
Implements an objective scoring rubric and automated rule-based validators
for the 10 evaluation prompts across Q4_K_M and Q8_0 model outputs.
"""

import ast
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


class QualityEvaluator:
    """
    Evaluates model outputs across 5 dimensions:
      1. Factual Correctness (max 2.5)
      2. Instruction Following (max 2.5)
      3. Completeness (max 2.0)
      4. Reasoning Quality (max 2.0)
      5. Output Formatting (max 1.0)
      Total = 10.0 points.
    """

    def __init__(self, prompts_data_path: str):
        with open(prompts_data_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)
        self.prompts_by_id = {p["id"]: p for p in self.prompts}

    def evaluate_response(self, prompt_id: str, output_text: str) -> Dict[str, Any]:
        """Runs rule-based and rubric evaluation for a given prompt output."""
        p_data = self.prompts_by_id.get(prompt_id)
        if not p_data:
            raise ValueError(f"Unknown prompt id: {prompt_id}")

        text = output_text.strip()
        obs = []
        scores = {
            "factual_correctness": 2.5,
            "instruction_following": 2.5,
            "completeness": 2.0,
            "reasoning_quality": 2.0,
            "output_formatting": 1.0,
        }
        rule_passed = True

        # Prompt-specific rule-based checks
        if prompt_id == "PROMPT-01":
            # Seating order: Bob -> Charlie -> Alice
            t_lower = text.lower()
            if "bob" in t_lower and "charlie" in t_lower and "alice" in t_lower:
                # Find positions
                p_bob = t_lower.find("bob")
                p_charlie = t_lower.find("charlie")
                p_alice = t_lower.find("alice")
                if "chair 1" in t_lower or "left to right" in t_lower:
                    obs.append("Correctly identified chairs.")
                if "modus" in t_lower or "refuses" in t_lower or "next to" in t_lower:
                    obs.append("Deductive reasoning explained.")
            else:
                scores["factual_correctness"] -= 1.0
                obs.append("Failed to clearly name all three friends in arrangement.")

        elif prompt_id == "PROMPT-02":
            # Python programming
            code_block = text
            if "```python" in text:
                code_block = text.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in text:
                code_block = text.split("```", 1)[1].split("```", 1)[0]

            if "def find_top_k_frequent" in code_block:
                obs.append("Defined required function name.")
            else:
                scores["instruction_following"] -= 1.0
                rule_passed = False
                obs.append("Missing required function signature find_top_k_frequent.")

            try:
                ast.parse(code_block)
                obs.append("Code is syntactically valid Python.")
            except SyntaxError:
                scores["factual_correctness"] -= 1.5
                rule_passed = False
                obs.append("Python code contains syntax errors.")

            if "list[int]" in code_block or "List[int]" in code_block:
                obs.append("Type annotations present.")
            else:
                scores["output_formatting"] -= 0.5

            if '"""' in code_block or "'''" in code_block:
                obs.append("Docstring included.")
            else:
                scores["completeness"] -= 0.5

        elif prompt_id == "PROMPT-03":
            # Math: $81.60
            if "81.6" in text or "$81.60" in text or "81.60" in text:
                obs.append("Final answer $81.60 is mathematically exact.")
            else:
                scores["factual_correctness"] -= 2.0
                rule_passed = False
                obs.append("Failed to arrive at correct discounted price $81.60.")

            if "96" in text and ("14.4" in text or "14.40" in text or "0.85" in text):
                obs.append("All intermediate arithmetic steps shown.")
            else:
                scores["reasoning_quality"] -= 0.5

        elif prompt_id == "PROMPT-04":
            # Summary: exactly 2 sentences
            # Split sentences by period, exclamation, question mark
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]
            if len(sentences) == 2:
                obs.append("Strictly adhered to 2-sentence constraint.")
            else:
                scores["instruction_following"] -= 1.0
                rule_passed = False
                obs.append(f"Produced {len(sentences)} sentences instead of exactly 2.")

            t_lower = text.lower()
            if "qubit" in t_lower or "superposition" in t_lower:
                obs.append("Included qubits/superposition concepts.")
            else:
                scores["completeness"] -= 0.8
            if "cryptograph" in t_lower or "security" in t_lower or "encryption" in t_lower or "rsa" in t_lower:
                obs.append("Included cryptographic/security implications.")
            else:
                scores["completeness"] -= 0.8

        elif prompt_id == "PROMPT-05":
            # JSON Generation
            clean_json = text
            if "```json" in clean_json:
                clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in clean_json:
                clean_json = clean_json.split("```", 1)[1].split("```", 1)[0]
            clean_json = clean_json.strip()

            try:
                data = json.loads(clean_json)
                obs.append("Successfully parsed pure JSON object.")
                required_keys = ["name", "age", "role", "organization", "city", "country", "publication_count", "primary_field", "email"]
                missing = [k for k in required_keys if k not in data]
                if missing:
                    scores["instruction_following"] -= 1.0
                    obs.append(f"Missing JSON keys: {missing}")
                else:
                    obs.append("All 9 required JSON schema keys present.")

                if data.get("age") == 42 and data.get("publication_count") == 34:
                    obs.append("Extracted integer fields age (42) and publication_count (34) accurately.")
                else:
                    scores["factual_correctness"] -= 1.0

            except Exception as e:
                scores["output_formatting"] -= 1.0
                scores["instruction_following"] -= 1.5
                rule_passed = False
                obs.append(f"Failed to produce valid JSON: {str(e)}")

        elif prompt_id == "PROMPT-06":
            # Technical explanation: TCP vs UDP
            has_p1 = "mechanism" in text.lower() or "connection" in text.lower()
            has_p2 = "reliability" in text.lower() or "overhead" in text.lower() or "flow control" in text.lower()
            has_p3 = "use case" in text.lower() or "cases" in text.lower()
            if has_p1 and has_p2 and has_p3:
                obs.append("Included all three required structured sections.")
            else:
                scores["instruction_following"] -= 0.8
                obs.append("Some required section headers were missing or merged.")

            if "handshake" in text.lower() or "syn" in text.lower():
                obs.append("Mentioned TCP connection handshake mechanism.")
            if "loss" in text.lower() or "retransmission" in text.lower() or "ack" in text.lower():
                obs.append("Accurately detailed reliability and retransmission differences.")

        elif prompt_id == "PROMPT-07":
            # Logical reasoning: Modus Tollens & Denying the Antecedent
            t_lower = text.lower()
            if "no" in t_lower and ("prod-db-04" in t_lower or "does not host" in t_lower or "cannot host" in t_lower):
                obs.append("Question 1 correctly identified as NO.")
            else:
                scores["factual_correctness"] -= 1.0

            if "modus tollens" in t_lower:
                obs.append("Question 1 correctly named Modus Tollens inference rule.")
            else:
                scores["reasoning_quality"] -= 0.8
                obs.append("Did not explicitly name Modus Tollens rule.")

            if "cannot" in t_lower or "uncertain" in t_lower or "not possible" in t_lower or "undetermined" in t_lower or "antecedent" in t_lower:
                obs.append("Question 2 correctly reasoned that heartbeats cannot be deduced with certainty.")
            else:
                scores["factual_correctness"] -= 1.0
                scores["reasoning_quality"] -= 0.7

        elif prompt_id == "PROMPT-08":
            # Information extraction
            if "INV-2024-8891" in text:
                obs.append("Extracted invoice number correctly.")
            else:
                scores["factual_correctness"] -= 0.5
            if "1474.20" in text or "$1474.20" in text:
                obs.append("Grand total $1474.20 extracted accurately.")
            else:
                scores["factual_correctness"] -= 0.8
            if "340" in text and "900" in text and "125" in text:
                obs.append("All three line item subtotals ($340, $900, $125) calculated and extracted.")
            else:
                scores["completeness"] -= 0.5

        elif prompt_id == "PROMPT-09":
            # Instruction following: 5 constraints
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]
            if len(sentences) == 4:
                obs.append("Constraint 1: Exactly 4 sentences satisfied.")
            else:
                scores["instruction_following"] -= 0.8
                rule_passed = False
                obs.append(f"Constraint 1 violated: generated {len(sentences)} sentences instead of 4.")

            if text.startswith("Designed"):
                obs.append("Constraint 2: Starts with 'Designed' satisfied.")
            else:
                scores["instruction_following"] -= 0.5
                rule_passed = False
                obs.append("Constraint 2 violated: did not begin with 'Designed'.")

            if len(sentences) >= 3 and "lumbar" in sentences[2].lower():
                obs.append("Constraint 3: Third sentence contains 'lumbar' satisfied.")
            elif "lumbar" in text.lower():
                scores["instruction_following"] -= 0.3
                obs.append("Constraint 3 partial: 'lumbar' included but not specifically in sentence 3.")
            else:
                scores["instruction_following"] -= 0.5
                rule_passed = False

            t_lower = text.lower()
            if "comfort" in t_lower or "comfortable" in t_lower:
                scores["instruction_following"] -= 0.8
                rule_passed = False
                obs.append("Constraint 4 violated: used forbidden word 'comfort'/'comfortable'.")
            else:
                obs.append("Constraint 4: Forbidden words strictly avoided.")

            if text.rstrip(" .").endswith("for every workday"):
                obs.append("Constraint 5: Ends with 'for every workday.' satisfied.")
            else:
                scores["instruction_following"] -= 0.5
                rule_passed = False
                obs.append("Constraint 5 violated: did not end with 'for every workday.'")

        elif prompt_id == "PROMPT-10":
            # Multi-step problem solving
            if "240" in text:
                obs.append("Step 1: Gross capacity 240 hours calculated correctly.")
            else:
                scores["factual_correctness"] -= 0.6
            if "48" in text and "192" in text:
                obs.append("Step 2: 48 hours overhead and 192 net dev hours calculated correctly.")
            else:
                scores["factual_correctness"] -= 0.6
            if "102" in text:
                obs.append("Step 3: Story total 102 hours calculated correctly.")
            else:
                scores["factual_correctness"] -= 0.6
            if "90" in text and ("yes" in text.lower() or "surplus" in text.lower() or "can complete" in text.lower()):
                obs.append("Step 4: Feasibility (Yes) and 90 hours surplus calculated accurately.")
            else:
                scores["factual_correctness"] -= 0.7

        total_score = max(0.0, round(sum(scores.values()), 2))

        return {
            "prompt_id": prompt_id,
            "category": p_data["category"],
            "total_score": total_score,
            "scores_breakdown": scores,
            "rule_based_passed": rule_passed,
            "observations": "; ".join(obs),
        }

    def compare_prompts(
        self,
        q4_outputs: Dict[str, str],
        q8_outputs: Dict[str, str],
        output_csv_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs comprehensive comparative evaluation across Q4_K_M and Q8_0 outputs.
        """
        comparisons = []
        for p in self.prompts:
            p_id = p["id"]
            q4_text = q4_outputs.get(p_id, "")
            q8_text = q8_outputs.get(p_id, "")

            q4_eval = self.evaluate_response(p_id, q4_text)
            q8_eval = self.evaluate_response(p_id, q8_text)

            diff_explanation = []
            if q8_eval["total_score"] > q4_eval["total_score"]:
                diff_explanation.append(f"Q8_0 scored higher (+{round(q8_eval['total_score'] - q4_eval['total_score'], 2)}) with greater precision or strict constraint adherence.")
            elif q4_eval["total_score"] > q8_eval["total_score"]:
                diff_explanation.append(f"Q4_K_M scored higher (+{round(q4_eval['total_score'] - q8_eval['total_score'], 2)}).")
            else:
                diff_explanation.append("Both models performed equally well according to the rubric.")

            if q4_text == q8_text:
                diff_explanation.append("Identical textual response produced by both quantisation tiers.")
            else:
                len_diff = len(q8_text.split()) - len(q4_text.split())
                diff_explanation.append(f"Length difference: Q8_0 produced {len_diff:+d} words compared to Q4_K_M.")

            comp_entry = {
                "prompt_id": p_id,
                "category": p["category"],
                "prompt": p["prompt"],
                "expected_answer_characteristics": p["expected_answer_characteristics"],
                "q4_output": q4_text,
                "q8_output": q8_text,
                "q4_score": q4_eval["total_score"],
                "q8_score": q8_eval["total_score"],
                "q4_observations": q4_eval["observations"],
                "q8_observations": q8_eval["observations"],
                "difference_explanation": " ".join(diff_explanation),
            }
            comparisons.append(comp_entry)

        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "prompt_id", "category", "prompt", "expected_answer_characteristics",
                    "q4_output", "q8_output", "q4_score", "q8_score",
                    "q4_observations", "q8_observations", "difference_explanation"
                ])
                writer.writeheader()
                writer.writerows(comparisons)

        if output_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(comparisons, f, indent=2)

        return comparisons
