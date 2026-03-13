"""
Day 3: Simple Evaluation of LLM Answers using Large PDFs
Comparing llama-3.1-8b-instant vs llama-3.3-70b-versatile

Key Evaluation Metrics:
1. Relevance Score: Semantic similarity to ground truth (0-100)
2. Completeness Score: Coverage of key expected information (0-100)
3. Hallucination Risk: Detection of unsupported claims (0-100, lower is better)
4. Answer Quality: Combined metric considering all factors (0-100)
"""

import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import re
from datetime import datetime

# Initialize with Groq API
try:
    from groq import Groq
except ImportError:
    print("Installing groq SDK...")
    import subprocess
    subprocess.check_call(["pip", "install", "groq"])
    from groq import Groq

# Load environment
from dotenv import load_dotenv
load_dotenv()

@dataclass
class EvaluationScore:
    """Store evaluation metrics for a single response"""
    question_id: int
    model: str
    relevance_score: float  # 0-100: How similar to ground truth
    completeness_score: float  # 0-100: Coverage of key information
    hallucination_risk: float  # 0-100: Risk of false claims (lower is better)
    answer_quality: float  # 0-100: Combined quality score
    passed_quality_gate: bool  # Simple pass/fail (>= 70 on quality)
    response: str
    ground_truth: str


class LLMEvaluator:
    """Evaluates LLM responses against ground truth using multiple metrics"""
    
    def __init__(self):
        """Initialize Groq client"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.client = Groq(api_key=api_key)
        self.model_a = "llama-3.1-8b-instant"
        self.model_b = "llama-3.3-70b-versatile"
        
        # Load test set
        with open("test_set.json", "r") as f:
            self.test_data = json.load(f)["test_set"]
    
    def query_llm(self, question: str, model: str, temperature: float = 0.0) -> str:
        """Query LLM with deterministic settings"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": question
                }],
                temperature=temperature,  # Deterministic: no randomness
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying {model}: {e}")
            return ""
    
    def calculate_relevance_score(self, response: str, ground_truth: str) -> float:
        """
        Estimated relevance using keyword overlap and length match.
        In production, use embedding-based similarity (cosine similarity)
        score: 0-100
        """
        if not response or not ground_truth:
            return 0.0
        
        response_lower = response.lower()
        truth_lower = ground_truth.lower()
        
        # Simple keyword matching (rookie approach)
        truth_keywords = set(truth_lower.split())
        response_keywords = set(response_lower.split())
        
        if len(truth_keywords) == 0:
            return 50.0
        
        # Overlap percentage
        overlap = len(truth_keywords & response_keywords)
        keyword_match = (overlap / len(truth_keywords)) * 100
        
        # Length ratio (response should be similar length to ground truth)
        length_ratio = min(
            len(response) / max(len(ground_truth), 1),
            len(ground_truth) / max(len(response), 1)
        ) * 100
        
        # Combined score
        relevance = (keyword_match * 0.6 + length_ratio * 0.4)
        return min(100.0, max(0.0, relevance))
    
    def calculate_completeness_score(self, response: str, expected_keywords: List[str]) -> float:
        """
        Score based on coverage of expected keywords.
        Completeness: 0-100
        """
        if not response or not expected_keywords:
            return 50.0
        
        response_lower = response.lower()
        found_keywords = sum(
            1 for keyword in expected_keywords 
            if keyword.lower() in response_lower
        )
        
        completeness = (found_keywords / len(expected_keywords)) * 100
        return min(100.0, max(0.0, completeness))
    
    def calculate_hallucination_risk(self, response: str, source_type: str) -> float:
        """
        Detect potential hallucination/unsupported claims.
        Risk score: 0-100 (higher = more risky/hallucinated)
        
        Simple heuristics:
        - Suspicious phrases: "according to...", "supposedly", "was reported"
        - Specificity without basis: very specific numbers/dates
        - Contradictions with common knowledge
        """
        if not response:
            return 50.0
        
        risk_score = 0.0
        response_lower = response.lower()
        
        # Check for uncertain language patterns (for "NOT in PDFs" questions, this is OK)
        uncertain_phrases = [
            "i don't know",
            "i'm not sure",
            "it's not mentioned",
            "not in the document",
            "not specified",
            "outside the scope"
        ]
        
        if any(phrase in response_lower for phrase in uncertain_phrases):
            if source_type == "NOT in PDFs":
                # Good - model acknowledged it's not in PDFs
                risk_score = 10.0
            else:
                # Concerning - should have answered from PDFs
                risk_score = 40.0
        else:
            # Check for suspicious specificity
            number_count = len(re.findall(r'\b\d+\b', response))
            if number_count > 5:
                risk_score += 15  # Many numbers might indicate hallucination
            
            # Length (very short = likely hallucination, very long = padding)
            if len(response) < 30:
                risk_score += 20
            elif len(response) > 1500:
                risk_score += 10
        
        return min(100.0, max(0.0, risk_score))
    
    def calculate_answer_quality(self, relevance: float, completeness: float, hallucination: float) -> float:
        """
        Combined quality metric: 0-100
        Formula: (relevance * 0.4 + completeness * 0.4 + (100 - hallucination) * 0.2)
        """
        quality = (
            relevance * 0.4 +
            completeness * 0.4 +
            (100 - hallucination) * 0.2
        )
        return min(100.0, max(0.0, quality))
    
    def evaluate_response(self, question_id: int, response: str, model: str, 
                         ground_truth: str, expected_keywords: List[str], 
                         source_type: str) -> EvaluationScore:
        """Evaluate a single response"""
        
        relevance = self.calculate_relevance_score(response, ground_truth)
        completeness = self.calculate_completeness_score(response, expected_keywords)
        hallucination = self.calculate_hallucination_risk(response, source_type)
        quality = self.calculate_answer_quality(relevance, completeness, hallucination)
        
        # Pass/fail gate: quality >= 70
        passed = quality >= 70
        
        return EvaluationScore(
            question_id=question_id,
            model=model,
            relevance_score=round(relevance, 2),
            completeness_score=round(completeness, 2),
            hallucination_risk=round(hallucination, 2),
            answer_quality=round(quality, 2),
            passed_quality_gate=passed,
            response=response,
            ground_truth=ground_truth
        )
    
    def run_evaluation(self) -> Tuple[List[EvaluationScore], List[EvaluationScore]]:
        """Run evaluation on all questions for both models"""
        
        results_a = []
        results_b = []
        
        print("=" * 80)
        print("Starting LLM Evaluation")
        print(f"Model A: {self.model_a}")
        print(f"Model B: {self.model_b}")
        print(f"Total questions: {len(self.test_data['questions'])}")
        print("=" * 80)
        
        for idx, q_data in enumerate(self.test_data['questions'], 1):
            question = q_data['question']
            question_id = q_data['id']
            ground_truth = q_data['ground_truth_summary']
            expected_keywords = q_data['expected_keywords']
            source_type = q_data['source']
            
            print(f"\n[{idx}/{len(self.test_data['questions'])}] Q{question_id}: {question[:60]}...")
            
            # Query Model A
            response_a = self.query_llm(question, self.model_a)
            score_a = self.evaluate_response(
                question_id, response_a, self.model_a, 
                ground_truth, expected_keywords, source_type
            )
            results_a.append(score_a)
            print(f"  Model A Quality: {score_a.answer_quality}/100 [{'PASS' if score_a.passed_quality_gate else 'FAIL'}]")
            
            # Query Model B
            response_b = self.query_llm(question, self.model_b)
            score_b = self.evaluate_response(
                question_id, response_b, self.model_b,
                ground_truth, expected_keywords, source_type
            )
            results_b.append(score_b)
            print(f"  Model B Quality: {score_b.answer_quality}/100 [{'PASS' if score_b.passed_quality_gate else 'FAIL'}]")
        
        return results_a, results_b
    
    def print_comparison_table(self, results_a: List[EvaluationScore], results_b: List[EvaluationScore]):
        """Print comparison table of metrics"""
        
        print("\n" + "=" * 120)
        print("COMPARISON TABLE: Model A vs Model B")
        print("=" * 120)
        print(f"{'Q#':<3} {'Model A':<8} {'Relevance':<12} {'Complete':<12} {'Halluc':<10} {'Quality':<10} {'Pass':<6}")
        print("-" * 120)
        
        for sa, sb in zip(results_a, results_b):
            print(f"{sa.question_id:<3} {sa.model[:8]:<8} {sa.relevance_score:<12.1f} {sa.completeness_score:<12.1f} {sa.hallucination_risk:<10.1f} {sa.answer_quality:<10.1f} {'✓' if sa.passed_quality_gate else '✗':<6}")
            print(f"{'   '} {sb.model[:8]:<8} {sb.relevance_score:<12.1f} {sb.completeness_score:<12.1f} {sb.hallucination_risk:<10.1f} {sb.answer_quality:<10.1f} {'✓' if sb.passed_quality_gate else '✗':<6}")
            print("-" * 120)
    
    def generate_summary_stats(self, results_a: List[EvaluationScore], results_b: List[EvaluationScore]) -> Dict:
        """Generate summary statistics"""
        
        def calc_stats(results):
            return {
                "avg_relevance": round(sum(r.relevance_score for r in results) / len(results), 2),
                "avg_completeness": round(sum(r.completeness_score for r in results) / len(results), 2),
                "avg_hallucination": round(sum(r.hallucination_risk for r in results) / len(results), 2),
                "avg_quality": round(sum(r.answer_quality for r in results) / len(results), 2),
                "pass_count": sum(1 for r in results if r.passed_quality_gate),
                "pass_rate": round(sum(1 for r in results if r.passed_quality_gate) / len(results) * 100, 1)
            }
        
        return {
            "model_a": {self.model_a: calc_stats(results_a)},
            "model_b": {self.model_b: calc_stats(results_b)}
        }
    
    def print_summary_stats(self, stats: Dict):
        """Print summary statistics"""
        
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS")
        print("=" * 100)
        
        model_a_name = self.model_a
        model_b_name = self.model_b
        
        stats_a = list(stats["model_a"].values())[0]
        stats_b = list(stats["model_b"].values())[0]
        
        print(f"\n{'Metric':<25} {model_a_name:<30} {model_b_name:<30}")
        print("-" * 85)
        print(f"{'Avg Relevance Score':<25} {stats_a['avg_relevance']:<30.2f} {stats_b['avg_relevance']:<30.2f}")
        print(f"{'Avg Completeness Score':<25} {stats_a['avg_completeness']:<30.2f} {stats_b['avg_completeness']:<30.2f}")
        print(f"{'Avg Hallucination Risk':<25} {stats_a['avg_hallucination']:<30.2f} {stats_b['avg_hallucination']:<30.2f}")
        print(f"{'Avg Quality Score':<25} {stats_a['avg_quality']:<30.2f} {stats_b['avg_quality']:<30.2f}")
        print(f"{'Pass Count':<25} {stats_a['pass_count']}/25 {' ':<24} {stats_b['pass_count']}/25")
        print(f"{'Pass Rate':<25} {stats_a['pass_rate']:.1f}% {' ':<26} {stats_b['pass_rate']:.1f}%")
        print("=" * 100)
    
    def find_failures(self, results_a: List[EvaluationScore], results_b: List[EvaluationScore]) -> List[Dict]:
        """Find top 5 failure cases"""
        
        all_results = []
        
        for r in results_a:
            if not r.passed_quality_gate:
                all_results.append({
                    "question_id": r.question_id,
                    "model": r.model,
                    "quality": r.answer_quality,
                    "hallucination": r.hallucination_risk,
                    "response": r.response,
                    "reason": self._categorize_failure(r)
                })
        
        for r in results_b:
            if not r.passed_quality_gate:
                all_results.append({
                    "question_id": r.question_id,
                    "model": r.model,
                    "quality": r.answer_quality,
                    "hallucination": r.hallucination_risk,
                    "response": r.response,
                    "reason": self._categorize_failure(r)
                })
        
        # Sort by quality score (worst first)
        all_results.sort(key=lambda x: x['quality'])
        
        return all_results[:5]
    
    def _categorize_failure(self, score: EvaluationScore) -> str:
        """Categorize why a response failed"""
        
        if score.hallucination_risk > 60:
            return "High hallucination risk"
        elif score.completeness_score < 30:
            return "Incomplete answer"
        elif score.relevance_score < 30:
            return "Low relevance to ground truth"
        else:
            return "Poor overall quality"
    
    def print_failure_analysis(self, failures: List[Dict]):
        """Print top failure examples"""
        
        print("\n" + "=" * 100)
        print("TOP 5 FAILURE EXAMPLES")
        print("=" * 100)
        
        for idx, failure in enumerate(failures, 1):
            print(f"\n{idx}. Q{failure['question_id']} - {failure['model']}")
            print(f"   Quality: {failure['quality']}/100 | Hallucination Risk: {failure['hallucination']}/100")
            print(f"   Reason: {failure['reason']}")
            print(f"   Response: {failure['response'][:150]}...")
    
    def save_results(self, results_a: List[EvaluationScore], results_b: List[EvaluationScore], stats: Dict):
        """Save results to JSON"""
        
        output = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_a": self.model_a,
            "model_b": self.model_b,
            "total_questions": len(results_a),
            "temperature_setting": 0.0,
            "results_model_a": [asdict(r) for r in results_a],
            "results_model_b": [asdict(r) for r in results_b],
            "summary_statistics": stats
        }
        
        with open("evaluation_results.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print("\n✓ Results saved to evaluation_results.json")


def main():
    """Run the evaluation"""
    
    evaluator = LLMEvaluator()
    
    # Run evaluation
    results_a, results_b = evaluator.run_evaluation()
    
    # Generate statistics
    stats = evaluator.generate_summary_stats(results_a, results_b)
    
    # Print results
    evaluator.print_comparison_table(results_a, results_b)
    evaluator.print_summary_stats(stats)
    
    # Find and print failures
    failures = evaluator.find_failures(results_a, results_b)
    evaluator.print_failure_analysis(failures)
    
    # Save results
    evaluator.save_results(results_a, results_b, stats)


if __name__ == "__main__":
    main()
