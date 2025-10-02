#!/usr/bin/env python3
"""
Test script for test.csv format
"""

import os
import pandas as pd
import subprocess

def test_exam_questions(csv_path="test.csv", pipeline="agent_pipeline_optimized.py"):
    """Test all questions in test.csv format"""
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} questions from {csv_path}")
    
    correct = 0
    total = 0
    results = []
    
    for idx, row in df.iterrows():
        question_id = str(row['question_id'])
        print(f"\n{'='*50}")
        print(f"Testing question {idx + 1}/{len(df)}: {question_id}")
        print(f"{'='*50}")
        
        try:
            # Format question with options
            formatted_question = format_question_with_options(row['question'], row['final_options'])
            print(f"Question: {formatted_question}")
            ground_truth = str(row['truth_letter']).upper()
            print(f"Ground truth: {ground_truth}")
            
            # Run the agent using subprocess to generate logs
            cmd = [
                "python3", pipeline,
                "--question_id", question_id
            ]
            
            # Set environment variables for the subprocess
            env = os.environ.copy()
            env["QUESTIONS_CSV_PATH"] = csv_path
            env["KERN_FOLDER"] = "kern_exam_new"
            # Clear any conflicting environment variables
            if "KERN_FOLDER" in env and env["KERN_FOLDER"] == "kern_reddit_new":
                del env["KERN_FOLDER"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                print(f"Error running agent for {question_id}: {result.stderr}")
                prediction = "ERROR"
            else:
                # Parse prediction from output
                lines = result.stdout.strip().split('\n')
                prediction = None
                for line in lines:
                    if line.startswith("Prediction:"):
                        prediction = line.split(":", 1)[1].strip()
                        break
                
                if prediction is None:
                    prediction = "ERROR"
            
            print(f"Prediction: {prediction}")
            
            # Check if correct
            is_correct = prediction.upper() == ground_truth
            if is_correct:
                correct += 1
            total += 1
            
            accuracy = (correct * 100) // total if total > 0 else 0
            print(f"Correct: {is_correct} | Running accuracy: {correct}/{total} ({accuracy}%)")
            
            # Store result
            results.append({
                'question_id': question_id,
                'question': row['question'],
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            total += 1
            results.append({
                'question_id': question_id,
                'question': row['question'],
                'ground_truth': str(row['truth_letter']).upper(),
                'prediction': 'ERROR',
                'correct': False
            })
    
    # Final results
    final_accuracy = (correct * 100) // total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {correct}/{total} correct ({final_accuracy}% accuracy)")
    print(f"{'='*60}")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv("exam_test_results.csv", index=False)
    print(f"Detailed results saved to exam_test_results.csv")
    
    return results_df

def format_question_with_options(question, options_str):
    """Format question with options for display"""
    formatted_question = question + "\n\n"
    
    # Parse options from the list format
    import ast
    try:
        options = ast.literal_eval(options_str)
    except:
        options = []
    
    # Add options with A, B, C, D labels
    for i, option in enumerate(options):
        letter = chr(ord('A') + i)  # A, B, C, D
        formatted_question += f"{letter}. {option}\n"
    
    return formatted_question

if __name__ == "__main__":
    # Set up environment
    # Note: Set OPENAI_API_KEY environment variable before running
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", type=str, default="agent_pipeline_optimized.py", help="Pipeline script to run for each question")
    ap.add_argument("--csv", type=str, default="test.csv")
    args = ap.parse_args()

    # Test all questions
    results = test_exam_questions(csv_path=args.csv, pipeline=args.pipeline)