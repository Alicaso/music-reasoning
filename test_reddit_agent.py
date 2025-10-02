#!/usr/bin/env python3
"""
Test script for reddit_data_combined_final.csv format
Handles questions with separate final_options column and truth_letter column
"""

import os
import sys
import pandas as pd
import ast
import subprocess

# Set environment variables BEFORE importing agent_pipeline_optimized
os.environ["QUESTIONS_CSV_PATH"] = "reddit_data_combined_final.csv"
os.environ["KERN_FOLDER"] = "kern_reddit_new"

def format_question_with_options(question, final_options):
    """Format question with options from final_options list"""
    formatted_question = question + "\n"
    
    # Parse the options list if it's a string
    if isinstance(final_options, str):
        try:
            options = ast.literal_eval(final_options)
        except:
            # Fallback: split by comma and clean up
            options = [opt.strip().strip('"\'') for opt in final_options.split(',')]
    else:
        options = final_options
    
    # Add options with A, B, C, D labels
    for i, option in enumerate(options):
        letter = chr(ord('A') + i)  # A, B, C, D
        formatted_question += f"{letter}. {option}\n"
    
    return formatted_question

def test_reddit_questions(csv_path="reddit_data_combined_final.csv", kern_folder="kern_reddit_new", pipeline="agent_pipeline_optimized_9.14.py", start_from=1):
    """Test questions in reddit format starting from a specific question number"""
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} questions from {csv_path}")
    
    # Filter to start from the specified question number
    # Note: question_id in reddit dataset is string, so we'll process all questions
    df_filtered = df.reset_index(drop=True)
    print(f"Testing all {len(df_filtered)} questions")
    
    correct = 0
    total = 0
    results = []
    
    for idx, row in df_filtered.iterrows():
        question_id = str(row['question_id'])
        print(f"\n{'='*50}")
        print(f"Testing question {idx + 1}/{len(df)}: {question_id}")
        print(f"{'='*50}")
        
        try:
            # Format question with options
            formatted_question = format_question_with_options(row['question'], row['final_options'])
            print(f"Question: {formatted_question}")
            print(f"Ground truth: {row['truth_letter']}")
            
            # Run the agent using subprocess to generate logs
            cmd = [
                "python3", pipeline,
                "--question_id", question_id
            ]
            
            # Set environment variables for the subprocess
            env = os.environ.copy()
            env["QUESTIONS_CSV_PATH"] = csv_path
            env["KERN_FOLDER"] = kern_folder
            
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
            is_correct = prediction.upper() == str(row['truth_letter']).upper()
            if is_correct:
                correct += 1
            total += 1
            
            accuracy = (correct * 100) // total if total > 0 else 0
            print(f"Correct: {is_correct} | Running accuracy: {correct}/{total} ({accuracy}%)")
            
            # Store result
            results.append({
                'question_id': question_id,
                'question': row['question'],
                'ground_truth': row['truth_letter'],
                'prediction': prediction,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            total += 1
            results.append({
                'question_id': question_id,
                'question': row['question'],
                'ground_truth': row['truth_letter'],
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
    results_df.to_csv("reddit_test_results.csv", index=False)
    print(f"Detailed results saved to reddit_test_results.csv")
    
    return results_df

if __name__ == "__main__":
    # Set up environment
    # Note: Set OPENAI_API_KEY environment variable before running
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", type=str, default="agent_pipeline_optimized_9.14.py", help="Pipeline script to run for each question")
    ap.add_argument("--csv", type=str, default="reddit_data_combined_final.csv")
    ap.add_argument("--kern", type=str, default="kern_reddit_new")
    ap.add_argument("--start_from", type=int, default=1, help="Start testing from this question ID")
    args = ap.parse_args()

    # Test questions starting from specified ID
    results = test_reddit_questions(csv_path=args.csv, kern_folder=args.kern, pipeline=args.pipeline, start_from=args.start_from)