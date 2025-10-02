#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of Music Theory AI Agent
"""

import os
import sys

# Add the parent directory to the path so we can import the agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_pipeline_optimized_9.14 import run_episode

def main():
    """Example of running the agent on a single question."""
    
    # Set up environment variables
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
    os.environ["QUESTIONS_CSV_PATH"] = "examples/test.csv"
    os.environ["KERN_FOLDER"] = "data/kern_exam_new"
    
    # Run a single question
    question_id = "1"
    
    print(f"Running analysis for question {question_id}...")
    print("=" * 50)
    
    try:
        prediction, state = run_episode(question_id)
        
        print(f"Question ID: {state.question_id}")
        print(f"Music ID: {state.music_id}")
        print(f"Question: {state.question}")
        print(f"Prediction: {prediction}")
        print(f"Ground Truth: {state.answer_gt}")
        print(f"Correct: {prediction == state.answer_gt}")
        
    except Exception as e:
        print(f"Error running analysis: {e}")

if __name__ == "__main__":
    main()
