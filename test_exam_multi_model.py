#!/usr/bin/env python3
"""
Test multiple models on exam dataset with simple prompt
"""

import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import re
import google.generativeai as genai
import anthropic
import time

load_dotenv()

# Initialize API clients
openai_client = OpenAI()
# Configure API clients with environment variables
# Set GEMINI_API_KEY and ANTHROPIC_API_KEY environment variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

def get_llm_response(prompt: str, model: str = "gpt-4o") -> str:
    """Send the prompt to the LLM and return the text response."""
    if model == "deepseek-chat":
        # Use DeepSeek API with custom client
        from openai import OpenAI
        deepseek_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com/v1"
        )
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    elif model == "gemini-1.5-pro":
        # Use Gemini API
        model_instance = genai.GenerativeModel('gemini-1.5-pro')
        response = model_instance.generate_content(prompt)
        return response.text.strip()
    elif model == "claude-3-7-sonnet-20250219":
        # Use Claude API with 3.7 Sonnet model
        response = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    else:
        # Use OpenAI API
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

valid_letters = "a, b, c, d"

def build_prompt(question: str, kern_text: str) -> str:
    """Construct a simple prompt with kern content and the question."""
    return (
        f"Kern Format:\n{kern_text}\n\n"
        f"Question:\n{question}\n\n"
        f"You are provided with a kern format of a musical score and a question on symbolic music theory. Please choose the correct answer from the options provided. Respond only with the option letter ({valid_letters})."
    )

def extract_answer(response: str) -> str:
    """Extract the answer letter from the model response."""
    response = response.strip().lower()
    # Look for single letter at the end
    match = re.search(r'([a-d])$', response)
    if match:
        return match.group(1)
    # Look for any single letter in the response
    match = re.search(r'([a-d])', response)
    if match:
        return match.group(1)
    return "invalid"

def calculate_accuracy(predictions: list, ground_truth: list) -> float:
    """Calculate accuracy given predictions and ground truth."""
    correct = 0
    total = 0
    for pred, truth in zip(predictions, ground_truth):
        if pred != "file_not_found" and pred != "processing_error" and pred != "invalid":
            total += 1
            if pred == truth.lower():
                correct += 1
    return correct / total if total > 0 else 0.0

# Load your dataframe
df = pd.read_csv("test.csv")
df = df.iloc[:41]  # Use first 41 questions

# Define models to test
models = {
    "deepseek-chat": "DeepSeek-V3.1",
    "gemini-1.5-pro": "Gemini 1.5 Pro", 
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet"
}

# Initialize results storage
results = {model: [] for model in models.keys()}

print(f"Processing {len(df)} exam questions with {len(models)} models", flush=True)

for idx, row in df.iterrows():
    print(f"\nProcessing question {idx + 1}/{len(df)}", flush=True)
    # Use the music_id from the current row (not question_id)
    music_id = str(row['music_id']).zfill(3)  # Pad with zeros to 3 digits
    kern_file_path = os.path.join("kern_exam_new", f"{music_id}.krn")
    
    try:
        with open(kern_file_path, 'r') as kern_file:
            kern_content = kern_file.read()
        
        # Build the prompt using the question text and kern content
        prompt = build_prompt(row['question'], kern_content)
        
        # Test each model
        for model_key, model_name in models.items():
            try:
                print(f"Testing {model_name}...", flush=True)
                answer = get_llm_response(prompt, model_key)
                extracted_answer = extract_answer(answer)
                results[model_key].append(extracted_answer)
                print(f"{model_name} answer: {answer[:100]}... -> extracted: {extracted_answer}", flush=True)
                
                # Add delay to avoid rate limiting
                time.sleep(1.0)
                
            except Exception as e:
                error_msg = f"Error with {model_name} on question {idx + 1}: {e}"
                print(error_msg, flush=True)
                results[model_key].append("processing_error")
        
        print(f"Question {idx + 1} completed successfully", flush=True)
        
    except FileNotFoundError:
        error_msg = f"Kern file not found: {kern_file_path}"
        print(error_msg, flush=True)
        for model_key in models.keys():
            results[model_key].append("file_not_found")
    except Exception as e:
        error_msg = f"Error processing question {idx + 1}: {e}"
        print(error_msg, flush=True)
        for model_key in models.keys():
            results[model_key].append("processing_error")

# Add results to dataframe
for model_key, model_name in models.items():
    df[f'predicted_answer_{model_key}'] = results[model_key]

# Calculate and print accuracies
print("\n" + "="*50)
print("EXAM DATASET ACCURACY RESULTS")
print("="*50)

ground_truth = df['truth_letter'].tolist()

for model_key, model_name in models.items():
    predictions = results[model_key]
    accuracy = calculate_accuracy(predictions, ground_truth)
    print(f"{model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Save results
df.to_csv("exam_multi_model_predictions.csv", index=False)
print(f"\nResults saved to exam_multi_model_predictions.csv", flush=True)