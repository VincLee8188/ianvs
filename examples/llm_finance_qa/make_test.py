import json

def main(node):
    # Input JSON file containing Q&A data
    input_file = f"./FinReportQA/report_{node}_qa.json"
    # Output JSONL file where each line is a JSON object with the new question and the original answer
    output_file = "data.jsonl"
    
    # Open and load the input JSON file
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    
    # Open the output file in write mode
    with open(output_file, "w", encoding="utf-8") as outfile:
        # Iterate over each sample in the input data
        for sample in data:
            # Get the original context, question, and answer; use empty string as fallback if not present
            context = sample.get("context", "")
            original_question = sample.get("question", "")
            answer = sample.get("answer", "")
            
            # Build the new question using the provided template
            new_question = (
                "You are an intelligent financial assistant. Given the following context from an annual report and a question, "
                "please provide a concise answer in English.\n\n"
                "--- Context ---\n"
                f"{context}\n\n"
                "--- Question ---\n"
                f"{original_question}\n\n"
                "Output only the answer in English, without additional commentary."
            )
            
            # Construct a new sample with only the required keys
            new_sample = {
                "question": new_question,
                "answer": answer
            }
            
            # Write the new sample as a JSON line into the output file
            outfile.write(json.dumps(new_sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    node = 20
    main(node)
