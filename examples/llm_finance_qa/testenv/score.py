import os
from openai import OpenAI
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["score"]

# Instantiate a global client using the official style
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # API key is taken from the environment variable
)

def score_with_gpt4o(reference, answer, model="gpt-4o"):
    """
    Use GPT-4o to score the predicted answer against the reference answer.
    
    The prompt instructs GPT-4o to output a numeric score (0 to 10) where:
      - 10 indicates an answer that exactly matches the reference,
      - 5 indicates a partially correct answer,
      - 0 indicates a completely incorrect answer.
    
    Parameters:
      reference (str): The ground truth (reference) answer.
      answer (str): The predicted answer to evaluate.
      model (str): The GPT model to use.
    
    Returns:
      float: The numeric score given by GPT-4o.
    """
    prompt = (
        "You are a strict evaluator of financial text answers. "
        "Given the reference answer: \"{reference}\" and the predicted answer: \"{answer}\", "
        "please score the predicted answer on a scale of 0 to 10, where 10 means an exact match, "
        "5 means partially correct, and 0 means completely incorrect. "
        "Output only the numeric score."
    ).format(reference=reference, answer=answer)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        score_str = response.choices[0].message.content.strip()
        score_value = float(score_str)
        return score_value
    except Exception as e:
        print(f"[ERROR] GPT scoring failed: {e}")
        return 0.0

@ClassFactory.register(ClassType.GENERAL, alias="score")
def score(y_true, y_pred):
    """
    Calculate the average GPT-4o score of predicted answers compared to reference answers.
    
    For each sample, the function calls GPT-4o with a prompt that includes the reference answer 
    and the predicted answer. GPT-4o returns a numeric score (0-10) which is then averaged across 
    all samples.
    
    Parameters:
      y_true (list of str): List of reference answers.
      y_pred (list of str): List of predicted answers.
    
    Returns:
      float: The average score across all samples.
    """
    scores = []
    for ref, pred in zip(y_true, y_pred):
        score_value = score_with_gpt4o(ref, pred)
        scores.append(score_value)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score
