import pandas as pd
from utils.data_tools import get_associated_info

def get_teacher_ai_prompt(row_dict):
    """
    Generate system and user prompts to analyze the teacher data.
    """
    system_prompt = """You are a senior academic talent evaluator and HR assistant. Your task is to deeply analyze and summarize the academic background, expertise, and main achievements based on the detailed metadata of the provided researcher or teacher profile.
Key Requirements:
1. The output must be perfectly fluent and professional, strictly entirely in English.
2. The evaluation should be highly positive, fully exploring and highlighting their academic contributions, teaching experience, and real-world application value within their field. The tone should be affirmative and encouraging.
3. Keep the structure clear. It is highly recommended to use Markdown formatting, including the following sections:
   - Profile Overview
   - Key Expertise & Research Areas
   - Research Trajectory & Achievements Over Time
4. Maintain a professional and objective corporate tone without using overly colloquial internet slang, while keeping the overall impression encouraging and elevated.
    """
    
    user_data = []
    # Extract data, filtering out raw_json and empty items
    for k, v in row_dict.items():
        if k == 'raw_json': 
            continue
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v): 
            continue
        if isinstance(v, (list, dict, str)) and len(v) == 0:
            continue
        user_data.append(f"- **{k}**: {v}")
        
    associated_info_text = get_associated_info('teacher', row_dict)
        
    user_content = (
        "Here is the detailed profile metadata for this researcher/teacher:\n"
        + "\n".join(user_data)
        + "\n" + associated_info_text
        + "\n\nBased on the information above, please explore and carefully summarize the academic background, expertise, and main achievements of this individual. The summary should innovatively articulate the evolution of the teacher's research directions and achievements over time, and explore potential future research trajectories based on these interconnected works. The content must be concise and avoiding redundancy. Each content under a section should be no more than 2 sentences. Make sure each sentence is short and concise. "
        + "\n\nPlease ensure that the output is positive and highlights their academic contributions, teaching experience, and professional value. "
        + "\n\n Please ensure there is a '##' before each section subtitle as in markdown format. And the items in each section should not descibed in bullet point, please group them into one paragraph."
    )
    
    return system_prompt, user_content