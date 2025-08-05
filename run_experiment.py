import argparse
from analyzer.similarity import compare_responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", type=str, required=True, help="Persona name (e.g., coach-v1)")
    parser.add_argument("--models", nargs="+", required=True, help="List of model names to compare (e.g., gpt gemini)")
    args = parser.parse_args()

    # 샘플 경로는 실제 구성에 맞게 조정
    gpt_path = f"data/inputs/{args.persona}_gpt.txt"
    gemini_path = f"data/inputs/{args.persona}_gemini.txt"

    compare_responses(gpt_path, gemini_path, out_path=f"data/outputs/{args.persona}_comparison.csv")
