import sys
from rag import answer_question

def main():
    if len(sys.argv) < 3:
        print("Usage: python cli.py <document_path> <question>")
        sys.exit(1)

    filepath = sys.argv[1]
    question = sys.argv[2]

    result = answer_question(question, filepath)

    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"\n[{i}] {source[:200]}...")

if __name__ == "__main__":
    main()