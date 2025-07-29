import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Any
import openai
from huggingface_hub import create_repo


class DatasetBuilder:
    """Class to build a dataset with reasoning capabilities with Magistral."""

    # Magistral-Small official system prompt
    SYS_PROMPT = """A user will ask you to solve a task. You must first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \boxed{} in your response.

        Your thinking process must follow the template below:
        <think>
        Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
        </think>

        Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.

        Problem:
        """

    def __init__(
        self,
        client: openai.Client,
        model_name: str,
    ):
        self.client = client
        self.model_name = model_name

    def _build_api_messages(self, question: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": question},
        ]

    def _make_api_call(self, messages: List[Dict[str, Any]]) -> str:
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=8192,
                    top_p=0.95,
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception:
                continue
        raise RuntimeError("API call failed after 3 attempts")

    def generate(self, dataset_sample) -> List[Dict[str, Any]]:
        question = dataset_sample["messages"][0]["content"]
        answer = self._make_api_call(self._build_api_messages(question))
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]


def get_client(base_url: str, api_key: str) -> openai.Client:
    """Get a client for the model API."""
    return openai.Client(base_url=base_url, api_key=api_key)


def generation_pipeline(
    dataset, client: openai.Client, model_name: str, num_workers: int
):
    """Generate a dataset."""
    ds_builder = DatasetBuilder(client, model_name)
    generated = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(ds_builder.generate, sample) for sample in dataset]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing data"
        ):
            try:
                result = future.result()
                generated.append({"messages": result})
            except Exception as e:
                print(f"Error: {e}")
    print(f"Generated dataset: {len(generated)} samples")
    return generated


def main(args):
    # Load dataset
    dataset = load_dataset(args.data_repo, split="train")

    # Generate samples
    client = get_client(args.base_url, args.api_key)
    generated = generation_pipeline(dataset, client, args.model_name, args.num_workers)

    # Ensure HF repo exists
    create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True)

    # Push to Hub
    hf_ds = HFDataset.from_list(generated)
    hf_ds.push_to_hub(repo_id=args.hf_repo, private=True)

    print(f"Pushed to {args.hf_repo} with {len(generated)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data and push to HF.")
    parser.add_argument(
        "--data_repo", type=str, default="kurakurai/smoltalk2-french-reasoning"
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers for generation."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the API.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the model API.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Magistral-Small-2506",
        help="Model name to use for generation.",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="kurakurai/smoltalk2-french-reasoning-true",
        help="Your Hugging Face repository name.",
    )
    args = parser.parse_args()
    main(args)
