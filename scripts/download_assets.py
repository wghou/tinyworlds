import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from huggingface_hub import hf_hub_download, list_repo_files

DATASET_REPO_ID_DEFAULT = "datasets/AlmondGod/tinyworlds"
MODELS_REPO_ID_DEFAULT = "AlmondGod/tinyworlds-models"


def repo_root() -> Path:
	return Path(__file__).resolve().parents[1]


def expand_patterns(repo_id: str, patterns: List[str], repo_type: str) -> List[str]:
	if not patterns:
		return []
	files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
	matched: List[str] = []
	from fnmatch import fnmatch
	for f in files:
		if any(fnmatch(f, pat) for pat in patterns):
			matched.append(f)
	return matched


def download_pairs(pairs: List[Tuple[str, str]], output_dir: Path, resume: bool = True, repo_type: str = "model") -> List[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	paths: List[Path] = []
	for repo_id, filename in pairs:
		p = hf_hub_download(
			repo_id=repo_id,
			filename=filename,
			repo_type=repo_type,
			local_dir=output_dir,
			local_dir_use_symlinks=False,
			resume_download=resume,
		)
		paths.append(Path(p))
	return paths


def cmd_datasets(args) -> int:
	# Defaults to datasets repo and repo_root/data
	repo_id = args.repo or DATASET_REPO_ID_DEFAULT
	data_dir = args.out or (repo_root() / "data")
	patterns = args.pattern or ["*.h5", "*.json", "*.md", "assets/*"]

	matched = expand_patterns(repo_id, patterns, repo_type="dataset")
	if not matched:
		print(f"No files matched in dataset repo {repo_id} for patterns {patterns}")
		return 1

	pairs = [(repo_id, m) for m in matched]
	paths = download_pairs(pairs, data_dir, resume=(not args.no_resume), repo_type="dataset")
	for p in paths:
		print(p)
	return 0


def cmd_models(args) -> int:
	# Defaults to model repo and results/<timestamp>_<suite>/
	repo_id = args.repo or MODELS_REPO_ID_DEFAULT
	base_results = args.out or (repo_root() / "results")
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	suite = args.suite_name or "models"
	folder_name = f"{timestamp}_{suite}"
	model_root = base_results / folder_name

	model_type = args.type
	if model_type not in {"video_tokenizer", "latent_actions", "dynamics"}:
		print("--type must be one of: video_tokenizer, latent_actions, dynamics")
		return 2

	# Default patterns: checkpoints for the selected model type
	patterns = args.pattern or [f"{model_type}/checkpoints/*.pth"]
	matched = expand_patterns(repo_id, patterns, repo_type="model")
	if not matched:
		print(f"No files matched in model repo {repo_id} for patterns {patterns}")
		return 1

	# Target directory: results/<timestamp>_<suite>/<model_type>/checkpoints
	target_root = model_root / model_type / "checkpoints"
	pairs = [(repo_id, m) for m in matched]
	paths = download_pairs(pairs, target_root, resume=(not args.no_resume), repo_type="model")
	for p in paths:
		print(p)
	return 0


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Download datasets/checkpoints from Hugging Face Hub")
	subparsers = parser.add_subparsers(dest="cmd")

	# datasets subcommand
	p_data = subparsers.add_parser("datasets", help="Download dataset files into repo_root/data")
	p_data.add_argument("--repo", type=str, help="HF dataset repo_id (default: AlmondGod/tinyworlds)")
	p_data.add_argument("--pattern", action="append", help="Glob(s) to match within dataset repo (default: *.h5, assets/*)")
	p_data.add_argument("--out", type=Path, help="Target directory (default: repo_root/data)")
	p_data.add_argument("--no-resume", action="store_true", help="Disable resume for interrupted downloads")
	p_data.set_defaults(func=cmd_datasets)

	# models subcommand
	p_models = subparsers.add_parser("models", help="Download model checkpoints into results/<timestamp>_<suite>/<type>/checkpoints")
	p_models.add_argument("--repo", type=str, help="HF model repo_id (default: AlmondGod/tineyworlds-models)")
	p_models.add_argument("--type", required=True, help="Model type: video_tokenizer | latent_actions | dynamics")
	p_models.add_argument("--suite-name", dest="suite_name", type=str, help="Folder name suffix (e.g., 'sonic_models')")
	p_models.add_argument("--pattern", action="append", help="Glob(s) to match within model repo (default: <type>/checkpoints/*.pth)")
	p_models.add_argument("--out", type=Path, help="Base results directory (default: repo_root/results)")
	p_models.add_argument("--no-resume", action="store_true", help="Disable resume for interrupted downloads")
	p_models.set_defaults(func=cmd_models)

	return parser


def main(argv=None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	if not getattr(args, "cmd", None):
		parser.print_help()
		return 2
	return args.func(args)


if __name__ == "__main__":
	sys.exit(main()) 