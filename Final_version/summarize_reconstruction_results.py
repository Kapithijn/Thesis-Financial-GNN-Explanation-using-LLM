#!/usr/bin/env python3
"""Create grouped reconstruction summaries from a raw results JSON file."""
import argparse
import csv
import json
from pathlib import Path

from Evalueation import evaluate_reconstruction


GROUP_SPECS = [
	("dataset", "model", "llm"),
	("dataset", "model"),
	("dataset", "llm"),
	("model",),
	("llm",),
]


def build_grouped_summaries(rows):
	"""Return flat grouped reconstruction metrics for plotting."""
	summaries = []
	for group_fields in GROUP_SPECS:
		groups = {}
		for row in rows:
			key = tuple(str(row.get(field, "unknown")) for field in group_fields)
			groups.setdefault(key, []).append(row)
		for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
			summary = {
				"group_by": "+".join(group_fields),
				"n": len(group_rows),
			}
			for field, value in zip(group_fields, key):
				summary[field] = value
			summary.update(evaluate_reconstruction(group_rows))
			summaries.append(summary)
	return summaries


def write_csv(rows, path):
	"""Write rows with stable columns even when grouping fields differ."""
	preferred = [
		"group_by",
		"n",
		"dataset",
		"model",
		"llm",
		"precision",
		"recall",
		"f1",
		"jaccard",
		"overlap",
		"edit_distance",
	]
	keys = preferred + sorted({key for row in rows for key in row.keys()} - set(preferred))
	with Path(path).open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=keys)
		writer.writeheader()
		writer.writerows(rows)


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("raw_json", help="Path to results_raw_reconstruction_*.json")
	parser.add_argument(
		"--out-prefix",
		default=None,
		help="Output prefix without extension. Defaults to '<raw_json stem>_grouped'.",
	)
	args = parser.parse_args(argv)

	raw_path = Path(args.raw_json)
	rows = json.loads(raw_path.read_text(encoding="utf-8"))
	if not isinstance(rows, list):
		raise TypeError(f"Expected a list of raw rows in {raw_path}")

	out_prefix = Path(args.out_prefix) if args.out_prefix else raw_path.with_suffix("")
	if args.out_prefix is None:
		out_prefix = out_prefix.with_name(out_prefix.name + "_grouped")
	out_prefix.parent.mkdir(parents=True, exist_ok=True)

	summaries = build_grouped_summaries(rows)
	json_path = out_prefix.with_suffix(".json")
	csv_path = out_prefix.with_suffix(".csv")
	json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
	write_csv(summaries, csv_path)

	print(f"Wrote {len(summaries)} grouped summaries")
	print(f"JSON: {json_path}")
	print(f"CSV:  {csv_path}")


if __name__ == "__main__":
	raise SystemExit(main())
