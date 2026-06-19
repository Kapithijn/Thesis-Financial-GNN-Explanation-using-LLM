import unittest

import LLM_Module as llm_module
from Evalueation import compute_classification_metrics
from LLM_Module import generate_response, parse_neighbor_selection_response, parse_prediction, run_inference_all


class ParsePredictionTests(unittest.TestCase):
    def test_exact_prompt_format(self):
        self.assertEqual(parse_prediction("The predicted class is 1"), 1)

    def test_markdown_wrapped_binary_label(self):
        self.assertEqual(parse_prediction("The predicted class is: **0** (licit)."), 0)

    def test_json_response(self):
        self.assertEqual(parse_prediction('{"predicted_class": "1"}'), 1)

    def test_fenced_json_response(self):
        response = '```json\n{"answer": 0, "confidence": 0.76}\n```'
        self.assertEqual(parse_prediction(response), 0)

    def test_final_bare_label_after_reasoning(self):
        response = "The embedding resembles the illicit example.\n\nFinal answer:\n1"
        self.assertEqual(parse_prediction(response), 1)

    def test_textual_label_response(self):
        self.assertEqual(parse_prediction("The transaction is illicit."), 1)
        self.assertEqual(parse_prediction("The transaction is suspicious."), 1)

    def test_classification_key_response(self):
        self.assertEqual(parse_prediction("Classification: 0"), 0)

    def test_common_freeform_choice_response(self):
        self.assertEqual(parse_prediction("I choose 0."), 0)
        self.assertEqual(parse_prediction("I predict 1."), 1)
        self.assertEqual(parse_prediction("I would classify it as licit."), 0)

    def test_tail_answer_wins_over_prompt_echo(self):
        response = (
            "Class definition:\n"
            "- 0 = licit\n"
            "- 1 = illicit\n\n"
            "Example\n"
            "Correct label: 1\n\n"
            "Now classify the following case.\n"
            "The predicted class is 0"
        )
        self.assertEqual(parse_prediction(response), 0)

    def test_instruction_without_answer_stays_unknown(self):
        response = "where X is either 0 (licit) or 1 (illicit). Do not output anything else."
        self.assertEqual(parse_prediction(response), "Unknown")


class ClassificationMetricTests(unittest.TestCase):
    def test_unknown_prediction_does_not_crash_metrics(self):
        rows = [{"gnn_pred": 1, "llm_pred": "Unknown"}]
        metrics = compute_classification_metrics(rows)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["parse_rate"], 0.0)
        self.assertEqual(metrics["unknown_n"], 1)
        self.assertEqual(rows[0]["llm_pred"], 2)


class NeighborSelectionParseTests(unittest.TestCase):
    def test_embedded_json_array_wins_over_prose_numbers(self):
        response = (
            "Use a threshold of 0.5.\n"
            "```json\n"
            '{"selected_neighbors": [18186, 89428], "confidence": 0.8}\n'
            "```"
        )
        self.assertEqual(parse_neighbor_selection_response(response), [18186, 89428])

    def test_truncated_array_can_still_be_filtered_to_candidates(self):
        response = (
            "Here is the JSON object:\n"
            '{"selected_neighbors": [18186, 89428, 1983'
        )
        self.assertEqual(
            parse_neighbor_selection_response(response, allowed_ids=[18186, 89428, 198399]),
            [18186, 89428],
        )

    def test_selected_neighbors_phrase_is_parsed(self):
        response = "The selected neighbors are [12, 34] with confidence 0.9."
        self.assertEqual(parse_neighbor_selection_response(response, allowed_ids=[12, 34, 56]), [12, 34])

    def test_unlabeled_prose_numbers_are_ignored(self):
        response = "I used threshold 0.5 and selected 12 and 34 with confidence 0.9."
        self.assertEqual(parse_neighbor_selection_response(response, allowed_ids=[12, 34, 56]), [])

    def test_thinking_schema_echo_does_not_win_over_final_answer(self):
        response = (
            "Thinking Process:\n"
            "The requested output format is {\"selected_neighbors\": [], \"confidence\": 0.0}.\n"
            "Now compare candidate ids.\n\n"
            "Final answer:\n"
            "{\"selected_neighbors\": [12, 34], \"confidence\": 0.6}"
        )
        self.assertEqual(parse_neighbor_selection_response(response, allowed_ids=[12, 34, 56]), [12, 34])


class GenerateResponseTests(unittest.TestCase):
    def test_neighbor_selection_prompt_includes_candidate_context(self):
        template = (
            "Target:\n{embedding}\n"
            "Target features:\n{target_features}\n"
            "Candidates:\n{candidates}\n"
            "Context:\n{candidate_context}"
        )

        prompt = llm_module.build_neighbor_selection_prompt(
            "embedding: [0.1000]",
            "12, 34",
            template,
            target_features_text="[1.0000, 2.0000]",
            candidate_context_text="node 12: raw_features=[1.0000]; embedding=[0.2000]",
        )

        self.assertIn("Target features:\n[1.0000, 2.0000]", prompt)
        self.assertIn("Candidates:\n12, 34", prompt)
        self.assertIn("node 12: raw_features=[1.0000]; embedding=[0.2000]", prompt)

    def test_chat_template_tensor_output_can_be_decoded_and_parsed(self):
        class TensorChatTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def apply_chat_template(self, *args, **kwargs):
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

            def decode(self, generated_ids, skip_special_tokens=True):
                return "The predicted class is 1"

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                next_token = llm_module.torch.tensor([[99]], dtype=llm_module.torch.long, device=input_ids.device)
                return llm_module.torch.cat([input_ids, next_token], dim=1)

        response = generate_response(FakeModel(), TensorChatTokenizer(), "prompt", "cpu")
        self.assertEqual(parse_prediction(response), 1)

    def test_chat_template_allows_thinking_by_default(self):
        class ThinkingTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def __init__(self):
                self.enable_thinking_values = []

            def apply_chat_template(self, *args, **kwargs):
                self.enable_thinking_values.append(kwargs.get("enable_thinking"))
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

            def decode(self, generated_ids, skip_special_tokens=True):
                return "The predicted class is 1"

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                next_token = llm_module.torch.tensor([[99]], dtype=llm_module.torch.long, device=input_ids.device)
                return llm_module.torch.cat([input_ids, next_token], dim=1)

        tokenizer = ThinkingTokenizer()
        response = generate_response(FakeModel(), tokenizer, "prompt", "cpu")

        self.assertEqual(parse_prediction(response), 1)
        self.assertEqual(tokenizer.enable_thinking_values[0], None)

    def test_chat_template_can_disable_thinking_when_requested(self):
        class ThinkingTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def __init__(self):
                self.enable_thinking_values = []

            def apply_chat_template(self, *args, **kwargs):
                self.enable_thinking_values.append(kwargs.get("enable_thinking"))
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

            def decode(self, generated_ids, skip_special_tokens=True):
                return "The predicted class is 1"

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                next_token = llm_module.torch.tensor([[99]], dtype=llm_module.torch.long, device=input_ids.device)
                return llm_module.torch.cat([input_ids, next_token], dim=1)

        tokenizer = ThinkingTokenizer()
        response = generate_response(FakeModel(), tokenizer, "prompt", "cpu", disable_thinking=True)

        self.assertEqual(parse_prediction(response), 1)
        self.assertEqual(tokenizer.enable_thinking_values[0], False)

    def test_chat_template_can_pass_thinking_budget(self):
        class ThinkingTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def __init__(self):
                self.kwargs_seen = []

            def apply_chat_template(self, *args, **kwargs):
                self.kwargs_seen.append(dict(kwargs))
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

            def decode(self, generated_ids, skip_special_tokens=True):
                return "The predicted class is 1"

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                next_token = llm_module.torch.tensor([[99]], dtype=llm_module.torch.long, device=input_ids.device)
                return llm_module.torch.cat([input_ids, next_token], dim=1)

        tokenizer = ThinkingTokenizer()
        response = generate_response(FakeModel(), tokenizer, "prompt", "cpu", thinking_budget=64)

        self.assertEqual(parse_prediction(response), 1)
        self.assertEqual(tokenizer.kwargs_seen[0].get("enable_thinking"), True)
        self.assertEqual(tokenizer.kwargs_seen[0].get("thinking_budget"), 64)

    def test_run_inference_can_return_raw_text_for_reconstruction(self):
        class TensorChatTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def apply_chat_template(self, *args, **kwargs):
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

            def decode(self, generated_ids, skip_special_tokens=True):
                return '{"selected_neighbors": [12, 34], "confidence": 0.8}'

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                next_token = llm_module.torch.tensor([[99]], dtype=llm_module.torch.long, device=input_ids.device)
                return llm_module.torch.cat([input_ids, next_token], dim=1)

        original_load_llm = llm_module.load_llm
        try:
            llm_module.load_llm = lambda model_name, device: (TensorChatTokenizer(), FakeModel())
            outputs = run_inference_all(["fake-model"], ["prompt"], "cpu", parse_predictions=False)
        finally:
            llm_module.load_llm = original_load_llm

        self.assertEqual(outputs["fake-model"][0], '{"selected_neighbors": [12, 34], "confidence": 0.8}')

    def test_run_inference_can_keep_raw_text_with_parsed_prediction(self):
        class TensorChatTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def apply_chat_template(self, *args, **kwargs):
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

            def decode(self, generated_ids, skip_special_tokens=True):
                return "I choose 0."

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                next_token = llm_module.torch.tensor([[99]], dtype=llm_module.torch.long, device=input_ids.device)
                return llm_module.torch.cat([input_ids, next_token], dim=1)

        original_load_llm = llm_module.load_llm
        try:
            llm_module.load_llm = lambda model_name, device: (TensorChatTokenizer(), FakeModel())
            outputs = run_inference_all(["fake-model"], ["prompt"], "cpu", return_raw=True)
        finally:
            llm_module.load_llm = original_load_llm

        self.assertEqual(outputs["fake-model"][0]["raw_response"], "I choose 0.")
        self.assertEqual(outputs["fake-model"][0]["parsed_prediction"], 0)

    def test_run_inference_can_log_errors_and_continue(self):
        class TensorChatTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def apply_chat_template(self, *args, **kwargs):
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

        class FailingModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                raise RuntimeError("synthetic failure")

        original_load_llm = llm_module.load_llm
        try:
            llm_module.load_llm = lambda model_name, device: (TensorChatTokenizer(), FailingModel())
            outputs = run_inference_all(
                ["fake-model"],
                ["prompt"],
                "cpu",
                parse_predictions=False,
                continue_on_error=True,
            )
        finally:
            llm_module.load_llm = original_load_llm

        self.assertEqual(outputs["fake-model"][0]["raw_response"], None)
        self.assertEqual(outputs["fake-model"][0]["error"]["type"], "RuntimeError")
        self.assertIn("synthetic failure", outputs["fake-model"][0]["error"]["message"])

    def test_run_inference_supports_batched_prompts(self):
        class BatchTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"
            padding_side = "right"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, return_tensors=None):
                if tokenize:
                    return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)
                return f"chat:{messages[0]['content']}"

            def __call__(self, texts, return_tensors="pt", padding=True):
                batch_size = len(texts)
                input_ids = llm_module.torch.arange(
                    10,
                    10 + batch_size * 3,
                    dtype=llm_module.torch.long,
                ).reshape(batch_size, 3)
                attention_mask = llm_module.torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            def decode(self, generated_ids, skip_special_tokens=True):
                token = int(generated_ids.reshape(-1)[0].item())
                return f"The predicted class is {token % 2}"

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                batch_size = input_ids.shape[0]
                next_tokens = llm_module.torch.arange(
                    101,
                    101 + batch_size,
                    dtype=llm_module.torch.long,
                    device=input_ids.device,
                ).reshape(batch_size, 1)
                return llm_module.torch.cat([input_ids, next_tokens], dim=1)

        original_load_llm = llm_module.load_llm
        try:
            llm_module.load_llm = lambda model_name, device: (BatchTokenizer(), FakeModel())
            outputs = run_inference_all(
                ["fake-model"],
                ["prompt-a", "prompt-b", "prompt-c"],
                "cpu",
                return_raw=True,
                llm_batch_size=2,
            )
        finally:
            llm_module.load_llm = original_load_llm

        self.assertEqual(len(outputs["fake-model"]), 3)
        self.assertEqual(outputs["fake-model"][0]["parsed_prediction"], 1)
        self.assertEqual(outputs["fake-model"][1]["parsed_prediction"], 0)
        self.assertEqual(outputs["fake-model"][2]["parsed_prediction"], 1)


if __name__ == "__main__":
    unittest.main()
