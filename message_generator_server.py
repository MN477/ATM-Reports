from flask import Flask, request, jsonify
from llama_cpp import Llama
import json
from threading import Lock
import logging
import sys
import re
from logging import StreamHandler, FileHandler, Formatter
from data_processor import TermClassifier
import random
import os

# --- Logging setup ---
logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel(logging.INFO)

console_handler = StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(Formatter('%(asctime)s - %(levelname)s - %(message)s'))

file_handler = FileHandler('classification.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- Model Configuration ---
local_model_path = "C:/Users/mouss/Desktop/MistralAI_LLM_TSS_2/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
n_ctx = 2048
n_gpu_layers = 0

class ModelManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        try:
            self.model = Llama(
                model_path=local_model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers
            )
            logger.info(" Model loaded successfully")
        except Exception as e:
            logger.error(f" Error loading model: {e}")
            raise

class TechnicalMessageGenerator:
    def __init__(self):
        self.model_manager = ModelManager()
        self.model = self.model_manager.model
        self._load_translations()
        self.component_pattern = re.compile(
            r'\b(sensor|unit|module|device|component|presenter|dispenser|mechanism|assembly|interface)\b',
            flags=re.IGNORECASE
        )

    def _load_translations(self):
        try:
            with open("ticket_types.json", "r", encoding='utf-8') as f:
                self.ticket_types = json.load(f)
            with open("components.json", "r", encoding='utf-8') as f:
                raw_components = json.load(f)
                self.tech_abbreviations = {
                    k: v for k, v in raw_components.items()
                    if v.get('en') and v['en'].lower() != "nan"
                }
            with open("actions.json", "r", encoding='utf-8') as f:
                self.action_keywords = json.load(f)
        except Exception as e:
            logger.error(f" Error loading translation files: {e}")
            self.ticket_types = {}
            self.tech_abbreviations = {}
            self.action_keywords = {}

    def translate_component(self, component: str) -> str:
        original = component.strip()
        normalized = original.lower().replace("_", " ").strip()

        if original in self.tech_abbreviations:
            return self.tech_abbreviations[original].get('en', original)

        for key, value in self.tech_abbreviations.items():
            key_normalized = key.lower().replace("_", " ").strip()
            if key_normalized == normalized:
                return value.get('en', original)
            if normalized in key_normalized:
                return value.get('en', original)

        return original.replace("_", " ")

    def translate_action(self, action: str) -> str:
        if action in self.action_keywords:
            return self.action_keywords[action].get('en', action)
        return action.replace("_", " ").title()

    def enhance_problem_description(self, description: str) -> str:
        prompt = (
            f"Rephrase the following technical problem description for an ATM system, while maintaining:\n"
            f"- The exact component names\n"
            f"- The original hierarchical relationships between components\n"
            f"- Technical accuracy\n\n"
            f"Improvements to make:\n"
            f"1. Replace 'which is within' with more technical relationship terms\n"
            f"2. Add a brief functional context for the affected component\n"
            f"3. Keep the description concise (1-2 sentences)\n"
            f"4. Maintain the original meaning exactly\n\n"
            f"Original description: \"{description}\"\n\n"
            f"Enhanced technical description:"
        )
        
        try:
            output = self.model(
                prompt,
                max_tokens=100,
                temperature=0.3,
                top_p=0.9,
                stop=["Enhanced technical description:"]
            )
            generated_text = output['choices'][0]['text'].strip()
            if "Enhanced technical description:" in generated_text:
                enhanced = generated_text.split("Enhanced technical description:", 1)[-1].strip()
            else:
                enhanced = description
            enhanced = re.sub(r'[\[\]"\']', '', enhanced)
            enhanced = enhanced.split('\n')[0].strip()
            return enhanced
        except Exception as e:
            logger.warning(f" Failed LLM enhancement: {e}")
            return description

    def generate_intervention_phrase(self, action: str) -> str:
        translated_action = self.translate_action(action)
        action_verb = self._action_to_verb(translated_action)
        
        prompt = (
            f"Generate a concise, professional technical intervention report for an ATM repair.\n"
            f"Action taken: {action_verb}\n"
            f"Rules:\n"
            f"1. Use formal language\n"
            f"2. Be concise (one sentence)\n"
            f"3. Focus ONLY on the solution\n"
            f"4. NEVER mention specific components - use ONLY 'the affected component'\n"
            f"5. Avoid mentioning the issue or problem\n"
            f"6. Do NOT use technical component names\n"
            f"7. Example format: 'The affected component was {action_verb}.'\n"
            f"Intervention report:"
        )
        
        try:
            output = self.model(
                prompt,
                max_tokens=40,
                temperature=0.1,
                top_p=0.9,
                stop=["Intervention report:"]
            )
            generated_text = output['choices'][0]['text'].strip()
            report = generated_text.split("Intervention report:", 1)[-1].strip() if "Intervention report:" in generated_text else generated_text
            return self._clean_and_validate_report(report, action_verb)
        except Exception as e:
            logger.warning(f" Failed LLM intervention generation: {e}")
            return f"The affected component was {action_verb}."

    def _action_to_verb(self, action_noun: str) -> str:
        verb_mapping = {
            "Replacement": "replaced",
            "Cleaning": "cleaned",
            "Adjustment": "adjusted",
            "Configuration": "configured",
            "Update": "updated",
            "Calibration": "calibrated",
            "Inspection": "inspected",
            "Repair": "repaired"
        }
        return verb_mapping.get(action_noun, action_noun.lower())

    def _clean_and_validate_report(self, report: str, action_verb: str) -> str:
        report = re.sub(r'[\[\]"\']', '', report)
        report = re.sub(r'\bATM\b', '', report, flags=re.IGNORECASE)
        report = self.component_pattern.sub('component', report)
        report = report.split('.')[0].strip()

        if not report or len(report.split()) < 4 or "component" not in report.lower():
            return f"The affected component was {action_verb}."

        if "affected component" not in report:
            report = re.sub(r'(the|a|an) \w+ component', 'the affected component', report, flags=re.IGNORECASE)

        return report

    def build_hierarchical_description(self, issue_tokens):
        if not issue_tokens:
            return "unknown issue"

        ticket_type_str = ""
        for i in range(len(issue_tokens)):
            candidate = "_".join(issue_tokens[:i+1])
            if candidate in self.ticket_types:
                ticket_type_str = candidate
                break

        if not ticket_type_str:
            ticket_type_str = issue_tokens[0]

        nature = self.ticket_types.get(ticket_type_str, {}).get('en', ticket_type_str.replace("_", " ").title())
        components = issue_tokens[len(ticket_type_str.split('_')):]
        translated_components = [self.translate_component(comp) for comp in components]

        if not translated_components:
            return f"A {nature}"

        if len(translated_components) == 1:
            base_description = f"A {nature} affecting the {translated_components[0]}"
        else:
            reversed_components = list(reversed(translated_components))
            base_description = f"A {nature} affecting the {reversed_components[0]}"
            for comp in reversed_components[1:]:
                base_description += f", which is within the {comp}"

        return base_description

    def generate_technical_phrase(self, technical_input: str) -> tuple:
        tokens = technical_input.strip().split("_")
        last_token = tokens[-1]

        if last_token in self.action_keywords:
            action = last_token
            issue_tokens = tokens[:-1]
        else:
            action = None
            issue_tokens = tokens

        problem_description = self.build_hierarchical_description(issue_tokens)
        return problem_description, action

    def generate_client_message(self, technical_input: str) -> dict:
        parts = [p.strip() for p in technical_input.split("/") if p.strip()]
        problem_actions = [self.generate_technical_phrase(part) for part in parts]

        is_closing_ticket = any(action for (_, action) in problem_actions)

        if is_closing_ticket:
            problem_descriptions = [desc for (desc, _) in problem_actions]
            actions = [action for (_, action) in problem_actions]
            intervention_phrases = [
                self.generate_intervention_phrase(action)
                for action in actions
            ]

            if len(problem_descriptions) == 1:
                message = (
                    f"Dear Customer,\n"
                    f"Following the completion of the required intervention, we confirm that the ATM has been returned to service and is now operational.\n"
                    f"Our technical team identified the following problem: {problem_descriptions[0]}\n"
                    f"Please find below our intervention report: {intervention_phrases[0]}"
                )
            else:
                problems_list = "\n".join([f"{idx+1}. {desc}" for idx, desc in enumerate(problem_descriptions)])
                interventions_list = "\n".join([f"{idx+1}. {phrase}" for idx, phrase in enumerate(intervention_phrases)])

                message = (
                    f"Dear Customer,\n"
                    f"Following the completion of the required intervention, we confirm that the ATM has been returned to service and is now operational.\n"
                    f"Our technical team identified the following problems:\n"
                    f"{problems_list}\n"
                    f"Please find below our intervention report:\n"
                    f"{interventions_list}"
                )
        else:
            problem_descriptions = [desc for (desc, _) in problem_actions]

            if len(problem_descriptions) == 1:
                message = (
                    f"Dear Customer,\n"
                    f"We have received your request regarding the following problem:\n"
                    f"{problem_descriptions[0]}.\n"
                    f"Our technical team will review the request and take the necessary actions to resolve it."
                )
            else:
                problems_list = "\n".join([f"{idx+1}. {desc}" for idx, desc in enumerate(problem_descriptions)])
                message = (
                    f"Dear Customer,\n"
                    f"We have received your request regarding the following problems:\n"
                    f"{problems_list}\n"
                    f"Our technical team will review the request and take the necessary actions to resolve them."
                )

        return {"english_report": message}

# --- Flask App ---
app = Flask(__name__)
generator = TechnicalMessageGenerator()

@app.route('/generate-message', methods=['POST'])
def generate_message():
    data = request.get_json()
    technical_input = data.get("technical_input", "")

    if not technical_input:
        return jsonify({"error": "Missing technical_input"}), 400

    try:
        result = generator.generate_client_message(technical_input)
        logger.info(f"Generated result: {result}")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f" Error in /generate-message: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    classifier = TermClassifier("Abreviation_et_Description_V1.xlsx")
    if classifier.run():
        logger.info(" Classification terminée avec succès")
    else:
        logger.error(" Erreur pendant la classification")
    app.run(port=5000, threaded=True)
