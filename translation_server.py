import json
import re
import logging
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from datetime import datetime

# --- Logging setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler('translation.log', maxBytes=1000000, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
except Exception as e:
    fallback_handler = logging.FileHandler('translation.log')
    fallback_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fallback_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

app = Flask(__name__)

class FrenchTranslator:
    def __init__(self):
        logger.info("Initializing FrenchTranslator")
        try:
            model_name = "facebook/nllb-200-distilled-600M"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.translator = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                src_lang="eng_Latn",
                tgt_lang="fra_Latn",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(" NLLB-200 model loaded successfully.")
        except Exception as e:
            logger.error(f" Failed to load NLLB-200 model: {e}")
            raise

        self.json_files = {
            'actions': 'actions.json',
            'components': 'components.json',
            'ticket_types': 'ticket_types.json'
        }

        self._load_json_files()
        self._build_term_maps()

    def _load_json_files(self):
        try:
            with open(self.json_files['actions'], encoding='utf-8') as f:
                self.actions = json.load(f)
            with open(self.json_files['components'], encoding='utf-8') as f:
                self.components = json.load(f)
            with open(self.json_files['ticket_types'], encoding='utf-8') as f:
                self.ticket_types = json.load(f)
            logger.info(" JSON files loaded successfully.")
        except Exception as e:
            logger.error(f" JSON loading error: {e}")
            raise

    def _build_term_maps(self):
        self.term_maps = {'actions': {}, 'components': {}, 'types': {}}
        for dico, data in [('actions', self.actions), ('components', self.components), ('types', self.ticket_types)]:
            for key, entry in data.items():
                if not isinstance(entry, dict) or 'en' not in entry or 'fr' not in entry:
                    continue
                en_term = entry['en'].strip().lower()
                fr_term = re.sub(r'[@#*]+$', '', entry['fr'].strip())
                if en_term and fr_term:
                    self.term_maps[dico][en_term] = fr_term
        logger.info(" Term maps built.")

    def _replace_technical_terms_with_placeholders(self, text: str):
        all_terms = {
            **self.term_maps['actions'],
            **self.term_maps['components'],
            **self.term_maps['types']
        }

        text = re.sub(r'\bATM\b', 'DAB', text, flags=re.IGNORECASE)
        sorted_terms = sorted(all_terms.keys(), key=lambda x: -len(x))

        placeholder_map = {}
        placeholder_counter = 0

        def replacer(match):
            nonlocal placeholder_counter
            term = match.group(0).lower()
            fr_term = all_terms.get(term)
            if not fr_term:
                return match.group(0)
            placeholder = f"@@TERM{placeholder_counter}@@"
            placeholder_map[placeholder] = fr_term
            placeholder_counter += 1
            return placeholder

        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_terms)) + r')\b', flags=re.IGNORECASE)
        replaced_text = pattern.sub(replacer, text)
        logger.debug(f"Text with placeholders: {replaced_text}")
        logger.debug(f"Placeholder map: {placeholder_map}")

        return replaced_text, placeholder_map

    def translate_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            logger.warning("Empty text for translation.")
            return "Description du problème indisponible."

        preprocessed, placeholder_map = self._replace_technical_terms_with_placeholders(text)
        logger.info(f"Text before translation: {preprocessed}")

        try:
            # Split into lines to preserve structure
            lines = preprocessed.split('\n')
            translated_lines = []
            has_intervention_header = False
            problem_count = 0

            # Detect structure
            for line in lines:
                if "our technical team identified the following problem" in line.lower():
                    problem_count += 1
                if "intervention report:" in line.lower():
                    has_intervention_header = True

            # Append default intervention based on problem count
            if has_intervention_header and not any(line.strip().startswith(str(i) + ". ") for i in range(1, problem_count + 1) for line in lines):
                logger.info("Debug: Appending default intervention due to incomplete structure")
                if problem_count > 1:
                    default_intervention = f"\n1. Le composant affecté a été remplacé"  # Enumerated for multiple problems
                else:
                    default_intervention = "\nLe composant affecté a été remplacé"  # Non-enumerated for single problem
                lines.append(default_intervention)

            # Translate each line
            for line in lines:
                if line.strip():
                    match = re.match(r'^(\d+\.\s+)', line)
                    if match:
                        number_part = match.group(0)
                        text_part = line[len(number_part):].strip()
                        translated_text_part = self.translator(text_part, max_length=512)[0]['translation_text']
                        translated_lines.append(number_part + translated_text_part)
                    else:
                        translated_lines.append(self.translator(line.strip(), max_length=512)[0]['translation_text'])
                else:
                    translated_lines.append("")  # Preserve empty lines
            result = '\n'.join(translated_lines)

            logger.info(f"Raw translation: {result}")

            for placeholder, fr_term in placeholder_map.items():
                clean_fr_term = re.sub(r'[@#*]+$', '', fr_term).strip()
                result = result.replace(placeholder, clean_fr_term)

            result = re.sub(r'[@#*]+', '', result)
            result = re.sub(r'\.(?=\S)', ' ', result)
            result = re.sub(r'\.+', '.', result)
            result = re.sub(r'[.,;!?]+(?=[.,;!?])', '', result)

            result = re.sub(r'\b(?:le|la|Le|La)\s+Application\s+DAB\b', r"l'Application DAB", result, flags=re.IGNORECASE)
            result = re.sub(r'\bDAB\s+de\s+l\'application\b', r"Application DAB", result, flags=re.IGNORECASE)
            result = re.sub(r'\bL\'équipe\b', r"l'équipe", result, flags=re.IGNORECASE)

            if not result.endswith(('.', '!', '?')):
                result += '.'

            logger.info(f"Final translated text: '{result}'")
            return result
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return "Description du problème indisponible."

# Initialize translator with current time
translator = None
try:
    translator = FrenchTranslator()
except Exception as e:
    logger.error(f"Translator initialization error: {e}")

@app.route('/translate', methods=['POST'])
def translate():
    if translator is None:
        return jsonify({"error": "Translator not initialized"}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data['text']
    try:
        translated_text = translator.translate_text(text)
        return jsonify({"translated_text": translated_text}), 200
    except Exception as e:
        logger.error(f" Translation error: {e}")
        return jsonify({"error": "Translation failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)