import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import sys

logger = logging.getLogger(__name__)
def configure_logging():
    # Fonction à appeler pour configurer le logging (console + fichier)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('classification.log', encoding='utf-8')
        ]
    )
class TermClassifier:
    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.actions_path = Path('actions.json')
        self.ticket_types_path = Path('ticket_types.json')
        self.components_path = Path('components.json')
        self.model = None
        self.tokenizer = None

    def load_existing_classifications(self) -> Dict[str, str]:
        classifications = {}
        category_files = [
            (self.actions_path, 'action'),
            (self.ticket_types_path, 'ticket_type'),
            (self.components_path, 'component')
        ]
        for path, category in category_files:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for term in data:
                        classifications[term.upper()] = category
        return classifications

    def get_new_terms(self, df: pd.DataFrame, existing: Dict[str, str]) -> pd.DataFrame:
        excel_terms = set(df['abbr'].str.strip().str.upper())
        existing_terms = set(existing.keys())
        new_terms = excel_terms - existing_terms
        return df[df['abbr'].str.strip().str.upper().isin(new_terms)].copy()

    def load_model(self):
        if self.model is None:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-llm-7b-chat",
                quantization_config=quant_config,
                device_map={"": 0}  # éviter "auto" qui tente de tout charger sur le GPU
            )
            logger.info("Modèle chargé avec config adaptée")

    def generate_prompt(self, term: str, en: str, fr: str) -> str:
        return f"""Classify this ATM technical term into exactly one of these categories: 
'action', 'ticket_type', or 'component':

TERM: "{term}"
ENGLISH: "{en}"
FRENCH: "{fr}"

CATEGORY DEFINITIONS:
1. 'action': Repair processes, system operations, approval processes
2. 'ticket_type': Types of problems or maintenance categories
3. 'component': Physical or logical parts of the ATM

EXAMPLES:
- "Repair" -> action
- "HW ISSUE" -> ticket_type
- "CDM" -> component
- "Reset" -> action
- "PREVENTIVE MAINTENANCE" -> ticket_type
- "Feeder" -> component

ANALYSIS: Consider the term's function and context in both languages.

RESPOND ONLY WITH THE CATEGORY NAME: 'action', 'ticket_type', or 'component'"""

    def classify_term(self, term: str, en: str, fr: str) -> str:
        self.load_model()
        prompt = self.generate_prompt(term, en, fr)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=15,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        ).strip().lower()

        if 'action' in response:
            return 'action'
        elif 'ticket_type' in response or 'ticket' in response:
            return 'ticket_type'
        elif 'component' in response:
            return 'component'

        # fallback heuristique
        text = f"{term} {en} {fr}".lower()
        if any(kw in text for kw in {'repair', 'replace', 'reset', 'upgrade', 'install', 'approval', 'completed', 'reported', 'client issue'}):
            return 'action'
        elif any(kw in text for kw in {'issue', 'fault', 'error', 'vandalism', 'misuse', 'non conform', 'maintenance', 'conformity', 'service'}):
            return 'ticket_type'
        return 'component'

    def process_new_terms(self, new_terms: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
        actions, ticket_types, components = {}, {}, {}
        for _, row in new_terms.iterrows():
            term = str(row['abbr']).strip()
            en = str(row['en']).strip()
            fr = str(row['fr']).strip()
            if not term or pd.isna(en) or pd.isna(fr):
                continue
            try:
                classification = self.classify_term(term, en, fr)
                result = {'en': en, 'fr': fr}
                if classification == 'action':
                    actions[term] = result
                    logger.info(f"CLASSIFIED: {term} -> action")
                elif classification == 'ticket_type':
                    ticket_types[term] = result
                    logger.info(f"CLASSIFIED: {term} -> ticket_type")
                else:
                    components[term] = result
                    logger.info(f"CLASSIFIED: {term} -> component")
            except Exception as e:
                logger.error(f"Error classifying {term}: {e}")
                components[term] = {'en': en, 'fr': fr}
        return actions, ticket_types, components

    def merge_results(self, new_actions: Dict, new_ticket_types: Dict, new_components: Dict) -> Tuple[Dict, Dict, Dict]:
        final_actions = self.load_json(self.actions_path)
        final_ticket_types = self.load_json(self.ticket_types_path)
        final_components = self.load_json(self.components_path)

        final_actions.update(new_actions)
        final_ticket_types.update(new_ticket_types)
        final_components.update(new_components)
        return final_actions, final_ticket_types, final_components

    def load_json(self, path: Path) -> Dict:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_results(self, actions: Dict, ticket_types: Dict, components: Dict):
        with open(self.actions_path, 'w', encoding='utf-8') as f:
            json.dump(actions, f, ensure_ascii=False, indent=2)
        with open(self.ticket_types_path, 'w', encoding='utf-8') as f:
            json.dump(ticket_types, f, ensure_ascii=False, indent=2)
        with open(self.components_path, 'w', encoding='utf-8') as f:
            json.dump(components, f, ensure_ascii=False, indent=2)

    def run(self):
        try:
            df = pd.read_excel(self.excel_path, sheet_name='Feuil1')
            df = df.rename(columns={
                'Abréviation': 'abbr',
                'Nom complet anglais': 'en', 
                'Nom en français': 'fr'
            }).dropna(subset=['abbr'])
            existing = self.load_existing_classifications()
            new_terms_df = self.get_new_terms(df, existing)

            if not new_terms_df.empty:
                logger.info(f"{len(new_terms_df)} nouveaux termes à classifier")
                new_actions, new_ticket_types, new_components = self.process_new_terms(new_terms_df)
                final_actions, final_ticket_types, final_components = self.merge_results(
                    new_actions, new_ticket_types, new_components
                )
                self.save_results(final_actions, final_ticket_types, final_components)
                logger.info("Classification à 3 catégories terminée avec succès")
            else:
                logger.info("Aucun nouveau terme à classifier")
            return True
        except Exception as e:
            logger.exception(f"Erreur du processus: {e}")
            return False

if __name__ == "__main__":
    configure_logging()  # configure logging uniquement si on exécute ce fichier directement
    classifier = TermClassifier("Abreviation_et_Description_V1.xlsx")
    success = classifier.run()
    sys.exit(0 if success else 1)
