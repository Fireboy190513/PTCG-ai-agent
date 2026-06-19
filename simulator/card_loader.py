# simulator/card_loader.py
import csv
import json
import os
from typing import Dict, Any

CARD_DB_PATH = "data/cards.json"
CARD_CSV_PATH = "data/EN_Card_Data.csv"

def load_cards() -> Dict[int, Dict[str, Any]]:
    """
    Load card database from JSON if available, else generate from CSV.
    Returns dict mapping card_id -> card_definition
    """
    if os.path.exists(CARD_DB_PATH):
        with open(CARD_DB_PATH, 'r') as f:
            return json.load(f)
    
    # Fallback: generate minimal card database
    return _generate_minimal_db()

def _generate_minimal_db() -> Dict[int, Dict[str, Any]]:
    """Generate a minimal card database for testing"""
    cards = {}
    basic_pokemon = ["Greninja ex", "Magcargo ex", "Incineroar ex", "Pikachu", "Charizard"]
    energies = ["Fire Energy", "Water Energy", "Grass Energy", "Lightning Energy", "Psychic Energy"]
    trainers = ["Potion", "Rare Candy", "Quick Ball", "Max Potion", "Switch"]
    
    card_id = 1
    
    # Add basic pokemon
    for name in basic_pokemon:
        cards[card_id] = {
            "id": card_id,
            "name": name,
            "category": "Pokémon-Basic",
            "hp": 120,
            "retreat": 1,
            "moves": [
                {"name": "Tackle", "damage": 30},
                {"name": "Sonic Boom", "damage": 60}
            ],
            "Type": "{W}"
        }
        card_id += 1
    
    # Add energies
    for energy_type, energy_symbol in [("Fire", "{R}"), ("Water", "{W}"), ("Grass", "{G}"), ("Lightning", "{L}"), ("Psychic", "{P}")]:
        cards[card_id] = {
            "id": card_id,
            "name": f"{energy_type} Energy",
            "category": "Energy",
            "Type": energy_symbol
        }
        card_id += 1
    
    # Add trainers
    for trainer in trainers:
        cards[card_id] = {
            "id": card_id,
            "name": trainer,
            "category": "Trainer",
        }
        card_id += 1
    
    return cards

def generate_card_db_from_csv():
    """Parse EN_Card_Data.csv and generate cards.json"""
    if not os.path.exists(CARD_CSV_PATH):
        print(f"Warning: {CARD_CSV_PATH} not found. Using minimal database.")
        return
    
    cards = {}
    card_id = 1
    
    try:
        with open(CARD_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("Name"):
                    continue
                cards[card_id] = {
                    "id": card_id,
                    "name": row.get("Name", "Unknown"),
                    "category": row.get("Type", ""),
                    "hp": int(row.get("HP", 1)) if row.get("HP", "").isdigit() else 1,
                    "retreat": int(row.get("Retreat Cost", 0)) if row.get("Retreat Cost", "").isdigit() else 0,
                    "Type": row.get("Type", ""),
                    "moves": []
                }
                card_id += 1
        
        os.makedirs("data", exist_ok=True)
        with open(CARD_DB_PATH, 'w') as f:
            json.dump(cards, f, indent=2)
        print(f"Generated {card_id - 1} cards in {CARD_DB_PATH}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
