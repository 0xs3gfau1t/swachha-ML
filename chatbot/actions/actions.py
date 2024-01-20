from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import nltk
from nltk.stem import WordNetLemmatizer

# Download the WordNet resource
nltk.download('wordnet')

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# Define a word to lemmatize
word = "running"


RESP = {
	"paper":"Recycling paper helps conserve trees and reduces energy consumption. Collect newspapers, cardboard, and office paper. Remove contaminants like plastic or metal, and drop them in your recycling bin. Local recycling centers often accept paper, transforming it into new products.",
	
	"glass":"Glass recycling saves energy and reduces landfill waste. Collect glass bottles and jars, ensuring they are clean. Remove caps and lids, and place them in your recycling bin. Recycling centers melt glass to create new containers, minimizing environmental impact."
	}

class RecycleOrganic(Action):
	def name(self) -> Text:
		return "action_recycle_waste"

	def run(self, dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
		try:
			waste_type = tracker.get_latest_entity_values("waste_type", None)
			token = lemmatizer.lemmatize(next(waste_type))	
			res = RESP[token]
		except:
			res = "Sorrryyyyyy! I am unaware about this type of waste."
		dispatcher.utter_message(text=res)

		return []
