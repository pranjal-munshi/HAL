import re
from collections import Counter
import spacy
from nltk.corpus import stopwords

class SubtopicExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))

    def method1_regex_patterns(self, text):
        subtopics = []
        caps_pattern = r'\b[A-Z][A-Z\s]{2,}\b'
        subtopics.extend(re.findall(caps_pattern, text))
        bold_pattern = r'\*\*(.*?)\*\*'
        subtopics.extend(re.findall(bold_pattern, text))
        heading_pattern = r'^\d+\.?\s*([A-Z][A-Z\s]+)$'
        subtopics.extend(re.findall(heading_pattern, text, re.MULTILINE))
        line_pattern = r'^([A-Z][A-Z\s]+)[.:]?\s*$'
        subtopics.extend(re.findall(line_pattern, text, re.MULTILINE))
        return list(set(subtopics))

    def extract_all_methods(self, text):
        all_subtopics = self.method1_regex_patterns(text)
        topic_counts = Counter(all_subtopics)
        final_topics = []
        for topic, count in topic_counts.items():
            if len(topic) > 3 and topic.lower() not in self.stop_words and len(topic.split()) <= 5:
                final_topics.append((topic.strip(), count))
        return final_topics
