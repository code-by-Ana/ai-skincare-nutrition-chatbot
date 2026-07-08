import streamlit as st
import json
import re
import nltk
import logging
import random
import time
import pickle
import os
import requests
import pandas as pd
from html import escape
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv(dotenv_path=r"E:\My python\Skin & Nutrition\API_KEY.env")  



logging.basicConfig(filename='healthbot.log', level=logging.ERROR)
st.set_page_config(page_title="HealthBot Pro", layout="wide")


class Config:
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') # NEW
    DATA_PATHS = {
        'skincare': Path(r"E:\My python\Final Project\skincare.json"),
        'nutrition': Path(r"E:\My python\Final Project\nutri.json"),
        'questions': Path(r"E:\My python\Final Project\questions.txt"),
        'cosmetics': Path(r"E:\My python\Skin & Nutrition\cosmetics.csv")
    }
    MODEL_PATH = Path(r"E:\My python\Final Project\intent_classifier.pkl")
    STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
    CONTEXT_TIMEOUT = 300
    QA_THRESHOLD = 75


class SkincareNutritionBot:
    def __init__(self):
        try:
            if not Config.DEEPSEEK_API_KEY:
                raise ValueError("DeepSeek API key not found in environment variables")

            self.skincare_data = self.load_knowledge(Config.DATA_PATHS['skincare'])
            self.nutrition_data = self.load_knowledge(Config.DATA_PATHS['nutrition'])
            self.qa_pairs = self.load_qa_pairs()
            self.lemmatizer = WordNetLemmatizer()
            self.vectorizer = TfidfVectorizer()
            self.user_context = {}
            self.clf = self.train_or_load_intent_classifier()
        except Exception as e:
            logging.error(f"Failed to initialize bot: {str(e)}")
            raise

    def ask_deepseek(self, prompt):
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}"
            }
            payload = {
                "model": "deepseek/deepseek-chat:free",
                "messages": [{
                    "role": "user",
                    "content": f"""Provide a comprehensive answer to: {prompt}
                    Include:
                    1. Scientific explanation
                    2. Practical usage guidelines
                    3. Safety considerations
                    Keep response between 400-600 words. No open-ended questions."""
                }],
                "temperature": 0.7,
                "max_tokens": 1200, 
                "stop": ["\n\n4.", "### Next Section"]  
            }

            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()

            raw_response = response.json()['choices'][0]['message']['content'].strip()

            # Improved formatting
            cleaned_response = re.sub(r'\n{3,}', '\n\n', raw_response)
            cleaned_response = re.sub(r'(\w)(\n)(\w)', r'\1 \3', cleaned_response)


            return f" Expert Answer:\n{cleaned_response}" 


            def ensure_complete_ending(text):
                last_punctuation = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                return text[:last_punctuation+1] if last_punctuation != -1 else text
        
            cleaned_response = ensure_complete_ending(cleaned_response)

            return f" Expert Answer:\n{cleaned_response}"

        except requests.Timeout:
            return "⏳ Expert advice delayed - please try again in a moment!"
        except requests.RequestException as e:
            logging.error(f"API Connection Error: {str(e)}")
            return "🔌 Couldn't connect to expert knowledge - try a skincare/nutrition question instead!"
        except Exception as e:
            logging.error(f"API Processing Error: {str(e)}")
        return "⚠️ Temporary expert knowledge gap - try rephrasing your question!"


    def match_question(self, user_input):
        try:
            best_match = None
            highest_score = 0
        
            # Preprocess user input
            processed_input = self.preprocess_text(user_input)
        
            for question, answer in self.qa_pairs:
                # Compare with preprocessed questions
                score = fuzz.token_set_ratio(processed_input, self.preprocess_text(question))
                if score > highest_score and score >= Config.QA_THRESHOLD:
                    highest_score = score
                    best_match = answer
                
            return best_match
        except Exception as e:
            logging.error(f"Failed to match question: {str(e)}")
            return None

    
    def is_valid_response(self, response):
        
        empty_indicators = [
            "please specify", "not available", "no data", 
            "please tell", "context cleared", "start fresh"
        ]
        return (
            response.strip() and 
            not any(indicator in response.lower()  
                    for indicator in empty_indicators)
        )

    def load_qa_pairs(self):
    
        try:
            qa_path = Config.DATA_PATHS['questions']
            if not qa_path.exists():
                raise FileNotFoundError(f"Q&A file not found: {qa_path}")
                
            with open(qa_path, 'r', encoding='utf-8') as f:
                return [tuple(line.strip().split('|')) for line in f if '|' in line]
        except Exception as e:
            logging.error(f"Failed to load Q&A pairs: {str(e)}")
            return []

    def load_knowledge(self, path):
        try:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
                
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not data or not isinstance(data, dict):
                raise ValueError(f"Invalid data format in {path.name}")
                
            return data
        except Exception as e:
            logging.error(f"Failed to load {path}: {str(e)}")
            return {}

    def preprocess_text(self, text):
        try:
            text = re.sub(r'[^\w\s]', '', text)
            tokens = nltk.word_tokenize(text.lower())
            filtered = [self.lemmatizer.lemmatize(w) for w in tokens 
                       if w not in Config.STOP_WORDS and len(w) > 2]
            return ' '.join(filtered)
        except Exception as e:
            logging.error(f"Failed to preprocess text: {str(e)}")
            return ""

    def train_or_load_intent_classifier(self):
        try:
            if Config.MODEL_PATH.exists():
                with open(Config.MODEL_PATH, 'rb') as f:
                    clf, vectorizer = pickle.load(f)
                    self.vectorizer = vectorizer
                    return clf
            else:
                training_data = [
                    ("skin", self.generate_skin_examples()),
                    ("nutrition", self.generate_nutrition_examples()),
                    ("qa", [q for q, a in self.qa_pairs])
                ]
                
                texts, labels = [], []
                for label, examples in training_data:
                    texts.extend(examples)
                    labels.extend([label] * len(examples))
                
                processed = [self.preprocess_text(t) for t in texts]
                X = self.vectorizer.fit_transform(processed)
                clf = MultinomialNB()
                clf.fit(X, labels)
                
                Config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(Config.MODEL_PATH, 'wb') as f:
                    pickle.dump((clf, self.vectorizer), f)
                    
                return clf
        except Exception as e:
            logging.error(f"Failed to train or load intent classifier: {str(e)}")
            return None

    def generate_skin_examples(self):
        try:
            examples = []
            skin_types = self.skincare_data.get("skin_types", {}).keys()
            concerns = self.skincare_data.get("common_concerns", {}).keys()
            
            print(f"Generating skin examples from: {list(skin_types)} skin types")
            print(f"Available concerns: {list(concerns)}")
            
            for st in skin_types:
                examples.extend([f"my skin is {st}", f"{st} skin routine",
                                 f"best products for {st} skin", f"I have {st} complexion"])
            for concern in concerns:
                examples.extend([f"help with {concern}", f"treating {concern}",
                                 f"products for {concern}", f"dealing with {concern}"])
            return examples
        except Exception as e:
            logging.error(f"Failed to generate skin examples: {str(e)}")
            return []

    def generate_nutrition_examples(self):
        try:
            examples = []
            categories = self.nutrition_data.get("categories", {}).keys()
            print(f"Generating nutrition examples from categories: {list(categories)}")
            
            for cat in categories:
                clean_cat = cat.replace('_', ' ')
                examples.extend([f"foods for {clean_cat}", f"{clean_cat} diet",
                                 f"healthy {clean_cat} options", f"nutrition tips for {clean_cat}"])
            return examples
        except Exception as e:
            logging.error(f"Failed to generate nutrition examples: {str(e)}")
            return []

    def detect_entities(self, text):
        try:
            entities = {'skin_type': None, 'nutrition_category': None, 'concern': None}
            processed_text = self.preprocess_text(text)
            print(f"\nEntity Detection Debug for: '{text}'")
            print(f"Processed text: {processed_text}")
            
            # Skin type detection debug
            print("\nSkin Type Matching:")
            for skin_type in self.skincare_data.get("skin_types", {}):
                score = fuzz.partial_ratio(skin_type, processed_text)
                print(f"- '{skin_type}': {score}")
                if score > 90:
                    entities['skin_type'] = skin_type
                    self.update_context('skin_type', skin_type)
            
            print("\nNutrition Category Matching:")
            for category in self.nutrition_data.get("categories", {}):
                clean_cat = category.replace('_', ' ')
                score = fuzz.partial_ratio(clean_cat, processed_text)
                print(f"- '{clean_cat}': {score}")
                if score > 85:
                    entities['nutrition_category'] = category
                    self.update_context('nutrition_category', category)
            
            print("\nConcern Matching:")
            for concern in self.skincare_data.get("common_concerns", {}):
                score = fuzz.partial_ratio(concern, processed_text)
                print(f"- '{concern}': {score}")
                if score > 90:
                    entities['concern'] = concern
                    self.update_context('concern', concern)
                    
            print(f"Final entities detected: {entities}")
            return entities
        except Exception as e:
            logging.error(f"Failed to detect entities: {str(e)}")
            return {}

    def update_context(self, key, value):
        try:
            self.user_context[key] = {
                'value': value,
                'timestamp': time.time()
            }
            print(f"Updated context: {key} = {value}")
        except Exception as e:
            logging.error(f"Failed to update context: {str(e)}")
        
    def get_context(self, key):
        try:
            entry = self.user_context.get(key)
            if entry:
                # Automatically clear expired context
                if (time.time() - entry['timestamp']) < Config.CONTEXT_TIMEOUT:
                    return entry['value']
                else:
                    del self.user_context[key]
                    print(f"Auto-cleared expired context: {key}")
                    return None
            return None
        except Exception as e:
            logging.error(f"Failed to get context: {str(e)}")
            return None  
    
    def get_context_string(self):

        context = []
        for key in ['skin_type', 'concern', 'nutrition_category']:
            value = self.get_context(key)
            if value:
                context.append(f"{key.replace('_', ' ')}: {value}")
        return "Context: " + ", ".join(context) if context else ""


    def generate_skin_response(self, entities):
        try:
            print("\nGenerating Skin Response:")
            print(f"Entities: {entities}")
            print(f"Current context: {self.user_context}")
            
            if not self.skincare_data.get("skin_types"):
                return "⚠️ Skincare data not available. Please try again later."
                
            response = []
            skin_type = entities.get('skin_type') or self.get_context('skin_type')
            print(f"Using skin type: {skin_type}")
            
            if skin_type:
                data = self.skincare_data["skin_types"].get(skin_type)
                if data:
                    print(f"Found data for {skin_type}: {data.keys()}")
                    response.append(f"🌸 {skin_type.capitalize()} Skin Care Guide 🌸")
                    response.append(f"\n📝 Description: {data.get('description', '')}")
                    
                    if 'recommendations' in data:
                        response.append("\n🔄 Recommended Routine:")
                        for step, product in data["recommendations"].items():
                            response.append(f"  • {step.capitalize()}: {product}")
                            
                    if 'key_ingredients' in data:
                        response.append("\n🔬 Key Ingredients:")
                        response.append(", ".join(data["key_ingredients"]))
                    
                    if entities.get('concern'):
                        concern_data = self.skincare_data["common_concerns"].get(entities['concern'])
                        if concern_data:
                            response.append(f"\n❗ For {entities['concern'].capitalize()}:")
                            response.append(f"  - Effective ingredients: {', '.join(concern_data.get('ingredients', []))}")
                            response.append(f"  - Recommended products: {', '.join(concern_data.get('product_types', []))}")
                else:
                    print(f"No data found for skin type: {skin_type}")
            else:
                response.append("🔍 Please tell me your skin type (oily/dry/combination/sensitive) to get personalized advice!")
                
            return "\n".join(response)
        except Exception as e:
            logging.error(f"Failed to generate skin response: {str(e)}")
            return ""

    def generate_nutrition_response(self, entities):
        try:
            print("\nGenerating Nutrition Response:")
            print(f"Entities: {entities}")
            print(f"Current context: {self.user_context}")
            
            if not self.nutrition_data.get("categories"):
                return "⚠️ Nutrition data not available. Please try again later."
                
            response = []
            category = entities.get('nutrition_category') or self.get_context('nutrition_category') or 'general_health'
            print(f"Using nutrition category: {category}")
            
            if category in self.nutrition_data["categories"]:
                data = self.nutrition_data["categories"][category]
                response.append(f"🍎 {data.get('description', '').capitalize()} 🥦")
                
                for food in data.get("foods", []):
                    response.append(f"\n🥗 {food.get('name', '')}:")
                    response.append(f"  - Benefits: {food.get('benefit', '')}")
                    if 'examples' in food:
                        response.append(f"  - Examples: {', '.join(food['examples'])}")
            
                science_note = self.nutrition_data.get("science", {}).get("nutrient_synergy", "")
                if science_note:
                    response.append(f"\n💡 Did you know? {science_note}")
            else:
                response.append("🍽️ General nutrition tips: Focus on whole foods, stay hydrated, and eat a colorful variety!")
                
            return "\n".join(response)
        except Exception as e:
            logging.error(f"Failed to generate nutrition response: {str(e)}")
            return ""


    def process_input(self, user_input):
        try:
            # Greeting handling
            if user_input.lower() in ['hi', 'hello', 'hey']:
                return random.choice([
                    "🌸 Hello! I'm your Skincare & Nutrition Bot! How can I help?",
                    "🍎 Hi there! Ask me about skincare or diet plans!",
                    "👋 Welcome! Let's talk about your health goals!"
                ])
        
            # Farewell handling
            farewell_keywords = ['bye', 'goodbye', 'see you', 'ok', 'thanks']
            if user_input.lower().strip() in farewell_keywords:
                return random.choice([
                    "Thank you for chatting! Feel free to ask me anything else!",
                    "🍎 It was great helping you! Come back anytime!",
                    "👋 Goodbye! Remember to stay hydrated and wear sunscreen!"
                ])

            # Check if API gave valid response
            api_response = self.ask_deepseek(user_input)
            if not api_response.startswith(("⏳", "🔌", "⚠️")):
                return api_response

            #Proceed with intent classification if API failed
            processed = self.preprocess_text(user_input)
            X = self.vectorizer.transform([processed])
            proba = self.clf.predict_proba(X)[0]
            max_prob = max(proba)
            intent = self.clf.predict(X)[0]
            
            print(f"\nIntent Classification:")
            print(f"Processed input: {processed}")
            print(f"Predicted intent: {intent} (confidence: {max_prob:.2f})")

            # Generate local JSON-based responses
            entities = self.detect_entities(user_input)
            
            if intent == "skin":
                response = self.generate_skin_response(entities)
            elif intent == "nutrition":
                response = self.generate_nutrition_response(entities)
            else:
                response = ""

            # Check if JSON response is valid
            fallback_phrases = ["please tell", "not available", "specify", 
                               "no data", "please specify", "not found"]
            if response.strip() and not any(phrase in response.lower() 
                                        for phrase in fallback_phrases):
                # Add debug logging HERE
                print(f"[JSON Response] {response}")
                print(f"[Context] {self.user_context}")
                return response

            
            qa_answer = self.match_question(user_input)
            if qa_answer:
                print(f"[TXT Match] {qa_answer}")
                return f"📚 Predefined Answer: {qa_answer}"

          
            print("[Final API Fallback]")
            context_str = self.get_context_string()
            return self.ask_deepseek(
                f"""Act as a dermatologist/nutritionist. Context: {context_str}
                Question: {user_input}
                Provide a comprehensive answer with:
                1. Scientific explanation (2-3 paragraphs)
                2. Step-by-step practical guidance
                3. Key safety considerations
                Format requirements:
                 - Minimum 500 words, maximum 1000 words
                 - No markdown or section headers
                 - End with natural conclusion
                 - No product recommendations unless explicitly asked
                 - Never add open-ended questions"""
            )

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return random.choice([ 
                " Hmm, I'm having trouble with that one.",
                "🌱 Let's try a different health question!",
                "💡 Tip: Ask about specific skin concerns or diets!"
            ])



# ---------------------- Recommender System Components ----------------------
@st.cache_data
def load_cosmetics_data():
    return pd.read_csv(Config.DATA_PATHS['cosmetics'])


def recommend_cosmetics(skin_type, label_filter, rank_filter, brand_filter, 
                       price_range, ingredient_input=None, num_recommendations=10):
    try:
        df = st.session_state.df
        
        skin_type_column = skin_type  
        
        recommended_products = df[df[skin_type_column] == 1]
        
        if label_filter != 'All':
            recommended_products = recommended_products[recommended_products['Label'] == label_filter]
        
        recommended_products = recommended_products[
            (recommended_products['Rank'] >= rank_filter[0]) & 
            (recommended_products['Rank'] <= rank_filter[1])
        ]
        
        if brand_filter != 'All':
            recommended_products = recommended_products[recommended_products['Brand'] == brand_filter]
        
        recommended_products = recommended_products[
            (recommended_products['Price'] >= price_range[0]) & 
            (recommended_products['Price'] <= price_range[1])
        ]

        if ingredient_input:
            input_vec = st.session_state.tfidf.transform([ingredient_input])
            cosine_similarities = cosine_similarity(input_vec, st.session_state.tfidf_matrix).flatten()
            recommended_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
            ingredient_recommendations = df.iloc[recommended_indices]
            recommended_products = recommended_products[
                recommended_products.index.isin(ingredient_recommendations.index)
            ]
        
        return recommended_products.sort_values(by=['Rank']).head(num_recommendations)
    except KeyError as e:
        st.error(f"Column not found in data: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return pd.DataFrame()



def show_visualizations(recommended_products):
    if not recommended_products.empty:
        try:
            st.subheader("Price Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(recommended_products['Price'], kde=True, bins=20, ax=ax)
            plt.xlabel('Price ($)')
            st.pyplot(fig)

            st.subheader("Top Brands in Recommendations")
            fig, ax = plt.subplots(figsize=(10, 4))
            brand_counts = recommended_products['Brand'].value_counts().head(5)
            brand_counts.plot(kind='bar', ax=ax, color='skyblue')
            plt.xticks(rotation=45)
            plt.xlabel('Brand')
            plt.ylabel('Number of Products')
            st.pyplot(fig)

            st.subheader("Rank vs Price Relationship")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(x='Rank', y='Price', data=recommended_products, 
                           hue='Brand', size='Price', sizes=(20, 200), ax=ax)
            plt.xlabel('Product Rank')
            plt.ylabel('Price ($)')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")


def calculate_metrics(recommended_products):
    metrics = {}
    if not recommended_products.empty:
        try:
            metrics['average_price'] = recommended_products['Price'].mean()
            metrics['average_rank'] = recommended_products['Rank'].mean()
            metrics['total_products'] = len(recommended_products)
            metrics['brand_diversity'] = len(recommended_products['Brand'].unique()) / len(recommended_products)
            metrics['category_diversity'] = len(recommended_products['Label'].unique()) / len(recommended_products)
            metrics['top_ranked'] = len(recommended_products[recommended_products['Rank'] <= 3]) / len(recommended_products)
        except Exception as e:
            st.error(f"Metrics error: {str(e)}")
    return metrics


def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.bot = SkincareNutritionBot()
    if 'df' not in st.session_state:
        st.session_state.df = load_cosmetics_data()
    if 'tfidf' not in st.session_state:
        # Initialize TF-IDF components once
        st.session_state.tfidf = TfidfVectorizer(stop_words='english')
        st.session_state.tfidf_matrix = st.session_state.tfidf.fit_transform(
            st.session_state.df['Ingredients']
        )

# ---------------------- Main Application ----------------------
def chat_interface():
    try:
        st.title("🌸🍎 HealthBot Pro - Your Skincare & Nutrition Assistant")
        
        # Custom CSS for chat bubbles
        st.markdown("""
        <style>
            .user-message {
                background-color: #FFB6C1 !important;  /* Light pink */
                color: #333333 !important;             /* Dark gray text */
                padding: 1em;
                border-radius: 15px 15px 0 15px;
                margin: 0.5em 0;
                max-width: 70%;
                float: right;
                clear: both;
                border: 1px solid #FF69B4;          /* Pink border */
                
            }
            .bot-message {
                background-color: #FFE4E1 !important;  /* Misty rose */
                color: #333333 !important;             /* Dark gray text */
                padding: 1em;
                border-radius: 15px 15px 15px 0;
                margin: 0.5em 0;
                max-width: 70%;
                float: left;
                clear: both;
                border: 1px solid #FFB6C1;           /* Light pink border */
                
            }
            .chat-container {
                padding-bottom: 100px;
            }
        </style>
        """, unsafe_allow_html=True)

        # Chat history
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            if not st.session_state.messages:
                st.markdown('<div class="bot-message">🤖 How can I help you today?</div>', 
                          unsafe_allow_html=True)
            else:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>',
                        unsafe_allow_html=True
                        )          
            st.markdown('</div>', unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask about skincare or nutrition..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = st.session_state.bot.process_input(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    except Exception as e:
        st.error(f"Chat error: {str(e)}")

def recommender_interface():
    try:
        st.title('✨ Skincare Products Recommendation System')
        
        with st.sidebar:
            st.header("Filter Preferences")
            skin_type = st.selectbox(
                'Select your skin type:', 
                ('Combination', 'Dry', 'Normal', 'Oily', 'Sensitive')
            )
            label_filter = st.selectbox(
                'Product Category:', 
                ['All'] + st.session_state.df['Label'].unique().tolist()
            )
            brand_filter = st.selectbox(
                'Brand Preference:', 
                ['All'] + st.session_state.df['Brand'].unique().tolist()
            )
            rank_filter = st.slider(
                'Product Rating Range:', 
                min_value=1, 
                max_value=5, 
                value=(3, 5)
            )
            price_range = st.slider(
                'Price Range ($):', 
                min_value=0.0,
                max_value=float(st.session_state.df['Price'].max()),
                value=(0.0, 100.0),
                step=5.0
            )
            ingredient_input = st.text_input(
                "Specific Ingredients (comma-separated):",
                placeholder="e.g., hyaluronic acid, vitamin C"
            )

        if st.button('🔍 Get Recommendations!', use_container_width=True):
            with st.spinner('Analyzing 1000+ products for your perfect match...'):
                try:
                    recommendations = recommend_cosmetics(
                        skin_type=skin_type,
                        label_filter=label_filter,
                        rank_filter=rank_filter,
                        brand_filter=brand_filter,
                        price_range=price_range,
                        ingredient_input=ingredient_input
                    )
                    
                    if not recommendations.empty:
                        st.success(f"Found {len(recommendations)} matching products!")
                        
                        # Display metrics
                        st.subheader("📊 Recommendation Insights")
                        metrics = calculate_metrics(recommendations)
                        
                        cols = st.columns(4)
                        metric_config = [
                            ("Average Price", f"${metrics.get('average_price', 0):.2f}", "$"),
                            ("Average Rating", f"{metrics.get('average_rank', 0):.1f}/5", "⭐"),
                            ("Brand Diversity", f"{metrics.get('brand_diversity', 0)*100:.1f}%", "🌐"),
                            ("Top Rated", f"{metrics.get('top_ranked', 0)*100:.1f}%", "🏆")
                        ]
                        
                        for col, (label, value, icon) in zip(cols, metric_config):
                            with col:
                                st.metric(f"{icon} {label}", value)

                        # Visualizations
                        show_visualizations(recommendations)

                        # Product Display
                        st.subheader("💄 Recommended Products")
                        st.dataframe(
                            recommendations[
                                ['Brand', 'Name', 'Label', 'Price', 'Rank', 'Ingredients']
                            ].sort_values(by=['Rank', 'Price']),
                            height=600,
                            column_config={
                                "Price": st.column_config.NumberColumn(
                                    format="$%.2f"
                                ),
                                "Rank": st.column_config.NumberColumn(
                                    format="⭐%.1f"
                                )
                            },
                            use_container_width=True
                        )
                    else:
                        st.warning("""
                        🧴 No products found matching your criteria. Try:
                        1. Expanding price range
                        2. Removing brand filters
                        3. Checking different categories
                        """)

                except Exception as e:
                    st.error(f"Recommendation error: {str(e)}")
                    logging.error(f"Recommendation failure: {str(e)}")

    except Exception as e:
        st.error(f"Interface error: {str(e)}")
        logging.error(f"Recommender interface crash: {str(e)}")

# ---------------------- App Control ----------------------
def main():
    init_session_state()
    
    with st.sidebar:
        st.title("Navigation")
        app_mode = st.radio(
            "Choose Mode:",
            ["Chat Bot", "Product Recommender"],  
            horizontal=True,
            label_visibility="visible"
        )
        
        st.markdown("---")
        st.markdown("### About HealthBot Pro")
        st.markdown("""
        Your AI-powered wellness assistant providing:
        - Personalized skincare advice
        - Nutrition guidance
        - Cosmetic product recommendations
        """) 

    try:
        if app_mode == "Chat Bot":
            chat_interface()
        elif app_mode == "Product Recommender":
            recommender_interface()
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logging.error(f"Runtime error: {str(e)}")


if __name__ == "__main__":
    try:
        # Verify NLTK data
        nltk_dependencies = ['wordnet', 'punkt', 'stopwords']
        for dep in nltk_dependencies:
            try:
                nltk.data.find(f'tokenizers/{dep}')
            except LookupError:
                nltk.download(dep)
        
        test_bot = SkincareNutritionBot()
        print("API TEST:", test_bot.ask_deepseek("What's 1+1?"))  
        
        main()
    except Exception as e:
        st.error("Critical application failure. Please check logs.")
        logging.critical(f"Startup failure: {str(e)}")
