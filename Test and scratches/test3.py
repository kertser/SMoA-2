"""
The algorithm evaluates the quality of a response based on semantic similarity, keyword matching, and length. It
generates embeddings for the prompt, context, and response using a pre-trained transformer model and calculates
cosine similarity scores to measure semantic relevance. A keyword match score is computed by comparing key terms in
the prompt and response. A length penalty is applied to optimize response length.
The final quality score is a weighted combination of similarity, keyword match, and length,
with penalties for poor keyword matches.
The response is then classified as "Excellent," "Good," "Satisfactory," or "Unsatisfactory."

"""

from sentence_transformers import SentenceTransformer, util
import warnings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Setup block for one-time downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Load the multilingual bi-encoder model (outside the function to avoid reloading)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


def preprocess_text(text):
    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + stopwords.words('russian'))
    return [word for word in tokens if word.isalnum() and word not in stop_words]


def keyword_match(prompt, response):
    prompt_words = set(preprocess_text(prompt))
    response_words = set(preprocess_text(response))
    if not prompt_words:
        return 0
    match_ratio = len(prompt_words.intersection(response_words)) / len(prompt_words)

    # Stronger penalty for low keyword match
    if match_ratio < 0.2:  # Less than 20% match
        return match_ratio * 0.5  # Apply a heavy penalty
    return match_ratio


def length_penalty(response, optimal_length=50):
    # Adjusting sigmoid to avoid skewing the length penalty
    length_ratio = len(response.split()) / optimal_length
    return 2 / (1 + pow(2.718, -length_ratio))


def evaluate_answer(incoming_prompt, operator_context, outgoing_response):
    try:
        # Get embeddings for the question, context, and answer
        embeddings = model.encode([incoming_prompt, operator_context, outgoing_response], convert_to_tensor=True)
        prompt_embedding = embeddings[0]
        context_embedding = embeddings[1]
        response_embedding = embeddings[2]

        # Calculate cosine similarities
        score_prompt_response = util.cos_sim(prompt_embedding, response_embedding).item()
        score_context_response = util.cos_sim(context_embedding, response_embedding).item()

        # Keyword matching score
        keyword_score = keyword_match(incoming_prompt, outgoing_response)

        # Weighted average score (more weight to keyword match now)
        relevance_score = (0.35 * score_prompt_response + 0.15 * score_context_response + 0.5 * keyword_score)

        # Apply length penalty
        length_score = length_penalty(outgoing_response)
        quality_score = relevance_score * length_score

        # Adjust quality score based on keyword presence (heavier penalty)
        if keyword_score < 0.3:  # If less than 30% of keywords are present
            quality_score *= 0.7  # Reduce the score by 30%

        # Classification based on threshold
        if quality_score > 0.75:
            classification = "Excellent answer"
        elif quality_score > 0.6:
            classification = "Good answer"
        elif quality_score > 0.4:
            classification = "Satisfactory answer"
        else:
            classification = "Unsatisfactory answer"

        print(f"Prompt: {incoming_prompt}")
        print(f"Response: {outgoing_response}")
        print(f"Context: {operator_context}")
        print(f"Relevance score: {relevance_score:.4f}")
        print(f"Keyword score: {keyword_score:.4f}")
        print(f"Length score: {length_score:.4f}")
        print(f"Quality score: {quality_score:.4f}")
        print(f"Classification: {classification}")
        return quality_score
    except Exception as e:
        print(f"Error occurred: {e}")
        return None


# Extended list of test examples (multilingual)
tests = [
    # Mathematics (Russian)
    ("Что такое интеграл?", "Объяснить основные понятия математического анализа.",
     "Интеграл — это обобщение операции суммирования, которое используется для вычисления площадей, объемов и других величин в математическом анализе.",
     "Correct answer"),
    ("Как решать квадратные уравнения?", "Рассказать о способах решения алгебраических уравнений.",
     "Квадратные уравнения решаются по формуле дискриминанта: x = (-b ± √(b² - 4ac)) / (2a), где a, b, c - коэффициенты уравнения ax² + bx + c = 0.",
     "Correct answer"),
    ("Что такое производная функции?", "Объяснить основные понятия дифференциального исчисления.",
     "Производная функции — это предел отношения приращения функции к приращению аргумента, когда приращение аргумента стремится к нулю.",
     "Correct answer"),
    ("Как вычислить площадь круга?", "Рассказать о геометрических формулах.",
     "Площадь прямоугольника равна произведению его длины на ширину.",
     "Incorrect answer"),

    # History (English)
    ("When did World War II end?", "Provide key dates of 20th century history.",
     "World War II ended in 1945 with the surrender of Germany in May and Japan in August.",
     "Correct answer"),
    ("Who was the first woman to fly solo across the Atlantic Ocean?", "Discuss important figures in aviation history.",
     "Amelia Earhart was the first woman to fly solo across the Atlantic Ocean in 1932.",
     "Correct answer"),
    ("What was the main cause of the French Revolution?",
     "Explain the socio-economic conditions in 18th century France.",
     "The main causes of the French Revolution included financial crisis, social inequality, and political corruption in the monarchy.",
     "Correct answer"),
    ("Who invented the telephone?", "Discuss the Industrial Revolution and technological advancements.",
     "The light bulb was invented by Thomas Edison.",
     "Incorrect answer"),

    # Science (Russian)
    ("Что такое фотосинтез?", "Объяснить основные процессы в биологии растений.",
     "Фотосинтез — это процесс, при котором растения используют энергию солнечного света для преобразования углекислого газа и воды в глюкозу и кислород.",
     "Correct answer"),
    ("Каковы основные части клетки?", "Рассказать о строении клетки.",
     "Основные части клетки включают ядро, цитоплазму, клеточную мембрану, митохондрии, эндоплазматический ретикулум и аппарат Гольджи.",
     "Correct answer"),
    ("Что такое гравитация?", "Объяснить фундаментальные силы в физике.",
     "Гравитация — это сила притяжения между всеми объектами во Вселенной, зависящая от их масс и расстояния между ними.",
     "Correct answer"),
    ("Как работает вакцина?", "Объяснить принципы иммунологии.",
     "Вакцины содержат ослабленные или убитые патогены, которые вызывают иммунный ответ без развития болезни.",
     "Partially correct answer"),

    # Literature (English)
    ("Who wrote 'Pride and Prejudice'?", "Discuss famous English authors.",
     "Jane Austen wrote 'Pride and Prejudice', which was first published in 1813.",
     "Correct answer"),
    ("What is the main theme of 'To Kill a Mockingbird'?", "Analyze themes in American literature.",
     "The main themes of 'To Kill a Mockingbird' include racial injustice, the loss of innocence, and moral education.",
     "Correct answer"),
    ("Who is considered the national poet of Russia?", "Discuss important figures in Russian literature.",
     "Alexander Pushkin is considered the national poet of Russia and the founder of modern Russian literature.",
     "Correct answer"),
    ("What is the plot of 'Romeo and Juliet'?", "Explain the works of William Shakespeare.",
     "Hamlet is a tragedy about the Prince of Denmark seeking revenge for his father's murder.",
     "Incorrect answer"),

    # Technology (Russian)
    ("Что такое искусственный интеллект?", "Объяснить современные технологии.",
     "Искусственный интеллект — это область информатики, занимающаяся созданием интеллектуальных машин, которые могут выполнять задачи, требующие человеческого интеллекта.",
     "Correct answer"),
    ("Как работает блокчейн?", "Рассказать о криптовалютах и их технологиях.",
     "Блокчейн — это децентрализованная, распределенная база данных, которая хранит информацию о транзакциях в виде цепочки блоков.",
     "Correct answer"),
    ("Что такое машинное обучение?", "Объяснить методы анализа данных.",
     "Машинное обучение — это метод анализа данных, который автоматизирует построение аналитических моделей.",
     "Correct answer"),
    ("Как работает 5G?", "Объяснить принципы беспроводной связи.",
     "4G — это четвертое поколение мобильной связи, обеспечивающее высокоскоростной интернет.",
     "Incorrect answer")
]

for prompt, context, response, description in tests:
    print(f"\n{description}:")
    evaluate_answer(prompt, context, response)
