from flask import Flask, render_template, send_file, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

emotion_classifier = pipeline(
    "text-classification", 
    model=AutoModelForSequenceClassification.from_pretrained("emotion_model", ignore_mismatched_sizes=True),
    tokenizer=AutoTokenizer.from_pretrained("emotion_model")
)

daily_sentiment = None

sentiment_map = {
    'happy': 1,
    'love': 1,
    'neutral': 0,
    'sad': -1,
    'anger': -1,
    'fear': -1,
    'surprise': 1,
    'admiration': 1,
    'amusement': 1,
    'annoyance': -1,
    'approval': 1,
    'caring': 1,
    'confusion': 0,
    'curiosity': 0,
    'desire': 1,
    'disappointment': -1,
    'disapproval': -1,
    'disgust': -1,
    'embarrassment': -1,
    'excitement': 1,
    'gratitude': 1,
    'grief': -1,
    'joy': 1,
    'nervousness': -1,
    'optimism': 1,
    'pride': 1,
    'realization': 1,
    'relief': 1,
    'remorse': -1,
    'sadness': -1,
    'surprise': 1,
    'neutral': 0,
}


@app.route('/product')
def product():
    global daily_sentiment

    file_name = request.args.get('file', 'default.csv')  

    try:
        df = pd.read_csv(file_name)

        if 'Review' not in df.columns:
            raise ValueError("The input CSV file must have a 'Review' column.")
        
        def get_emotion(review_text):
            if not isinstance(review_text, str) or not review_text.strip():
                return "Unknown" 
            try:
                result = emotion_classifier(review_text)
                return result[0]['label']  
            except Exception as e:
                return "Error"
        
        
        df['Emotion'] = df['Review'].apply(get_emotion)

        df['sentiment_score'] = df['Emotion'].map(sentiment_map)

        df['date_time'] = pd.to_datetime(df['date_time'])

        df['date'] = df['date_time'].dt.date

        daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()

        daily_sentiment['positive_reviews'] = df[df['sentiment_score'] > 0].groupby('date').size().reset_index(name='count')['count']
        daily_sentiment['negative_reviews'] = df[df['sentiment_score'] < 0].groupby('date').size().reset_index(name='count')['count']

        first_five_records = df.head(5)

        def get_most_popular_emotion(df):
            emotion_counts = df['Emotion'].value_counts()
            if emotion_counts.index[0] == 'neutral':
                emotion_counts = emotion_counts[emotion_counts.index != 'neutral']
                if len(emotion_counts) > 0:
                    return emotion_counts.index[0]
                else:
                    return None  
            else:
                return emotion_counts.index[0]


        most_popular_emotion = get_most_popular_emotion(df)

        positive_reviews_count = len(df[df['sentiment_score'] > 0])
        total_reviews_count = len(df)
        positive_percentage = (positive_reviews_count / total_reviews_count) * 100 if total_reviews_count > 0 else 0

        return render_template('product.html', 
                               product_name=file_name.split('.')[0].capitalize(),
                               first_five_records=first_five_records.to_html(classes="table table-striped", index=False),
                               most_popular_emotion=most_popular_emotion,
                               positive_percentage=positive_percentage)

    except FileNotFoundError:
        return f"File {file_name} not found.", 404




@app.route('/')
def home():
    return render_template('main.html')

@app.route('/product1.html')
def product1():
    return render_template('product1.html')

@app.route('/product2.html')
def product2():
    return render_template('product2.html')

@app.route('/product3.html')
def product3():
    return render_template('product3.html')

@app.route('/plot.png')
def plot_png():
    global daily_sentiment  

    if daily_sentiment is None:
        return "No data available to plot. Please process a product first.", 400
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_sentiment['date'], daily_sentiment['sentiment_score'], label='Average Sentiment Score', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Sentiment Trend Over Time')
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
