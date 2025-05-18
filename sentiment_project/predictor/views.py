from django.shortcuts import render
import os
from django.conf import settings
import pickle

model_path = os.path.join(settings.BASE_DIR, 'sentiment_model.pkl')
model = pickle.load(open(model_path, 'rb'))
vectorizer_path = os.path.join(settings.BASE_DIR, 'vectorizer.pkl')
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

def predict_sentiment(request):
    if request.method == 'POST':
        review = request.POST.get('review')
        if review:
            review_vector = vectorizer.transform([review])
            prediction = model.predict(review_vector)
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            return render(request, 'result.html', {'sentiment': sentiment, 'review': review})
    return render(request, 'form.html')

