from flask import Flask, request, render_template
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Initialize Flask application
app = Flask(__name__)

# IBM Watson service credentials
apikey = 'zu_AuOpy9ZXwoYirYouh7UfFUH_RNJ9lLTpBpTx39vbm'
url = 'https://api.us-east.natural-language-understanding.watson.cloud.ibm.com/instances/1df484b4-7e69-4781-9810-510ddc5a44f3'

# Authenticate to IBM Watson
authenticator = IAMAuthenticator(apikey)
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator
)
natural_language_understanding.set_service_url(url)

@app.route('/')
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/emotionDetector', methods=['GET'])
def analyze_text():
    # Extract text from the request
    text_to_analyze = request.args.get('textToAnalyze')

    # Analyze the text using IBM Watson
    response = natural_language_understanding.analyze(
        text=text_to_analyze,
        features=Features(entities=EntitiesOptions(), keywords=KeywordsOptions())
    ).get_result()

    # Return the analysis results
    return str(response)

if __name__ == '__main__':
    app.run(debug=True)
