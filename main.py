import logging
from flask import Flask, render_template
from news_team import write_stories
from create_stylesheet import write_css_stylesheet

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    try:
        # Fetch the daily news headlines from an API or a database
        headlines = [
            # Add more headlines here
        ]
        
        # Generate or fetch the result
        result = write_stories()
        write_css_stylesheet(result)

        return render_template('index.html', headlines=headlines, result=result)
    except Exception as e:
        logging.error(f"Error in index route: {e}")
        return "An error occurred", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)