from flask import Flask, render_template
from news_team import write_stories
from create_stylesheet import write_css_stylesheet

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch the daily news headlines from an API or a database
    headlines = [
        "Breaking News:...",
        # Add more headlines here
    ]
    
    # Generate or fetch the result
    result = write_stories()
    write_css_stylesheet(result)

    return render_template('index.html', headlines=headlines, result=result)

if __name__ == '__main__':
    app.run()