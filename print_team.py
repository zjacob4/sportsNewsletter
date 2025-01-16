from flask import Flask, render_template
from news_team import write_stories
from create_stylesheet import write_css_stylesheet
from graphics_team import add_graphics

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch the daily news headlines from an API or a database
    headlines = [
        # Add more headlines here
    ]
    
    # Generate or fetch the result
    result = write_stories()
    result = add_graphics(result)
    write_css_stylesheet(result)

    return render_template('index.html', headlines=headlines, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)