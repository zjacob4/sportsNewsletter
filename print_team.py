from flask import Flask, render_template
from news_team import write_stories
from create_stylesheet import write_css_stylesheet
from graphics_team import add_graphics
from research_team import research_stories
from fact_check_team import fact_check_stories

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch the daily news headlines from an API or a database
    headlines = [
        # Add more headlines here
    ]
    
    # Generate or fetch the result
    stories = research_stories()
    stories = fact_check_stories(stories)
    with open('latest_text/latest_research_fact_check.txt', 'w') as file:
        file.write(stories)
    result = write_stories(stories)
    result = fact_check_stories(result)
    with open('latest_text/latest_writing_fact_check.txt', 'w') as file:
        file.write(result)
    result = add_graphics(result)
    write_css_stylesheet(result, 'static/styles_few_shot.css')

    return render_template('index.html', headlines=headlines, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)