from flask import Flask, render_template
from news_team import write_stories, get_readable_result

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
    result = get_readable_result(result)
    

    return render_template('index.html', headlines=headlines, result=result)

if __name__ == '__main__':
    app.run()