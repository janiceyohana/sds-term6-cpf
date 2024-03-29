from flask import Flask, render_template, request
from openbalance_forecast import generate_graph

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-graph', methods=['POST'])
def generate_and_display_graph():
    # Get user input from the form
    period = int(request.form['period'])
    no_CSA = int(request.form['no_CSA'])
    no_CSE = int(request.form['no_CSE'])
    no_Temps = int(request.form['no_Temps'])
    no_TL = int(request.form['no_TL'])

    # Generate graph and get the path to the generated image
    graph_path = generate_graph(period, no_CSA, no_CSE, no_Temps, no_TL)

    # Render template to display the graph
    return render_template('graph.html', graph_path=graph_path)

if __name__ == '__main__':
    app.run(debug=True)

