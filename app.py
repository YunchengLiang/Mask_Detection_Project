from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/dojo')
def dojo():
    return 'Dojo!'

@app.route('/say/<name>')
def say(name):
    return f'Hi {name}!'

@app.route('/dojo/<int:num>')
def dojo_num(num):
    return 'the num is {}'.format(num)    

@app.route('/dojo/list')
def dojo_list():
    page = request.args.get('page', default = 1, type = int)
    return 'the page is {}'.format(page)

if __name__ == '__main__':
    app.run(debug = True)