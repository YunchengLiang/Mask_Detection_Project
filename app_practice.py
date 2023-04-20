from flask import Flask, request, render_template
from datetime import datetime
app = Flask(__name__)

class user:
    def __init__(self, name, email):
        self.name = name
        self.email = email

def todatetime(value, format='%Y-%m-%d'):
    return value.strftime(format)
app.add_template_filter(todatetime, 'datetime')

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/dojo/<name>')
def dojo(name):
    return render_template('dojo.html', name = name)

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


@app.route('/user')
def hello_world_v1():
    person={'name':'jerry','age':29}
    hobby='basketball'
    newuser=user(name='michael',email='870327837@qq.com')
    date=datetime.now()
    return render_template('user.html',user=newuser,person=person,hobby=hobby,date=date)

@app.route('/control')
def control():
    person={'name':'jerry','age':29}
    songs=[
        {'singer':'Badsakikush',
         'song':'floating'},
        {'singer':'Badsakikush',
         'song':'outlawz'},
           ]
    return render_template('control.html',person=person,songs=songs)

@app.route('/child1')
def child1():
    return render_template('child1.html')

if __name__ == '__main__':
    app.run(debug = True)