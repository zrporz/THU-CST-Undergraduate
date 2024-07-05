from flask import Flask, render_template, request, make_response
import sqlite3


# 启动flask
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def csrf_attack():
    return render_template('csrf_attack.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)