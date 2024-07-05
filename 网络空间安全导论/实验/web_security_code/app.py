from flask import Flask, render_template, request, make_response, session
import sqlite3
import os
import hashlib
import hmac

# 生成CSRF token
def generate_csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = hmac.new(app.secret_key, os.urandom(64), hashlib.sha256).hexdigest()
    return session['_csrf_token']

def validate_csrf_token(token):
    return token == session.get('_csrf_token')

# 连接数据库
def connect_db():
    db = sqlite3.connect('test.db')
    db.cursor().execute('CREATE TABLE IF NOT EXISTS comments '
                        '(id INTEGER PRIMARY KEY, '
                        'comment TEXT)')
    db.cursor().execute('CREATE TABLE IF NOT EXISTS users '
                        '(user_id TEXT PRIMARY KEY, '
                        'password TEXT)')
    db.commit()
    return db

# 添加评论
def add_comment(comment):
    db = connect_db()
    db.cursor().execute('INSERT INTO comments (comment) '
                        'VALUES (?)', (comment,))
    db.commit()

# 得到评论
def get_comments(search_query=None):
    db = connect_db()
    results = []
    get_all_query = 'SELECT comment FROM comments'
    for (comment,) in db.cursor().execute(get_all_query).fetchall():
        if search_query is None or search_query in comment:
            results.append(comment)
    return results

def set_login_cookie(response, user_id):
    response.set_cookie('user_id', user_id)
    print(f'set_login_cookie {user_id}')
    return response

# 启动flask
app = Flask(__name__)
app.secret_key = os.urandom(24)  # 使用os.urandom生成随机的密钥

@app.route('/', methods=['GET', 'POST'])
def index():
    login_info = None
    user_id = request.cookies.get('user_id')
    print(f'login_cookie is {user_id}')
    response = make_response()
    print(f"request.method:{request.method}")
    if request.method == 'POST':
        
        if 'comment' in request.form.keys():
            csrf_token = request.form.get('csrf_token')
            if not validate_csrf_token(csrf_token):
                return "CSRF token is invalid", 400
            if user_id:
                add_comment(request.form['comment'])
            else:
                return "Please log in to comment"
            # add_comment(request.form['comment'])
        elif  'user_id' and 'password' in request.form.keys():
            login_info, status = login(request.form['user_id'],request.form['password'])
            print(f"after login login_info is{login_info}")
            if status:
                response = set_login_cookie(response, request.form['user_id'])
        elif 'register_user_id' and 'register_password' in request.form.keys():
            login_info = register(request.form['register_user_id'],request.form['register_password'])

    if user_id:
        login_info = f"Welcome {user_id}!"
        
    search_query = request.args.get('q')

    comments = get_comments(search_query)
    print(f"login_info:{login_info}")
    
    response.data = render_template('index.html',
                                    comments=comments,
                                    search_query=search_query,
                                    login_info=login_info,
                                    csrf_token=generate_csrf_token())
    return response


def register(user_id, password):
    # 连接数据库
    print(f"user_id:{user_id}")
    db = connect_db()
    existing_user = db.cursor().execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    print(f"existing_user={existing_user}")
    # 检查用户名是否已存在
    if existing_user:
        
        return "User is existed!"
    else:
        db.cursor().execute('INSERT INTO users (user_id, password) '
                              'VALUES (?, ?)', (user_id, password))
        db.commit()
        print(f"注册用户:{user_id},密码：{password}")
        return f"Successfully register!{user_id}"
    
def login(user_id, password):
    # 连接数据库
    print(f"user_id:{user_id}")
    db = connect_db()
    existing_user = db.cursor().execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    # sql_request = f"SELECT * FROM users WHERE user_id = '{user_id}'"
    # print("sql_request:", sql_request)
    # existing_user = db.cursor().execute(sql_request).fetchone()
    print(f"existing_user={existing_user}")

    # 检查用户名是否已存在
    if existing_user:
        comment = db.cursor().execute('SELECT * FROM users WHERE user_id = ? AND password = ?', (user_id, password)).fetchone()
        # sql_request = f"SELECT * FROM users WHERE user_id = '{user_id}' AND password = '{password}'"
        # print("sql_request:", sql_request)
        # comment = db.cursor().execute(sql_request).fetchone()
        if comment:
            print("登录成功",comment)
            return (f"Welcome back!{comment[0]}", True)
        else:
            print("密码错误")
            return ("Password Error!", False)
    else:
        return (f"User {user_id} is not existed!", False)


if __name__ == '__main__':
    app.jinja_env.globals['csrf_token'] = generate_csrf_token  # 在模板中全局可用
    app.run(debug=True)