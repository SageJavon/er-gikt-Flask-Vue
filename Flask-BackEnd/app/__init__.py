"""
初始化文件
"""
import os.path

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
import sys
from urllib import parse


sys.path.append('..')

sys.path.append(os.path.join(os.path.dirname(__file__), 'alg'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'view'))
sys.path.append(
    'C:/Users/wuyz/Desktop/er-gikt-Flask-Vue-main/Flask-BackEnd')
# 解决了算法模块无法引用的问题
sys.path.append(
    'D:/LenovoSoftstore/Install/anaconda3/envs/python39/Lib/site-packages/flask/app.py')
# 解决了数据库迁移找不到包的问题

password = "Wyz789789@"

# 主机名
HOSTNAME = '127.0.0.1'
# mysql端口号
PORT = 3306
# 用户名
USERNAME = 'root'
# 密码
PASSWORD = parse.quote_plus(password)
# 数据库名称
DATABASE = 'test'

# 主文件配置

db = SQLAlchemy()  # 获取app中配置好的数据库


def create_app():
    # 声明
    app = Flask(__name__, instance_relative_config=True)
    cors = CORS(app)

    # 设置数据库配置,连接格式：dialect://username:password@host:port/database
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}'
    # mysql8版本以上不需要加任何参数
    app.config['DEBUG'] = True
    app.config['ENV'] = 'development'
    app.config['FLASK_APP'] = 'app'

    db.init_app(app)
    migrate = Migrate(app, db)

    # 这时候才导入蓝图，可以避免循环引用
    # from view.user_bp import user_bp
    # from view.skill_bp import skill_bp
    # from view.kt_bp import kt_bp
    # from view.question_bp import question_bp
    from app.view.user_bp import user_bp
    from app.view.skill_bp import skill_bp
    from app.view.kt_bp import kt_bp
    from app.view.question_bp import question_bp
    # from app.view.score import score_bp

    # app.register_blueprint(score_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(skill_bp)
    app.register_blueprint(kt_bp)
    app.register_blueprint(question_bp)

    # 测试内容
    with app.app_context():
        # db.create_all()
        # 生成初始数据，只需跑一次即可
        # from app.create_data import create_data
        # create_data()
        pass

    # ic(app.config) # 打印app配置
    return app


if __name__ == '__main__':
    app = create_app()
    # app.run()
    app.run(host='127.0.0.1',port=1000,debug=True)
