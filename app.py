# 设置matplotlib使用非交互式后端，避免多线程问题
import matplotlib

matplotlib.use('Agg')  # 必须在其他matplotlib导入之前设置

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime, timedelta
import scipy.optimize as optimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.font_manager import FontProperties

# 创建Flask应用实例
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 文件上传配置
UPLOAD_FOLDER = 'uploads'
HISTORY_FOLDER = 'history_data'
RESOURCE_FOLDER = 'resource_data'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 创建必要的目录
for folder in [UPLOAD_FOLDER, HISTORY_FOLDER, RESOURCE_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 存储上传的数据，避免重复处理
data_cache = {}
pollution_history = {}
resource_data_cache = {}
extraction_history = {}


# 设置中文字体
def set_chinese_font():
    try:
        # 尝试使用系统中的中文字体
        font_paths = ['C:/Windows/Fonts/simhei.ttf',  # Windows简黑
                      'C:/Windows/Fonts/msyh.ttf',  # 微软雅黑
                      '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']  # Linux文泉驿

        for font_path in font_paths:
            if os.path.exists(font_path):
                plt.rcParams['font.family'] = ['simhei']  # 使用简黑字体
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                return

                # 如果找不到指定字体，使用matplotlib内置支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"设置中文字体时出错: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_coal_layer(data):
    coal_conditions = (
            (data['双侧向电阻率'] >= 50) & (data['双侧向电阻率'] <= 2000) &
            (data['声波时差'] >= 300) & (data['声波时差'] <= 600) &
            (data['自然伽玛'] >= 20) & (data['自然伽玛'] <= 80) &
            (data['密度'] >= 1.0) & (data['密度'] <= 1.8))
    return coal_conditions


def get_coal_depth_ranges(data, coal_mask):
    coal_depths = data.loc[coal_mask, '深度']
    depth_ranges = []
    if not coal_depths.empty:
        current_start = coal_depths.iloc[0]
        current_end = coal_depths.iloc[0]
        for depth in coal_depths.iloc[1:]:
            if depth - current_end > 1:
                depth_ranges.append((current_start, current_end))
                current_start = depth
            current_end = depth
        depth_ranges.append((current_start, current_end))
    return depth_ranges


# 计算煤层资源储量
def calculate_coal_resources(data, coal_mask, area_square_meters=10000):
    """计算煤炭资源储量"""
    coal_data = data[coal_mask]

    if coal_data.empty:
        return {"total_resources": 0, "layers": []}

        # 计算每个煤层的资源情况
    depth_ranges = get_coal_depth_ranges(data, coal_mask)
    coal_layers = []
    total_volume = 0

    for i, (start, end) in enumerate(depth_ranges):
        thickness = end - start
        # 获取该煤层内的煤炭数据
        layer_data = data[(data['深度'] >= start) & (data['深度'] <= end)]

        # 计算煤层体积和质量
        layer_volume = thickness * area_square_meters  # 厚度*面积
        avg_density = layer_data['密度'].mean()
        layer_mass = layer_volume * avg_density * 1000  # 转换为吨

        # 煤层品质评估
        quality_score = assess_coal_quality(layer_data)

        # 计算开采难度
        mining_difficulty = assess_mining_difficulty(start, end, thickness)

        coal_layers.append({
            "layer_number": i + 1,
            "start_depth": float(start),
            "end_depth": float(end),
            "thickness": float(thickness),
            "volume": float(layer_volume),
            "density": float(avg_density),
            "mass_tons": float(layer_mass),
            "quality": quality_score,
            "mining_difficulty": mining_difficulty
        })

        total_volume += layer_volume

        # 计算总储量
    avg_density = data.loc[coal_mask, '密度'].mean()
    total_resources = total_volume * avg_density * 1000  # 转换为吨

    return {
        "total_resources": float(total_resources),
        "layers": coal_layers,
        "total_volume": float(total_volume),
        "area_square_meters": float(area_square_meters)
    }


# 评估煤炭品质
def assess_coal_quality(layer_data):
    """评估煤炭品质"""
    # 煤质评分：基于密度和伽马值的加权平均
    # 密度越低、伽马值越低通常表示煤质更好
    avg_density = layer_data['密度'].mean()
    avg_gamma = layer_data['自然伽玛'].mean()

    # 密度评分：1.1-1.8范围内，密度越低分数越高
    density_score = max(0, min(100, (1.8 - avg_density) / 0.7 * 100))

    # 伽马评分：20-80范围内，伽马值越低分数越高
    gamma_score = max(0, min(100, (80 - avg_gamma) / 60 * 100))

    # 总体品质评分（0-100）
    quality_score = 0.6 * density_score + 0.4 * gamma_score

    # 品质等级
    if quality_score >= 90:
        quality_grade = "特优"
    elif quality_score >= 75:
        quality_grade = "优质"
    elif quality_score >= 60:
        quality_grade = "良好"
    elif quality_score >= 45:
        quality_grade = "中等"
    else:
        quality_grade = "低质"

    return {
        "score": float(quality_score),
        "grade": quality_grade,
        "density": float(avg_density),
        "gamma": float(avg_gamma)
    }


# 评估开采难度
def assess_mining_difficulty(start_depth, end_depth, thickness):
    """评估开采难度"""
    avg_depth = (start_depth + end_depth) / 2

    # 深度因子：深度越大，开采难度越大
    depth_factor = min(10, avg_depth / 100)

    # 厚度因子：煤层过薄或过厚都会增加开采难度
    # 最佳厚度范围约为2-5米
    if thickness < 1:
        thickness_factor = 8
    elif thickness < 2:
        thickness_factor = 5
    elif thickness < 5:
        thickness_factor = 2
    elif thickness < 8:
        thickness_factor = 4
    else:
        thickness_factor = 6

        # 计算总难度（1-10分）
    difficulty_score = (depth_factor * 0.7 + thickness_factor * 0.3)

    # 难度等级
    if difficulty_score < 3:
        difficulty_grade = "容易"
    elif difficulty_score < 5:
        difficulty_grade = "中等"
    elif difficulty_score < 7:
        difficulty_grade = "困难"
    else:
        difficulty_grade = "极困难"

    return {
        "score": float(difficulty_score),
        "grade": difficulty_grade,
        "depth_factor": float(depth_factor),
        "thickness_factor": float(thickness_factor)
    }


# 优化开采规划
def optimize_mining_plan(coal_layers, extraction_rate=0.85):
    """优化开采顺序和方法"""
    # 设置中文字体
    set_chinese_font()

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 准备数据
    layers = sorted(coal_layers, key=lambda x: x["start_depth"])
    layer_numbers = [f"煤层{l['layer_number']}" for l in layers]
    qualities = [l["quality"]["score"] for l in layers]
    difficulties = [l["mining_difficulty"]["score"] * 10 for l in layers]  # 缩放为0-100
    resources = [l["mass_tons"] / 1000 for l in layers]  # 转换为千吨

    # 计算优先级得分 = 品质 * 0.5 + (100-难度) * 0.3 + 储量占比 * 0.2
    max_resource = max(resources) if resources else 1
    priorities = []
    for i in range(len(layers)):
        resource_factor = resources[i] / max_resource * 100
        priority = qualities[i] * 0.5 + (100 - difficulties[i]) * 0.3 + resource_factor * 0.2
        priorities.append(priority)

        # 排序并生成开采顺序建议
    combined_data = list(zip(layer_numbers, qualities, difficulties, resources, priorities, layers))
    ordered_layers = sorted(combined_data, key=lambda x: x[4], reverse=True)

    mining_plan = []
    for i, (name, quality, difficulty, resource, priority, layer) in enumerate(ordered_layers):
        # 计算最佳开采方法
        method = determine_mining_method(layer)

        # 计算预期回收率
        recovery_rate = calculate_recovery_rate(layer, method, extraction_rate)

        # 预期产量
        expected_output = layer["mass_tons"] * recovery_rate

        mining_plan.append({
            "order": i + 1,
            "layer": layer["layer_number"],
            "depth_range": f"{layer['start_depth']:.1f}m - {layer['end_depth']:.1f}m",
            "quality_score": float(quality),
            "difficulty_score": float(difficulty / 10),  # 转回1-10分
            "resource_ktons": float(resource),
            "priority_score": float(priority),
            "recommended_method": method["name"],
            "method_details": method["description"],
            "expected_recovery_rate": float(recovery_rate * 100),  # 百分比
            "expected_output_tons": float(expected_output)
        })

        # 绘制柱状图比较
    x = np.arange(len(layer_numbers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, qualities, width, label='品质得分', color='#4CAF50')
    rects2 = ax.bar(x, difficulties, width, label='开采难度', color='#F44336')
    rects3 = ax.bar(x + width, priorities, width, label='优先级指数', color='#2196F3')

    ax.set_title('煤层开采优先级分析')
    ax.set_ylabel('得分')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_numbers)
    ax.legend()

    plt.tight_layout()

    # 将图表转换为base64编码
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return {
        "mining_plan": mining_plan,
        "priority_chart": plot_data
    }


# 确定最佳开采方法
def determine_mining_method(layer):
    """基于煤层特性确定开采方法"""
    depth = layer["start_depth"]
    thickness = layer["thickness"]
    difficulty = layer["mining_difficulty"]["score"]

    if depth < 50 and thickness > 2:
        return {
            "name": "露天开采",
            "description": "适用于浅层且较厚的煤层，成本低，回收率高。"
        }
    elif 50 <= depth < 300 and 1.5 <= thickness <= 8:
        return {
            "name": "长壁开采",
            "description": "适用于中等深度、厚度适中的煤层，产量高，安全性好。"
        }
    elif 100 <= depth < 600 and thickness > 6:
        return {
            "name": "分层开采",
            "description": "适用于中深度、较厚的煤层，可分批次开采，提高安全性。"
        }
    elif thickness < 1.5:
        return {
            "name": "窄煤柱开采",
            "description": "适用于薄煤层，可提高回收率，但成本较高。"
        }
    elif depth >= 600:
        return {
            "name": "水力开采",
            "description": "适用于深层煤层，通过高压水冲击煤层，安全性较高但成本高。"
        }
    else:
        return {
            "name": "房柱式开采",
            "description": "通用性强的开采方法，适应性好，但回收率较低。"
        }

    # 计算预期回收率


def calculate_recovery_rate(layer, method, base_rate=0.85):
    """计算预期回收率"""
    difficulty = layer["mining_difficulty"]["score"]

    # 基于开采方法调整回收率
    method_factor = {
        "露天开采": 1.2,
        "长壁开采": 1.1,
        "分层开采": 1.0,
        "窄煤柱开采": 0.9,
        "水力开采": 0.8,
        "房柱式开采": 0.85
    }.get(method["name"], 1.0)

    # 基于难度调整回收率
    difficulty_factor = max(0.7, 1 - difficulty * 0.03)

    # 计算最终回收率，但不超过95%
    recovery_rate = min(0.95, base_rate * method_factor * difficulty_factor)

    return recovery_rate


# 预测储量变化趋势
def predict_resource_trend(resource_history):
    """基于历史数据预测未来储量变化趋势"""
    if len(resource_history) < 2:
        return None

        # 设置中文字体
    set_chinese_font()

    # 提取时间点和储量数据
    dates = []
    resources = []

    for record in resource_history:
        date_obj = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
        days_since_start = (date_obj - datetime.strptime(resource_history[0]["timestamp"], "%Y-%m-%d %H:%M:%S")).days
        dates.append(days_since_start)
        resources.append(record["total_resources"])

        # 将日期转换为数字以便拟合
    dates = np.array(dates).reshape(-1, 1)
    resources = np.array(resources)

    # 线性回归预测
    model = LinearRegression()
    model.fit(dates, resources)

    # 预测未来180天的资源趋势
    future_days = np.array(list(range(0, 180, 30))).reshape(-1, 1)
    predicted_resources = model.predict(future_days)

    # 计算枯竭日期（如果有消耗趋势）
    depletion_date = None
    if model.coef_[0] < 0:  # 如果斜率为负（储量在减少）
        days_to_depletion = -resources[-1] / model.coef_[0]
        depletion_date = (datetime.strptime(resource_history[-1]["timestamp"], "%Y-%m-%d %H:%M:%S") +
                          timedelta(days=float(days_to_depletion))).strftime("%Y-%m-%d")

        # 生成趋势图
    plt.figure(figsize=(10, 6))
    plt.scatter(dates, resources, color='blue', label='历史数据')
    plt.plot(future_days, predicted_resources, color='red', linestyle='--', label='预测趋势')
    plt.title('煤炭资源储量变化趋势')
    plt.xlabel('时间（天）')
    plt.ylabel('储量（吨）')
    plt.legend()
    plt.grid(True)

    # 将图表转换为base64编码
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    trend_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return {
        "model_slope": float(model.coef_[0]),
        "model_intercept": float(model.intercept_),
        "predicted_values": [float(x) for x in predicted_resources.tolist()],
        "prediction_days": [int(x[0]) for x in future_days.tolist()],
        "trend_chart": trend_chart,
        "depletion_date": depletion_date
    }


# 基础路由：上传文件分析煤层
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 处理数据
            data = pd.read_excel(filepath) if filename.endswith(('xlsx', 'xls')) else pd.read_csv(filepath)

            # 检查必要的列
            required_columns = ['深度', '深侧向', '浅侧向', '声波时差', '自然伽玛', '密度']
            if not all(col in data.columns for col in required_columns):
                return jsonify({'error': f'文件中缺少必要的列。请确保文件包含以下列：{", ".join(required_columns)}'}), 400

                # 计算双侧向电阻率
            data['双侧向电阻率'] = 0.7 * data['深侧向'] + 0.3 * data['浅侧向']

            # 识别煤层
            coal_mask = classify_coal_layer(data)
            coal_data = data.loc[coal_mask, ['深度', '声波时差', '自然伽玛', '双侧向电阻率', '密度']]

            # 获取煤层深度范围
            depth_ranges = get_coal_depth_ranges(data, coal_mask)
            formatted_ranges = [{'start': float(start), 'end': float(end),
                                 'thickness': float(end - start)} for start, end in depth_ranges]

            # 计算总厚度
            total_thickness = sum(end - start for start, end in depth_ranges)

            # 准备返回数据
            indicators = ['深侧向', '浅侧向', '声波时差', '自然伽玛', '密度', '双侧向电阻率']
            chart_data = {
                'depth': data['深度'].tolist(),
                'indicators': {indicator: data[indicator].tolist() for indicator in indicators},
                'coal_layers': formatted_ranges,
                'total_thickness': float(total_thickness),
                'min_depth': float(data['深度'].min()),
                'max_depth': float(data['深度'].max())
            }

            # 缓存数据，用于后续请求
            data_cache[filename] = chart_data

            return jsonify(chart_data), 200

        except Exception as e:
            return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500

    return jsonify({'error': '不允许的文件类型'}), 400


# 新增路由：资源评估
@app.route('/resource-assessment', methods=['POST'])
def assess_resources():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400

    file = request.files['file']
    location = request.form.get('location', '未知位置')
    area = float(request.form.get('area', 10000))  # 默认10000平方米
    notes = request.form.get('notes', '')

    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 处理数据
            data = pd.read_excel(filepath) if filename.endswith(('xlsx', 'xls')) else pd.read_csv(filepath)

            # 检查必要的列
            required_columns = ['深度', '深侧向', '浅侧向', '声波时差', '自然伽玛', '密度']
            if not all(col in data.columns for col in required_columns):
                return jsonify({'error': f'文件中缺少必要的列。请确保文件包含以下列：{", ".join(required_columns)}'}), 400

                # 计算双侧向电阻率
            data['双侧向电阻率'] = 0.7 * data['深侧向'] + 0.3 * data['浅侧向']

            # 识别煤层
            coal_mask = classify_coal_layer(data)

            # 计算资源储量
            resource_data = calculate_coal_resources(data, coal_mask, area)

            # 优化开采规划
            mining_plan = optimize_mining_plan(resource_data["layers"])

            # 添加元数据
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            assessment_data = {
                'timestamp': timestamp,
                'location': location,
                'area': float(area),
                'notes': notes,
                'filename': filename,
                'total_resources': resource_data["total_resources"],
                'total_volume': resource_data["total_volume"],
                'layers_count': len(resource_data["layers"]),
                'mining_plan': mining_plan["mining_plan"],
                'priority_chart': mining_plan["priority_chart"]
            }

            # 保存历史数据
            resource_key = f"{location}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            resource_file = os.path.join(RESOURCE_FOLDER, f"{resource_key}.json")

            with open(resource_file, 'w') as f:
                # 需要创建一个可序列化的版本
                serializable_data = assessment_data.copy()
                serializable_data['layers'] = resource_data["layers"]
                json.dump(serializable_data, f)

                # 更新历史记录缓存
            if location not in extraction_history:
                extraction_history[location] = []

            extraction_history[location].append({
                'key': resource_key,
                'timestamp': timestamp,
                'total_resources': resource_data["total_resources"],
                'layers_count': len(resource_data["layers"])
            })

            # 预测资源趋势（如果有历史数据）
            trend_data = None
            if len(extraction_history[location]) >= 2:
                trend_data = predict_resource_trend(extraction_history[location])
                assessment_data['trend_data'] = trend_data

            return jsonify(assessment_data), 200

        except Exception as e:
            return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500

    return jsonify({'error': '不允许的文件类型'}), 400


# 获取资源评估历史记录
@app.route('/resource-history', methods=['GET'])
def get_resource_history():
    location = request.args.get('location', None)

    if location and location in extraction_history:
        return jsonify(extraction_history[location]), 200

        # 返回所有位置的最新记录
    latest_records = {}
    for loc, records in extraction_history.items():
        if records:
            latest_records[loc] = sorted(records, key=lambda x: x['timestamp'], reverse=True)[0]

    return jsonify(latest_records), 200


# 污染历史记录接口
@app.route('/pollution-history', methods=['GET'])
def get_pollution_history():
    location = request.args.get('location', None)

    if location and location in pollution_history:
        return jsonify(pollution_history[location]), 200

        # 返回所有位置的最新记录
    latest_records = {}
    for loc, records in pollution_history.items():
        if records:
            latest_records[loc] = sorted(records, key=lambda x: x['timestamp'], reverse=True)[0]

    return jsonify(latest_records), 200


# 获取特定历史记录详情
@app.route('/resource-history/<resource_key>', methods=['GET'])
def get_resource_detail(resource_key):
    resource_file = os.path.join(RESOURCE_FOLDER, f"{resource_key}.json")

    if os.path.exists(resource_file):
        with open(resource_file, 'r') as f:
            resource_data = json.load(f)
        return jsonify(resource_data), 200

    return jsonify({'error': '未找到历史记录'}), 404


# 根路由
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')


# 煤污染评估页面路由
@app.route('/pollution')
def serve_pollution_page():
    return send_from_directory('.', 'pollution.html')


# 添加资源管理页面路由
@app.route('/resource')
def serve_resource_page():
    return send_from_directory('.', 'resource.html')


# 添加农业利用相关功能
import random
from sklearn.cluster import KMeans
# 导入agriculture模块中的所有函数
from agriculture import (
    assess_soil_quality,
    generate_reclamation_plan,
    recommend_agriculture
)

# 存储农业评估历史的全局变量
agriculture_history = {}


# 新增路由：土壤农业评估
@app.route('/agriculture-assessment', methods=['POST'])
def agriculture_assessment():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400

    file = request.files['file']
    location = request.form.get('location', '未知位置')
    area = float(request.form.get('area', 10000))  # 默认10000平方米
    notes = request.form.get('notes', '')
    assessment_type = request.form.get('type', 'both')  # 'reclamation', 'agriculture', 'both'

    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 处理数据
            data = pd.read_excel(filepath) if filename.endswith(('xlsx', 'xls')) else pd.read_csv(filepath)

            # 检查必要的列
            required_columns = ['深度', '深侧向', '浅侧向', '声波时差', '自然伽玛', '密度']
            if not all(col in data.columns for col in required_columns):
                return jsonify({'error': f'文件中缺少必要的列。请确保文件包含以下列：{", ".join(required_columns)}'}), 400

                # 计算双侧向电阻率
            data['双侧向电阻率'] = 0.7 * data['深侧向'] + 0.3 * data['浅侧向']

            # 识别煤层
            coal_mask = classify_coal_layer(data)

            # 评估土壤质量
            soil_quality = assess_soil_quality(data, coal_mask)

            # 生成结果
            result = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'location': location,
                'area': float(area),
                'notes': notes,
                'filename': filename,
                'soil_quality': soil_quality
            }

            # 根据评估类型生成建议
            if assessment_type in ['reclamation', 'both']:
                reclamation_plan = generate_reclamation_plan(soil_quality)
                result['reclamation_plan'] = reclamation_plan

            if assessment_type in ['agriculture', 'both']:
                agriculture_recommendation = recommend_agriculture(soil_quality)
                result['agriculture_recommendation'] = agriculture_recommendation

                # 保存历史数据
            agriculture_key = f"{location}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            if location not in agriculture_history:
                agriculture_history[location] = []

            agriculture_history[location].append({
                'key': agriculture_key,
                'timestamp': result['timestamp'],
                'location': location,
                'soil_type': soil_quality['soil_type'],
                'fertility_score': soil_quality['fertility_score'],
                'pollution_level': soil_quality['pollution_level']['level']
            })

            # 保存详细结果
            agriculture_file = os.path.join(RESOURCE_FOLDER, f"agri_{agriculture_key}.json")
            with open(agriculture_file, 'w') as f:
                json.dump(result, f)

            return jsonify(result), 200

        except Exception as e:
            return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500

    return jsonify({'error': '不允许的文件类型'}), 400


# 获取农业评估历史记录
@app.route('/agriculture-history', methods=['GET'])
def get_agriculture_history():
    location = request.args.get('location', None)

    if location and location in agriculture_history:
        return jsonify(agriculture_history[location]), 200

        # 返回所有位置的最新记录
    latest_records = {}
    for loc, records in agriculture_history.items():
        if records:
            latest_records[loc] = sorted(records, key=lambda x: x['timestamp'], reverse=True)[0]

    return jsonify(latest_records), 200


# 获取特定农业评估记录详情
@app.route('/agriculture-history/<agriculture_key>', methods=['GET'])
def get_agriculture_detail(agriculture_key):
    agriculture_file = os.path.join(RESOURCE_FOLDER, f"agri_{agriculture_key}.json")

    if os.path.exists(agriculture_file):
        with open(agriculture_file, 'r') as f:
            agriculture_data = json.load(f)
        return jsonify(agriculture_data), 200

    return jsonify({'error': '未找到历史记录'}), 404


# 添加农业利用页面路由
@app.route('/agriculture')
def serve_agriculture_page():
    return send_from_directory('.', 'agriculture.html')
# 启动应用
# 煤污染评估相关函数

def assess_coal_pollution(data, coal_mask):
    """评估煤污染程度"""
    # 深度分段（每10米一段）
    depth_min = data['深度'].min()
    depth_max = data['深度'].max()
    segment_size = 10  # 10米一段

    segments = []
    for start in np.arange(depth_min, depth_max, segment_size):
        end = min(start + segment_size, depth_max)
        segment_data = data[(data['深度'] >= start) & (data['深度'] < end)]

        if len(segment_data) > 0:
            # 计算该段内煤层占比
            segment_coal = np.sum(coal_mask[segment_data.index])
            coal_percentage = segment_coal / len(segment_data)

            # 计算污染指数 - 基于煤层占比和物理特性
            if coal_percentage > 0:
                # 考虑物理参数对污染的影响
                avg_density = segment_data['密度'].mean()
                avg_gamma = segment_data['自然伽玛'].mean()

                # 煤密度越小、伽马值越高，污染扩散可能性越大
                pollution_factor = coal_percentage * (2.0 - avg_density) * (avg_gamma / 50)
                pollution_level = min(10, pollution_factor * 10)  # 0-10范围
            else:
                pollution_level = 0

            segments.append({
                'start_depth': float(start),
                'end_depth': float(end),
                'coal_percentage': float(coal_percentage),
                'pollution_level': float(pollution_level),
                'segment_size': float(segment_size)
            })

    # 计算总体污染评分（0-100）
    overall_score = min(100, sum(s['pollution_level'] for s in segments) * 2)

    # 污染等级
    if overall_score < 20:
        pollution_grade = '轻微'
    elif overall_score < 40:
        pollution_grade = '轻度'
    elif overall_score < 60:
        pollution_grade = '中度'
    elif overall_score < 80:
        pollution_grade = '严重'
    else:
        pollution_grade = '极严重'

    # 风险评估
    risks = []
    if overall_score > 30:
        risks.append('土壤结构可能受损')
    if overall_score > 50:
        risks.append('地下水污染风险增加')
    if overall_score > 70:
        risks.append('生态系统平衡受到威胁')
    if overall_score > 80:
        risks.append('可能对人体健康构成直接威胁')
    if overall_score > 90:
        risks.append('区域需要紧急隔离和治理')

    return {
        'segments': segments,
        'overall_score': float(overall_score),
        'pollution_grade': pollution_grade,
        'risks': risks
    }

# Flask应用路由

@app.route('/pollution-assessment', methods=['POST'])
def assess_pollution():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400

    file = request.files['file']
    location = request.form.get('location', '未知位置')
    notes = request.form.get('notes', '')

    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 处理数据
            data = pd.read_excel(filepath) if filename.endswith(('xlsx', 'xls')) else pd.read_csv(filepath)

            # 检查必要的列
            required_columns = ['深度', '深侧向', '浅侧向', '声波时差', '自然伽玛', '密度']
            if not all(col in data.columns for col in required_columns):
                return jsonify({'error': f'文件中缺少必要的列。请确保文件包含以下列：{", ".join(required_columns)}'}), 400

            # 计算双侧向电阻率
            data['双侧向电阻率'] = 0.7 * data['深侧向'] + 0.3 * data['浅侧向']

            # 识别煤层
            coal_mask = classify_coal_layer(data)

            # 评估煤污染
            pollution_assessment = assess_coal_pollution(data, coal_mask)

            # 添加元数据
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            assessment_data = {
                'timestamp': timestamp,
                'location': location,
                'notes': notes,
                'filename': filename,
                'assessment': pollution_assessment
            }

            # 保存历史数据
            history_key = f"{location}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            history_file = os.path.join(HISTORY_FOLDER, f"{history_key}.json")
            with open(history_file, 'w') as f:
                json.dump(assessment_data, f)

            # 更新历史记录缓存
            if location not in pollution_history:
                pollution_history[location] = []
            pollution_history[location].append({
                'key': history_key,
                'timestamp': timestamp,
                'overall_score': pollution_assessment['overall_score'],
                'pollution_grade': pollution_assessment['pollution_grade']
            })

            return jsonify(assessment_data), 200

        except Exception as e:
            return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500

    return jsonify({'error': '不允许的文件类型'}), 400


@app.route('/pollution-history/<history_key>', methods=['GET'])
def get_history_detail(history_key):
    history_file = os.path.join(HISTORY_FOLDER, f"{history_key}.json")

    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        return jsonify(history_data), 200

    return jsonify({'error': '未找到历史记录'}), 404

# 煤污染评估页面路由


if __name__ == '__main__':
    app.run(debug=True, port=5000)
