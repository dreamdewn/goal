# agriculture.py - 农业利用功能模块

import random
import numpy as np
from sklearn.cluster import KMeans


def classify_soil_type(soil_data):
    """根据物理特性将土壤分类"""
    avg_density = soil_data['密度'].mean()
    avg_gamma = soil_data['自然伽玛'].mean()

    # 基于密度和伽马值的简单分类
    if avg_density < 1.3:
        if avg_gamma < 50:
            return "腐殖质壤土"
        else:
            return "粘土"
    elif avg_density < 1.5:
        if avg_gamma < 60:
            return "砂质壤土"
        else:
            return "粘壤土"
    else:
        if avg_gamma < 40:
            return "砂土"
        elif avg_gamma < 60:
            return "石质土"
        else:
            return "粘土"


def assess_soil_pollution(soil_data, coal_mask):
    """评估土壤污染程度"""
    avg_gamma = soil_data['自然伽玛'].mean()

    # 基于伽马值的简单污染评估
    if avg_gamma < 45:
        pollution = {
            "level": "轻微",
            "score": round(float(max(0, avg_gamma - 30) / 15 * 100), 1),
            "description": "土壤污染轻微，适合多种农作物种植"
        }
    elif avg_gamma < 60:
        pollution = {
            "level": "中度",
            "score": round(float((avg_gamma - 45) / 15 * 100 + 33), 1),
            "description": "土壤存在中度污染，需选择耐污染农作物或进行土壤改良"
        }
    else:
        pollution = {
            "level": "严重",
            "score": round(float(min(100, (avg_gamma - 60) / 20 * 100 + 66)), 1),
            "description": "土壤污染严重，建议进行专业修复处理后再利用"
        }

    return pollution


def assess_soil_quality(data, coal_mask):
    """评估土壤质量，分析煤含量和其他指标"""
    # 获取非煤层数据，作为土壤层
    soil_data = data[~coal_mask].copy()

    if soil_data.empty:
        return {"error": "没有足够的土壤数据"}

        # 计算关键指标
    avg_density = soil_data['密度'].mean()
    avg_gamma = soil_data['自然伽玛'].mean()
    resistivity = soil_data['双侧向电阻率'].mean()

    # 估算土壤pH值 (基于伽马值和电阻率的模拟)
    estimated_ph = 7.0 - (avg_gamma - 50) / 20
    estimated_ph = max(4.0, min(9.0, estimated_ph))

    # 估算有机质含量 (基于密度的模拟)
    organic_matter = max(0, 5.0 - (avg_density - 1.2) * 10)

    # 估算煤含量 (基于伽马值的模拟)
    coal_content = max(0, min(30, (avg_gamma - 40) / 2))

    # 估算水分含量 (基于电阻率的模拟)
    moisture = max(5, min(40, 1000 / resistivity))

    # 估算土壤肥力 (基于有机质和pH的综合评分)
    fertility_score = 0
    if 6.0 <= estimated_ph <= 7.5:
        ph_factor = 1.0
    else:
        ph_factor = 1.0 - abs(estimated_ph - 6.75) / 3

    fertility_score = (organic_matter * 0.6 + ph_factor * 40) * (1 - coal_content / 100)
    fertility_score = max(0, min(100, fertility_score))

    # 土壤类型分类
    soil_type = classify_soil_type(soil_data)

    # 土壤污染评估
    pollution_level = assess_soil_pollution(soil_data, coal_mask)

    return {
        "ph_value": round(float(estimated_ph), 1),
        "organic_matter": round(float(organic_matter), 1),
        "coal_content": round(float(coal_content), 1),
        "moisture": round(float(moisture), 1),
        "fertility_score": round(float(fertility_score), 1),
        "soil_type": soil_type,
        "pollution_level": pollution_level,
        "density": round(float(avg_density), 2),
        "gamma": round(float(avg_gamma), 2),
        "resistivity": round(float(resistivity), 2)
    }


def generate_reclamation_plan(soil_quality):
    """根据土壤质量生成土地复垦方案"""
    soil_type = soil_quality["soil_type"]
    ph_value = soil_quality["ph_value"]
    fertility = soil_quality["fertility_score"]
    coal_content = soil_quality["coal_content"]
    pollution_level = soil_quality["pollution_level"]["level"]

    # 根据污染程度确定复垦策略
    if pollution_level == "严重":
        reclamation_strategy = {
            "name": "深度修复重建",
            "description": "需要进行深层土壤置换和污染物固化处理，引入特定植物进行植物修复",
            "duration": "5-8年",
            "cost_level": "高"
        }
    elif pollution_level == "中度":
        reclamation_strategy = {
            "name": "中度改良修复",
            "description": "添加有机质和改良剂，结合抗性植物种植，逐步改善土壤质量",
            "duration": "3-5年",
            "cost_level": "中"
        }
    else:
        reclamation_strategy = {
            "name": "轻度调理修复",
            "description": "适当施加有机肥和微量元素肥料，恢复土壤生态系统",
            "duration": "1-3年",
            "cost_level": "低"
        }

        # 推荐植被方案
    vegetation_plan = recommend_vegetation(soil_type, ph_value, fertility, coal_content, pollution_level)

    # 建议措施
    measures = generate_reclamation_measures(soil_quality)

    return {
        "strategy": reclamation_strategy,
        "vegetation_plan": vegetation_plan,
        "measures": measures
    }


def recommend_vegetation(soil_type, ph_value, fertility, coal_content, pollution_level):
    """推荐适合的植被方案"""
    # 第一阶段植物 (先锋植物，用于改善土壤条件)
    pioneer_plants = []

    if pollution_level == "严重":
        pioneer_plants = [
            {"name": "芦苇", "type": "禾本科", "function": "超积累植物，可吸收土壤重金属"},
            {"name": "向日葵", "type": "菊科", "function": "根系发达，可富集多种污染物"},
            {"name": "紫花苜蓿", "type": "豆科", "function": "固氮能力强，改善土壤肥力"}
        ]
    elif pollution_level == "中度":
        pioneer_plants = [
            {"name": "狗牙根", "type": "禾本科", "function": "耐盐碱，覆盖速度快"},
            {"name": "田菁", "type": "豆科", "function": "生长快，固氮能力强"},
            {"name": "苏丹草", "type": "禾本科", "function": "生物量大，有机质还田效果好"}
        ]
    else:
        pioneer_plants = [
            {"name": "黑麦草", "type": "禾本科", "function": "建植速度快，保持土壤稳定"},
            {"name": "白三叶", "type": "豆科", "function": "固氮改良土壤"},
            {"name": "红豆草", "type": "豆科", "function": "适应性强，改善土壤结构"}
        ]

        # 第二阶段植物 (经济植物，可持续利用)
    economic_plants = []

    # 根据土壤类型和肥力选择经济植物
    if fertility > 60:
        if ph_value < 6.5:
            economic_plants = [
                {"name": "蓝莓", "type": "灌木", "function": "经济价值高，适合酸性土壤"},
                {"name": "茶树", "type": "灌木", "function": "深根系，经济效益稳定"},
                {"name": "松树", "type": "乔木", "function": "适应性强，可开发林下经济"}
            ]
        else:
            economic_plants = [
                {"name": "核桃", "type": "乔木", "function": "经济价值高，根系发达"},
                {"name": "苹果", "type": "乔木", "function": "经济效益好，改善生态环境"},
                {"name": "葡萄", "type": "藤本", "function": "适应性广，增加土地利用率"}
            ]
    else:
        if soil_type in ["砂土", "砂质壤土"]:
            economic_plants = [
                {"name": "沙棘", "type": "灌木", "function": "耐瘠薄，果实有经济价值"},
                {"name": "桉树", "type": "乔木", "function": "生长快，材质好"},
                {"name": "柠条", "type": "灌木", "function": "耐干旱瘠薄，可饲用"}
            ]
        else:
            economic_plants = [
                {"name": "刺槐", "type": "乔木", "function": "固氮能力强，蜜源植物"},
                {"name": "构树", "type": "乔木", "function": "适应性强，叶可饲养家蚕"},
                {"name": "杨树", "type": "乔木", "function": "生长快，用途广"}
            ]

    vegetation_types = {
        "pioneer_stage": {
            "duration": "1-2年",
            "plants": pioneer_plants
        },
        "economic_stage": {
            "duration": "长期",
            "plants": economic_plants
        }
    }

    return vegetation_types


def generate_reclamation_measures(soil_quality):
    """生成具体复垦措施"""
    measures = []

    # 土壤改良措施
    if soil_quality["ph_value"] < 5.5:
        measures.append({
            "type": "土壤改良",
            "name": "石灰调节",
            "description": "施用石灰或石灰石粉调节土壤酸度",
            "amount": f"{(6.5 - soil_quality['ph_value']) * 1000} kg/hm²"
        })
    elif soil_quality["ph_value"] > 8.0:
        measures.append({
            "type": "土壤改良",
            "name": "硫磺处理",
            "description": "施用硫磺或硫酸亚铁降低土壤pH值",
            "amount": f"{(soil_quality['ph_value'] - 7.5) * 800} kg/hm²"
        })

        # 有机质补充
    if soil_quality["organic_matter"] < 3.0:
        measures.append({
            "type": "土壤改良",
            "name": "有机质添加",
            "description": "施用腐熟有机肥或堆肥增加土壤有机质",
            "amount": f"{(3.0 - soil_quality['organic_matter']) * 10000} kg/hm²"
        })

        # 土壤重构
    if soil_quality["coal_content"] > 10:
        measures.append({
            "type": "土壤重构",
            "name": "表层置换",
            "description": "移除高煤含量表层土壤，引入优质客土",
            "amount": "20-30 cm厚度"
        })

        # 排水系统
    if soil_quality["moisture"] > 25:
        measures.append({
            "type": "工程措施",
            "name": "排水系统",
            "description": "建设明沟暗管排水系统，改善土壤通气性",
            "amount": "30-50 m/hm²"
        })

        # 肥料施用
    if soil_quality["fertility_score"] < 50:
        measures.append({
            "type": "养分补充",
            "name": "平衡施肥",
            "description": "施用NPK复合肥提高土壤养分水平",
            "amount": f"{(50 - soil_quality['fertility_score']) * 10} kg/hm²"
        })

    return measures


def recommend_agriculture(soil_quality):
    """根据土壤质量推荐适合种植的农作物和管理措施"""
    soil_type = soil_quality["soil_type"]
    ph_value = soil_quality["ph_value"]
    fertility = soil_quality["fertility_score"]
    organic_matter = soil_quality["organic_matter"]
    moisture = soil_quality["moisture"]
    pollution_level = soil_quality["pollution_level"]["level"]

    suitable_crops = []
    unsuitable_crops = []
    management_tips = []

    # 根据污染程度筛选
    if pollution_level == "严重":
        management_tips.append("不建议种植食用农作物，可考虑非食用经济作物或观赏植物")
        suitable_crops = recommend_non_food_crops(soil_type, ph_value, fertility)
        unsuitable_crops = [
            {"name": "叶菜类", "reason": "易吸收土壤污染物，不适合食用"},
            {"name": "根茎类", "reason": "直接接触污染土壤，存在安全风险"},
            {"name": "水果类", "reason": "长期种植易累积污染物"}
        ]
    else:
        # 根据土壤特性推荐作物
        suitable_crops = recommend_food_crops(soil_type, ph_value, fertility, pollution_level)
        unsuitable_crops = get_unsuitable_crops(soil_type, ph_value, fertility)

        # 管理建议
    if organic_matter < 2.0:
        management_tips.append("土壤有机质含量低，建议增施有机肥，实施秸秆还田")

    if ph_value < 5.5:
        management_tips.append("土壤偏酸，建议施用石灰等碱性物质调节pH值")
    elif ph_value > 8.0:
        management_tips.append("土壤偏碱，建议施用硫磺或硫酸亚铁等酸性物质降低pH值")

    if moisture > 30:
        management_tips.append("土壤水分含量高，注意排水，选择耐湿作物")
    elif moisture < 15:
        management_tips.append("土壤水分偏低，建议完善灌溉设施，实施节水灌溉")

    if pollution_level == "中度":
        management_tips.append("土壤存在一定污染，建议种植前进行土壤改良，选择抗性强的作物")

        # 施肥建议
    fertilizer_recommendation = recommend_fertilizer(soil_quality)

    # 灌溉建议
    irrigation_recommendation = recommend_irrigation(soil_quality)

    return {
        "suitable_crops": suitable_crops,
        "unsuitable_crops": unsuitable_crops,
        "management_tips": management_tips,
        "fertilizer_recommendation": fertilizer_recommendation,
        "irrigation_recommendation": irrigation_recommendation
    }


def recommend_food_crops(soil_type, ph_value, fertility, pollution_level):
    """推荐适合的食用农作物"""
    suitable_crops = []

    # 基于土壤类型的基础作物组
    base_crops = []
    if soil_type in ["砂土", "砂质壤土"]:
        base_crops = ["红薯", "花生", "胡萝卜", "甘薯", "西瓜"]
    elif soil_type in ["腐殖质壤土", "壤土"]:
        base_crops = ["玉米", "小麦", "大豆", "马铃薯", "甜菜"]
    elif soil_type in ["粘土", "粘壤土"]:
        base_crops = ["水稻", "小麦", "油菜", "棉花", "大豆"]
    else:
        base_crops = ["高粱", "谷子", "燕麦", "荞麦"]

        # 基于pH值筛选
    ph_suitable = []
    if ph_value < 6.0:
        ph_suitable = ["蓝莓", "马铃薯", "燕麦", "茶叶", "甘薯"]
    elif ph_value < 7.0:
        ph_suitable = ["水稻", "小麦", "玉米", "大豆", "花生"]
    else:
        ph_suitable = ["大麦", "甜菜", "芦笋", "菠菜", "卷心菜"]

        # 污染调整
    pollution_adjust = 1.0
    if pollution_level == "中度":
        pollution_adjust = 0.7

        # 计算综合得分并选择最适合的作物
    all_crops = list(set(base_crops + ph_suitable))
    crop_scores = []

    for crop in all_crops:
        # 基础分
        base_score = 60

        # 土壤类型适应性
        if crop in base_crops:
            base_score += 20

            # pH适应性
        if crop in ph_suitable:
            base_score += 15

            # 肥力要求调整
        fertility_adjust = 0
        if crop in ["水稻", "玉米", "小麦", "油菜"]:
            fertility_adjust = (fertility - 50) * 0.2
        elif crop in ["高粱", "谷子", "燕麦", "荞麦"]:
            fertility_adjust = (40 - fertility) * 0.2

            # 最终分数
        final_score = (base_score + fertility_adjust) * pollution_adjust

        crop_scores.append((crop, final_score))

        # 排序并选择得分高的作物
    crop_scores.sort(key=lambda x: x[1], reverse=True)
    top_crops = crop_scores[:5]

    # 构建返回信息
    for crop, score in top_crops:
        crop_info = {"name": crop, "score": round(score, 1)}

        # 添加种植推荐理由
        if crop in base_crops and crop in ph_suitable:
            crop_info["reason"] = f"非常适合此类土壤和pH条件"
        elif crop in base_crops:
            crop_info["reason"] = f"适合此类{soil_type}土壤结构"
        elif crop in ph_suitable:
            crop_info["reason"] = f"适合此pH值(pH {ph_value})的土壤条件"
        else:
            crop_info["reason"] = "综合评分较高，可考虑种植"

        suitable_crops.append(crop_info)

    return suitable_crops


def recommend_non_food_crops(soil_type, ph_value, fertility):
    """推荐非食用经济作物或景观植物"""
    non_food_crops = [
        {"name": "向日葵", "type": "油料/观赏", "feature": "有一定的污染物吸收能力，适应性强"},
        {"name": "杨树", "type": "林木", "feature": "速生林木，有较强的环境适应性"},
        {"name": "柳树", "type": "林木", "feature": "较强的修复能力，适合湿地地区"},
        {"name": "蜀葵", "type": "观赏", "feature": "观赏价值高，适应性较强"},
        {"name": "芒草", "type": "能源", "feature": "生物质能源作物，适应性强"},
        {"name": "灯心草", "type": "工业原料", "feature": "适合湿地种植，可用于编织"},
        {"name": "薰衣草", "type": "药用/观赏", "feature": "耐旱性强，经济价值高"},
        {"name": "金盏花", "type": "观赏/药用", "feature": "有一定的土壤修复能力"}
    ]

    # 基于土壤条件筛选最适合的非食用作物
    suitable_non_food = []
    for crop in non_food_crops:
        suitability_score = 70  # 基础适宜度分数

        # 根据不同作物特性调整分数
        if crop["name"] in ["向日葵", "芒草"]:
            suitability_score += (100 - fertility) * 0.2  # 这些作物在贫瘠土壤中也能表现良好

        if crop["name"] in ["柳树", "灯心草"]:
            if soil_type in ["腐殖质壤土", "粘土", "粘壤土"]:
                suitability_score += 15  # 这些作物在保水性好的土壤中表现更佳

        if crop["name"] in ["薰衣草", "金盏花"]:
            if 6.0 <= ph_value <= 7.5:
                suitability_score += 15  # 这些作物偏好中性土壤

        suitable_non_food.append({
            "name": crop["name"],
            "type": crop["type"],
            "score": round(suitability_score, 1),
            "reason": crop["feature"]
        })

        # 排序并返回得分最高的5种
    suitable_non_food.sort(key=lambda x: x["score"], reverse=True)
    return suitable_non_food[:5]


def get_unsuitable_crops(soil_type, ph_value, fertility):
    """获取不适合种植的作物"""
    unsuitable_crops = []

    if soil_type in ["砂土", "砂质壤土"]:
        unsuitable_crops.append({"name": "水稻", "reason": "保水性差，不适合种植需水量大的作物"})

    if soil_type in ["粘土"]:
        unsuitable_crops.append({"name": "红薯", "reason": "土壤黏重，不利于根系膨大"})
        unsuitable_crops.append({"name": "胡萝卜", "reason": "土壤过于黏重，根系发育受限"})

    if ph_value < 5.5:
        unsuitable_crops.append({"name": "甜菜", "reason": "不耐酸性土壤"})
        unsuitable_crops.append({"name": "芦笋", "reason": "偏酸土壤不适合生长"})

    if ph_value > 7.5:
        unsuitable_crops.append({"name": "蓝莓", "reason": "需要酸性土壤环境"})
        unsuitable_crops.append({"name": "茶叶", "reason": "碱性土壤不利于生长"})

    if fertility < 40:
        unsuitable_crops.append({"name": "玉米", "reason": "需要较高肥力土壤支持生长"})

        # 确保至少有3种不适合的作物
    common_unsuitable = [
        {"name": "葡萄", "reason": "需特定的土壤环境和管理条件"},
        {"name": "草莓", "reason": "需疏松、肥沃且排水良好的土壤"},
        {"name": "西红柿", "reason": "对土壤养分和病虫害管理要求高"}
    ]

    for crop in common_unsuitable:
        if len(unsuitable_crops) < 3:
            if not any(c["name"] == crop["name"] for c in unsuitable_crops):
                unsuitable_crops.append(crop)

    return unsuitable_crops


def recommend_fertilizer(soil_quality):
    """推荐肥料施用方案"""
    fertility = soil_quality["fertility_score"]
    organic_matter = soil_quality["organic_matter"]
    ph_value = soil_quality["ph_value"]

    fertilizer_plan = {
        "base_fertilizer": [],
        "top_dressing": [],
        "special_treatment": []
    }

    # 基肥推荐
    if organic_matter < 2.0:
        fertilizer_plan["base_fertilizer"].append({
            "name": "腐熟有机肥",
            "amount": "3000-4000 kg/hm²",
            "timing": "整地前施入",
            "purpose": "提高土壤有机质含量，改善土壤结构"
        })
    else:
        fertilizer_plan["base_fertilizer"].append({
            "name": "腐熟有机肥",
            "amount": "1500-2000 kg/hm²",
            "timing": "整地前施入",
            "purpose": "维持土壤有机质水平，促进土壤微生物活性"
        })

        # 根据肥力水平推荐复合肥
    if fertility < 40:
        fertilizer_plan["base_fertilizer"].append({
            "name": "高氮复合肥(N-P-K = 15-5-5)",
            "amount": "600-750 kg/hm²",
            "timing": "播种前施入",
            "purpose": "提供作物生长所需的基础养分，促进茎叶生长"
        })
    elif fertility < 60:
        fertilizer_plan["base_fertilizer"].append({
            "name": "平衡复合肥(N-P-K = 15-15-15)",
            "amount": "450-600 kg/hm²",
            "timing": "播种前施入",
            "purpose": "均衡提供作物生长所需的各类养分"
        })
    else:
        fertilizer_plan["base_fertilizer"].append({
            "name": "低氮复合肥(N-P-K = 5-15-15)",
            "amount": "300-450 kg/hm²",
            "timing": "播种前施入",
            "purpose": "维持基础养分水平，避免氮素过量"
        })

        # 追肥推荐
    fertilizer_plan["top_dressing"].append({
        "name": "尿素",
        "amount": "150-225 kg/hm²",
        "timing": "作物生长中期",
        "purpose": "补充氮素，促进植株生长"
    })

    # 特殊处理
    if ph_value < 5.5:
        fertilizer_plan["special_treatment"].append({
            "name": "石灰",
            "amount": f"{(6.5 - ph_value) * 1000} kg/hm²",
            "timing": "整地前施入",
            "purpose": "调节土壤酸度，改善养分有效性"
        })
    elif ph_value > 7.5:
        fertilizer_plan["special_treatment"].append({
            "name": "硫磺粉",
            "amount": f"{(ph_value - 7.0) * 200} kg/hm²",
            "timing": "整地前施入",
            "purpose": "降低土壤pH值，提高养分有效性"
        })

    if soil_quality["coal_content"] > 5:
        fertilizer_plan["special_treatment"].append({
            "name": "腐植酸肥料",
            "amount": "300-450 kg/hm²",
            "timing": "整地时施入",
            "purpose": "增强土壤团粒结构，减轻煤炭残留影响"
        })

    return fertilizer_plan


def recommend_irrigation(soil_quality):
    """推荐灌溉方案"""
    soil_type = soil_quality["soil_type"]
    moisture = soil_quality["moisture"]

    # 灌溉系统推荐
    if soil_type in ["砂土", "砂质壤土"]:
        irrigation_system = {
            "type": "滴灌系统",
            "advantage": "节水效果好，减少养分淋溶，适合砂质土壤",
            "cost_level": "中高"
        }
    elif soil_type in ["腐殖质壤土", "壤土"]:
        irrigation_system = {
            "type": "微喷灌系统",
            "advantage": "灌溉均匀，可调控湿度，适合多种作物",
            "cost_level": "中"
        }
    else:
        irrigation_system = {
            "type": "沟灌系统（改良型）",
            "advantage": "投资低，操作简单，适合粘性土壤",
            "cost_level": "低"
        }

        # 灌溉策略
    if moisture < 15:
        irrigation_strategy = {
            "frequency": "高频小量",
            "interval": "3-5天一次",
            "amount": "20-30 mm/次"
        }
    elif moisture < 25:
        irrigation_strategy = {
            "frequency": "适中",
            "interval": "7-10天一次",
            "amount": "30-45 mm/次"
        }
    else:
        irrigation_strategy = {
            "frequency": "低频大量",
            "interval": "10-15天一次",
            "amount": "45-60 mm/次"
        }

        # 节水技术
    water_saving_techniques = []

    water_saving_techniques.append({
        "name": "地表覆盖",
        "description": "使用秸秆、地膜等覆盖地表，减少水分蒸发",
        "effectiveness": "可减少蒸发损失20-30%"
    })

    if soil_type in ["砂土", "砂质壤土"]:
        water_saving_techniques.append({
            "name": "土壤改良",
            "description": "添加有机质增强土壤保水能力",
            "effectiveness": "可提高持水能力15-25%"
        })

    water_saving_techniques.append({
        "name": "智能灌溉控制",
        "description": "基于土壤墒情监测自动控制灌溉",
        "effectiveness": "可节水15-30%，提高灌溉效率"
    })

    return {
        "irrigation_system": irrigation_system,
        "irrigation_strategy": irrigation_strategy,
        "water_saving_techniques": water_saving_techniques
    }