from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import json
import logging
from typing import Dict, List, Optional
import time
from datetime import datetime
import os
import sys

# إضافة المسار الحالي للنظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_meal_planner import EnhancedMealPlanner
from data_enhancer import DataEnhancer, UserProfile
from performance_optimizer import PerformanceOptimizer
data_enhancer = DataEnhancer()

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'enhanced_meal_planner_secret_key_2024'

# إعداد CORS
CORS(app)

# إعداد الترميز للنص العربي
app.config['JSON_AS_ASCII'] = False

class MealPlannerAPI:
    
    def __init__(self):
        """تهيئة API"""
        self.model = None
        self.data_enhancer = None
        self.performance_optimizer = None
        self.is_loaded = False
        
        # إحصائيات الاستخدام
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "last_request_time": None
        }
        
        logger.info(" تم تهيئة Meal Planner API")
    
    def load_model(self, model_path: str = None):
        try:
            logger.info(" جاري تحميل النموذج المحسن...")
            
            # إنشاء محسن البيانات أولاً (لا يحتاج torch)
            self.data_enhancer = DataEnhancer()
            logger.info(" تم إنشاء محسن البيانات")
            
            # محاولة إنشاء النموذج (قد يفشل إذا لم يكن torch مثبت)
            try:
                self.model = EnhancedMealPlanner()
                self.model.load_tokenizer_and_config()
                self.model.create_enhanced_model()
                
                # تحميل النموذج المدرب إذا كان متوفراً
                if model_path and os.path.exists(model_path):
                    self.model.model = torch.load(model_path, map_location=self.model.device)
                    logger.info(f" تم تحميل النموذج المدرب من {model_path}")
                else:
                    logger.info("ℹ استخدام النموذج الجديد (غير مدرب)")
                
                # إنشاء محسن الأداء
                self.performance_optimizer = PerformanceOptimizer(
                    self.model.model, 
                    device=str(self.model.device)
                )
                self.performance_optimizer.setup_mixed_precision(enabled=True)
                self.performance_optimizer.optimize_model_for_inference()
                
                logger.info(" تم تحميل النموذج الكامل بنجاح!")
                
            except ImportError as e:
                logger.warning(f" PyTorch غير مثبت، استخدام الوضع البسيط: {e}")
                self.model = None
                self.performance_optimizer = None
            
            self.is_loaded = True
            logger.info(" تم تحميل النظام بنجاح!")
            
        except Exception as e:
            logger.error(f" خطأ في تحميل النموذج: {str(e)}")
            self.is_loaded = False
    
    def generate_meal_plan(self, user_info: str = None, goal: str = "", 
                          days: int = 7, detailed: bool = True,
                          age: int = None, gender: str = None,
                          weight: float = None, height: float = None,
                          activity_level: str = None,
                          dietary_restrictions: List[str] = None,
                          goals: List[str] = None,
                          medical_conditions: List[str] = None) -> Dict:
        start_time = time.time()
        
        try:
            # تحسين ملف المستخدم - استخدام البيانات المباشرة إذا كانت متوفرة
            if age is not None and weight is not None and height is not None:
                # استخدام البيانات الممررة مباشرة
                profile = self.data_enhancer.enhance_user_profile(
                    basic_info=None,
                    age=age,
                    gender=gender,
                    weight=weight,
                    height=height,
                    activity_level=activity_level,
                    dietary_restrictions=dietary_restrictions,
                    goals=goals,
                    medical_conditions=medical_conditions
                )
                logger.info(f"استخدام البيانات المباشرة: العمر={age}, الوزن={weight}, الطول={height}")
            else:
                # استخراج من النص
                profile = self.data_enhancer.enhance_user_profile(user_info or "")
                logger.info(f"استخراج البيانات من النص: {user_info}")
            
            # توليد خطة الوجبات المحسنة
            meal_plan = None
            if detailed:
                try:
                    meal_plan = self.data_enhancer.generate_enhanced_meal_plan(profile, days)
                except Exception as e:
                    logger.warning(f"فشل في توليد خطة مفصلة، استخدام الوضع البسيط: {e}")
                    detailed = False
            
            if not detailed or meal_plan is None:
                # استخدام النموذج للجيل السريع
                try:
                    if self.model and hasattr(self.model, 'generate_meal_plan'):
                        meal_plan_text = self.model.generate_meal_plan(user_info, goal)
                    else:
                        meal_plan_text = f"خطة وجبات مخصصة لـ {user_info} مع الهدف: {goal}"
                    
                    meal_plan = {
                        "user_profile": profile.__dict__,
                        "generated_plan": meal_plan_text,
                        "days": days,
                        "simple_mode": True
                    }
                except Exception as e:
                    logger.warning(f"فشل في توليد خطة بسيطة: {e}")
                    meal_plan = {
                        "user_profile": profile.__dict__,
                        "generated_plan": f"خطة وجبات أساسية لـ {user_info}",
                        "days": days,
                        "simple_mode": True,
                        "note": "تم إنشاء خطة أساسية بسبب مشاكل في النموذج"
                    }
            
            # إضافة معلومات إضافية
            if meal_plan:
                meal_plan["generation_info"] = {
                    "timestamp": datetime.now().isoformat(),
                    "model_version": "enhanced_v1.0",
                    "processing_time": time.time() - start_time,
                    "detailed_mode": detailed
                }
            
            return {
                "success": True,
                "meal_plan": meal_plan,
                "user_profile": profile.__dict__
            }
            
        except Exception as e:
            logger.error(f" خطأ في توليد خطة الوجبات: {str(e)}")
            return {
                "success": False,
                "error": f"خطأ في توليد خطة الوجبات: {str(e)}"
            }
    
    def batch_generate(self, requests: List[Dict]) -> List[Dict]:
        results = []
        
        for req in requests:
            user_info = req.get("user_info", "")
            goal = req.get("goal", "")
            days = req.get("days", 7)
            detailed = req.get("detailed", True)
            
            result = self.generate_meal_plan(user_info, goal, days, detailed)
            results.append(result)
        
        return results
    
    def get_nutritional_info(self, food_item: str) -> Dict:
        try:
            nutritional_db = self.data_enhancer.nutritional_database
            
            # البحث في قاعدة البيانات
            for category, foods in nutritional_db.items():
                if food_item.lower() in [food.lower() for food in foods.keys()]:
                    return {
                        "success": True,
                        "food_item": food_item,
                        "category": category,
                        "nutrition": foods[food_item.lower()]
                    }
            
            return {
                "success": False,
                "error": f"لم يتم العثور على {food_item} في قاعدة البيانات"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"خطأ في البحث: {str(e)}"
            }
    
    def get_seasonal_foods(self, season: str = None) -> Dict:
        try:
            if season is None:
                season = self.data_enhancer._get_current_season()
            
            seasonal_foods = self.data_enhancer.seasonal_foods.get(season, [])
            
            return {
                "success": True,
                "season": season,
                "foods": seasonal_foods
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"خطأ في الحصول على الأطعمة الموسمية: {str(e)}"
            }
    
    def update_usage_stats(self, success: bool, response_time: float):
        self.usage_stats["total_requests"] += 1
        self.usage_stats["last_request_time"] = datetime.now().isoformat()
        
        if success:
            self.usage_stats["successful_requests"] += 1
        else:
            self.usage_stats["failed_requests"] += 1
        
        # حساب متوسط وقت الاستجابة
        total_successful = self.usage_stats["successful_requests"]
        if total_successful > 0:
            current_avg = self.usage_stats["average_response_time"]
            new_avg = ((current_avg * (total_successful - 1)) + response_time) / total_successful
            self.usage_stats["average_response_time"] = new_avg

# إنشاء مثيل API
api = MealPlannerAPI()

@app.route('/')
def index():
    html_template = """
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>مترجم لغة الإشارة المحسن</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .api-info {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            .endpoint {
                background: #e9ecef;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #007bff;
            }
            .method {
                font-weight: bold;
                color: #007bff;
            }
            .status {
                padding: 10px;
                border-radius: 5px;
                margin: 20px 0;
                text-align: center;
                font-weight: bold;
            }
            .status.loaded {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.not-loaded {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> مترجم لغة الإشارة المحسن</h1>
            
            <div class="status {{ 'loaded' if api.is_loaded else 'not-loaded' }}">
                {{ ' النموذج محمل وجاهز للاستخدام' if api.is_loaded else '❌ النموذج غير محمل' }}
            </div>
            
            <div class="api-info">
                <h2> واجهات API المتاحة:</h2>
                
                <div class="endpoint">
                    <span class="method">POST</span> /api/generate_meal_plan
                    <br>توليد خطة وجبات مخصصة
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> /api/batch_generate
                    <br>توليد دفعات من خطط الوجبات
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> /api/nutritional_info
                    <br>الحصول على معلومات غذائية
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> /api/seasonal_foods
                    <br>الحصول على الأطعمة الموسمية
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> /api/status
                    <br>حالة النظام وإحصائيات الاستخدام
                </div>
            </div>
            
            <div class="api-info">
                <h2> إحصائيات الاستخدام:</h2>
                <p><strong>إجمالي الطلبات:</strong> {{ api.usage_stats.total_requests }}</p>
                <p><strong>الطلبات الناجحة:</strong> {{ api.usage_stats.successful_requests }}</p>
                <p><strong>الطلبات الفاشلة:</strong> {{ api.usage_stats.failed_requests }}</p>
                <p><strong>متوسط وقت الاستجابة:</strong> {{ "%.2f"|format(api.usage_stats.average_response_time) }} ثانية</p>
                <p><strong>آخر طلب:</strong> {{ api.usage_stats.last_request_time or 'لا يوجد' }}</p>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, api=api)

@app.route('/api/generate_meal_plan', methods=['POST'])
def generate_meal_plan():
    start_time = time.time()
    
    try:
        if not api.is_loaded:
            # محاولة تحميل النموذج تلقائياً
            try:
                api.load_model()
                if not api.is_loaded:
                    return jsonify({
                        "success": False,
                        "error": "فشل في تحميل النموذج. يرجى التحقق من المتطلبات."
                    }), 503
            except Exception as load_error:
                return jsonify({
                    "success": False,
                    "error": f"خطأ في تحميل النموذج: {str(load_error)}"
                }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "بيانات غير صحيحة"
            }), 400
        
        user_info = data.get('user_info', '')
        goal = data.get('goal', '')
        days = data.get('days', 7)
        detailed = data.get('detailed', True)
        
        # محاولة الحصول على البيانات المباشرة
        age = data.get('age')
        gender = data.get('gender')
        weight = data.get('weight')
        height = data.get('height')
        activity_level = data.get('activity_level')
        dietary_restrictions = data.get('dietary_restrictions', [])
        goals = data.get('goals', [])
        medical_conditions = data.get('medical_conditions', [])
        
        # إذا لم يتم توفير البيانات المباشرة، يجب توفير user_info و goal
        if (age is None or weight is None or height is None) and (not user_info or not goal):
            return jsonify({
                "success": False,
                "error": "يجب توفير معلومات المستخدم (user_info و goal) أو البيانات المباشرة (age, weight, height)"
            }), 400
        
        # تحويل الهدف إلى قائمة إذا كان نصاً
        if isinstance(goals, str):
            goals = [goals] if goals else []
        
        # تحويل الأمراض إلى قائمة إذا كان نصاً
        if isinstance(medical_conditions, str):
            medical_conditions = [medical_conditions] if medical_conditions else []
        
        # توليد خطة الوجبات
        result = api.generate_meal_plan(
            user_info=user_info,
            goal=goal,
            days=days,
            detailed=detailed,
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            activity_level=activity_level,
            dietary_restrictions=dietary_restrictions,
            goals=goals,
            medical_conditions=medical_conditions
        )
        
        # تحديث الإحصائيات
        response_time = time.time() - start_time
        api.update_usage_stats(result["success"], response_time)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f" خطأ في API: {str(e)}")
        response_time = time.time() - start_time
        api.update_usage_stats(False, response_time)
        
        return jsonify({
            "success": False,
            "error": f"خطأ في الخادم: {str(e)}"
        }), 500

@app.route('/api/batch_generate', methods=['POST'])
def batch_generate():
    start_time = time.time()
    
    try:
        if not api.is_loaded:
            return jsonify({
                "success": False,
                "error": "النموذج غير محمل"
            }), 503
        
        data = request.get_json()
        if not data or 'requests' not in data:
            return jsonify({
                "success": False,
                "error": "بيانات غير صحيحة"
            }), 400
        
        requests = data['requests']
        if not isinstance(requests, list) or len(requests) == 0:
            return jsonify({
                "success": False,
                "error": "قائمة الطلبات فارغة أو غير صحيحة"
            }), 400
        
        # توليد الدفعات
        results = api.batch_generate(requests)
        
        # تحديث الإحصائيات
        response_time = time.time() - start_time
        api.update_usage_stats(True, response_time)
        
        return jsonify({
            "success": True,
            "results": results,
            "total_requests": len(requests)
        })
        
    except Exception as e:
        logger.error(f" خطأ في Batch API: {str(e)}")
        response_time = time.time() - start_time
        api.update_usage_stats(False, response_time)
        
        return jsonify({
            "success": False,
            "error": f"خطأ في الخادم: {str(e)}"
        }), 500

@app.route('/api/nutritional_info', methods=['GET'])
def nutritional_info():
    try:
        food_item = request.args.get('food', '')
        if not food_item:
            return jsonify({
                "success": False,
                "error": "اسم الطعام مطلوب"
            }), 400
        
        result = api.get_nutritional_info(food_item)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f" خطأ في Nutritional Info API: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"خطأ في الخادم: {str(e)}"
        }), 500

@app.route('/api/seasonal_foods', methods=['GET'])
def seasonal_foods():
    try:
        season = request.args.get('season', None)
        result = api.get_seasonal_foods(season)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f" خطأ في Seasonal Foods API: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"خطأ في الخادم: {str(e)}"
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    try:
        status_info = {
            "model_loaded": api.is_loaded,
            "usage_stats": api.usage_stats,
            "system_info": {
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "device": str(api.model.device) if api.model else "غير متاح",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if api.performance_optimizer:
            status_info["performance_info"] = api.performance_optimizer.get_performance_summary()
        
        return jsonify({
            "success": True,
            "status": status_info
        })
        
    except Exception as e:
        logger.error(f" خطأ في Status API: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"خطأ في الخادم: {str(e)}"
        }), 500

@app.route('/api/load_model', methods=['POST'])
def load_model():
    try:
        data = request.get_json() or {}
        model_path = data.get('model_path', None)
        
        api.load_model(model_path)
        
        return jsonify({
            "success": True,
            "message": "تم تحميل النموذج بنجاح" if api.is_loaded else "فشل في تحميل النموذج",
            "model_loaded": api.is_loaded
        })
        
    except Exception as e:
        logger.error(f" خطأ في تحميل النموذج: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"خطأ في تحميل النموذج: {str(e)}"
        }), 500
    
@app.route('/api/all_recipes', methods=['GET'])
def all_recipes():
    """
    Returns all recipes grouped by category, with:
      - tags: generated server-side when possible
      - nutrition / calories: only if ingredients list exists and ingredients are found in nutritional DB
    No default/fallback values are injected for missing fields.
    """
    
    try:
        enhanced_db = {}

        # helper: find nutrition entry for an ingredient name (case-insensitive)
        def lookup_ingredient_nutrition(ingredient_name):
            if not ingredient_name:
                return None
            name_lower = ingredient_name.strip().lower()
            nd = data_enhancer.nutritional_database
            for cat, foods in nd.items():
                for food_name, vals in foods.items():
                    if food_name.lower() == name_lower:
                        return vals
            return None

        # helper: compute nutrition totals if recipe has ingredients list
        def compute_nutrition_from_ingredients(ingredients):
            totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
            found_any = False
            for ing in ingredients:
                nut = lookup_ingredient_nutrition(ing)
                if nut:
                    found_any = True
                    totals["calories"] += nut.get("calories", 0)
                    totals["protein"]  += nut.get("protein", 0)
                    totals["carbs"]    += nut.get("carbs", 0)
                    totals["fat"]      += nut.get("fat", 0)
            if not found_any:
                return None
            # round ints
            for k in totals:
                totals[k] = int(round(totals[k]))
            return totals

        # iterate categories
        for category, recipes in data_enhancer.recipe_database.items():
            enhanced_list = []
            for recipe in recipes:
                # make shallow copy to avoid mutating original
                r = dict(recipe)

                # tags: use DataEnhancer's tag generator if available
                try:
                    # DataEnhancer internal name is _generate_recipe_tags
                    if hasattr(data_enhancer, "_generate_recipe_tags"):
                        r_tags = data_enhancer._generate_recipe_tags(r)
                    elif hasattr(data_enhancer, "generate_recipe_tags"):
                        r_tags = data_enhancer.generate_recipe_tags(r)
                    else:
                        r_tags = None
                except Exception:
                    r_tags = None

                if r_tags:
                    r["tags"] = r_tags

                # nutrition/calories: only if ingredients list exists
                ingredients = r.get("ingredients")  # may be None
                if isinstance(ingredients, list) and len(ingredients) > 0:
                    nutrition_totals = compute_nutrition_from_ingredients(ingredients)
                    if nutrition_totals:
                        r["nutrition"] = {
                            "calories": nutrition_totals["calories"],
                            "protein_g": nutrition_totals["protein"],
                            "carbs_g": nutrition_totals["carbs"],
                            "fat_g": nutrition_totals["fat"]
                        }
                        r["calories"] = nutrition_totals["calories"]
                # if no ingredients or couldn't compute, we don't inject calories/nutrition

                # ensure image key remains as-is (do not replace)
                # r["image"] = r.get("image")  # no-op, kept for clarity

                enhanced_list.append(r)

            enhanced_db[category] = enhanced_list

        return jsonify({
            "success": True,
            "recipes": enhanced_db
        }), 200

    except Exception as e:
        logger.exception("خطأ عند تجهيز all_recipes")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500
    
@app.route('/api/recipe_nutrition', methods=['POST'])
def recipe_nutrition():
    try:
        data = request.get_json()
        if not data or "ingredients" not in data:
            return jsonify({"success": False, "error": "ingredients مطلوب"}), 400

        nutrition = data_enhancer.calculate_recipe_nutrition(data["ingredients"])
        return jsonify({"success": True, "nutrition": nutrition})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # تحميل النموذج عند بدء التشغيل
    logger.info("🚀 بدء تشغيل خادم API...")
    
    # محاولة تحميل النموذج
    api.load_model()
    
    # تشغيل الخادم
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5001,  # منفذ مختلف لتجنب التضارب
        threaded=True
    )
