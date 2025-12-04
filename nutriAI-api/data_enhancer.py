import json
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import re

@dataclass
class UserProfile:
    age: int
    gender: str
    weight: float
    height: float
    activity_level: str
    dietary_restrictions: List[str]
    allergies: List[str]
    goals: List[str]
    preferences: List[str]
    budget: str
    cooking_skill: str
    available_time: str
    medical_conditions: List[str]  # الأمراض والحالات الطبية
    daily_calories: float = 0.0         # السعرات اليومية
    target_macros: Dict[str, float] = None

class DataEnhancer:
    
    def __init__(self):
        self.nutritional_database = self._load_nutritional_database()
        self.recipe_database = self._load_recipe_database()
        self.seasonal_foods = self._load_seasonal_foods()
        self.medical_conditions_db = self._load_medical_conditions_database()
        self._attach_tags_to_recipes()
        
    def _load_nutritional_database(self) -> Dict:
     return {
        "proteins": {
            "صدر دجاج": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
            "ورك دجاج": {"calories": 209, "protein": 26, "carbs": 0, "fat": 10.9},
            "دجاج": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
            "دجاج مشوي": {"calories": 180, "protein": 32, "carbs": 0, "fat": 4},
            "سلمون": {"calories": 208, "protein": 20, "carbs": 0, "fat": 13},
            "سمك": {"calories": 90, "protein": 19, "carbs": 0, "fat": 1},
            "تونة": {"calories": 132, "protein": 29, "carbs": 0, "fat": 1},
            "بيض": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11},
            "بياض بيض": {"calories": 52, "protein": 11, "carbs": 0.7, "fat": 0.2},
            "لحم بقري": {"calories": 250, "protein": 26, "carbs": 0, "fat": 15},
            "لحم": {"calories": 250, "protein": 26, "carbs": 0, "fat": 15},
            "لحم مفروم": {"calories": 176, "protein": 20, "carbs": 0, "fat": 10},
            "لحم مفروم قليل الدهن": {"calories": 137, "protein": 22, "carbs": 0, "fat": 5},
            "توفو": {"calories": 76, "protein": 8, "carbs": 1.9, "fat": 4.8},
            "حمص": {"calories": 164, "protein": 9, "carbs": 27, "fat": 2.6},
            "عدس": {"calories": 116, "protein": 9, "carbs": 20, "fat": 0.4},
            "فول": {"calories": 110, "protein": 7.6, "carbs": 18, "fat": 0.4},
            "جبنة قريش": {"calories": 98, "protein": 11, "carbs": 3.4, "fat": 4.3},
            "جبنة بيضاء": {"calories": 264, "protein": 14, "carbs": 4, "fat": 21},
            "جبنة": {"calories": 403, "protein": 25, "carbs": 1.3, "fat": 33},
            "زبادي يوناني": {"calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4},
            "لبنة": {"calories": 130, "protein": 8.5, "carbs": 5.3, "fat": 8.5},
            "لبن زبادي": {"calories": 61, "protein": 3.5, "carbs": 4.7, "fat": 3.3},
        },

        "carbs": {
            "أرز": {"calories": 130, "protein": 2.4, "carbs": 28, "fat": 0.3},
            "أرز بني": {"calories": 111, "protein": 2.6, "carbs": 23, "fat": 0.9},
            "أرز أبيض": {"calories": 130, "protein": 2.4, "carbs": 28, "fat": 0.3},
            "كينوا": {"calories": 120, "protein": 4.4, "carbs": 22, "fat": 1.9},
            "برغل": {"calories": 342, "protein": 12, "carbs": 76, "fat": 1.3},
            "كسكسي": {"calories": 112, "protein": 3.8, "carbs": 23, "fat": 0.2},
            "بطاطا": {"calories": 87, "protein": 1.9, "carbs": 20, "fat": 0.1},
            "بطاطا حلوة": {"calories": 86, "protein": 1.6, "carbs": 20, "fat": 0.1},
            "خبز أسمر": {"calories": 247, "protein": 13, "carbs": 41, "fat": 4.2},
            "توست": {"calories": 247, "protein": 13, "carbs": 41, "fat": 4.2},
            "خبز محمص": {"calories": 350, "protein": 11, "carbs": 64, "fat": 3},
            "مكرونة": {"calories": 131, "protein": 5, "carbs": 25, "fat": 1.1},
            "معكرونة بينّه": {"calories": 131, "protein": 5, "carbs": 25, "fat": 1.1},
            "جرانولا": {"calories": 471, "protein": 10, "carbs": 64, "fat": 20},
            "شوفان": {"calories": 389, "protein": 17, "carbs": 66, "fat": 7},
            "دقيق الشوفان": {"calories": 389, "protein": 17, "carbs": 66, "fat": 7},
            "تمر": {"calories": 277, "protein": 1.8, "carbs": 75, "fat": 0.2},
            "عسل": {"calories": 304, "protein": 0.3, "carbs": 82, "fat": 0},
        },

        "vegetables": {
            "طماطم": {"calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
            "خيار": {"calories": 16, "protein": 0.7, "carbs": 3.6, "fat": 0.1},
            "جزر": {"calories": 41, "protein": 0.9, "carbs": 10, "fat": 0.2},
            "بصل": {"calories": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1},
            "باذنجان": {"calories": 25, "protein": 1, "carbs": 6, "fat": 0.2},
            "فلفل رومي": {"calories": 31, "protein": 1, "carbs": 6, "fat": 0.3},
            "سبانخ": {"calories": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4},
            "خس": {"calories": 15, "protein": 1.4, "carbs": 2.9, "fat": 0.2},
            "قرنبيط": {"calories": 25, "protein": 1.9, "carbs": 5, "fat": 0.3},
            "بروكلي": {"calories": 55, "protein": 3.7, "carbs": 11, "fat": 0.6},
            "كرفس": {"calories": 16, "protein": 0.7, "carbs": 3, "fat": 0.2},
            "بازلاء": {"calories": 81, "protein": 5, "carbs": 14, "fat": 0.4},
            "خضار مشكلة": {"calories": 65, "protein": 3, "carbs": 11, "fat": 0.5},
            "فلفل": {"calories": 20, "protein": 0.9, "carbs": 4.6, "fat": 0.2},
            "فطر": {"calories": 22, "protein": 3.1, "carbs": 3.3, "fat": 0.3},
            "كوسة": {"calories": 17, "protein": 1.2, "carbs": 3.1, "fat": 0.3},
            "بقدونس": {"calories": 36, "protein": 3, "carbs": 6, "fat": 0.8},
        },

        "fruits": {
            "زبيب": {"calories": 299, "protein": 3.1, "carbs": 79, "fat": 0.5},
            "مشمش مجفف": {"calories": 241, "protein": 3.4, "carbs": 63, "fat": 0.5},
            "تفاح": {"calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2},
            "موز": {"calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3},
            "توت": {"calories": 57, "protein": 0.7, "carbs": 14, "fat": 0.3},
            "فراولة": {"calories": 32, "protein": 0.7, "carbs": 7.7, "fat": 0.3},
            "عنب": {"calories": 69, "protein": 0.7, "carbs": 18, "fat": 0.2},
            "مانجا": {"calories": 60, "protein": 0.8, "carbs": 15, "fat": 0.4},
            "كيوي": {"calories": 61, "protein": 1.1, "carbs": 15, "fat": 0.5},
            "بطيخ": {"calories": 30, "protein": 0.6, "carbs": 8, "fat": 0.2},
            "ليمون": {"calories": 29, "protein": 1.1, "carbs": 9, "fat": 0.3},
        },

        "fats": {
            "زيتون": {"calories": 115, "protein": 0.8, "carbs": 6, "fat": 10.7},
            "زيت زيتون": {"calories": 884, "protein": 0, "carbs": 0, "fat": 100},
            "زيت زيتون بكر": {"calories": 884, "protein": 0, "carbs": 0, "fat": 100},
            "زعتر": {"calories": 276, "protein": 9, "carbs": 60, "fat": 7},
            "لوز": {"calories": 579, "protein": 21, "carbs": 22, "fat": 50},
            "مكسرات": {"calories": 607, "protein": 20, "carbs": 21, "fat": 53},
            "جوز": {"calories": 654, "protein": 15, "carbs": 14, "fat": 65},
            "فستق": {"calories": 562, "protein": 21, "carbs": 28, "fat": 45},
            "زبدة لوز": {"calories": 614, "protein": 21, "carbs": 20, "fat": 55},
            "زبدة فول سوداني": {"calories": 588, "protein": 25, "carbs": 20, "fat": 50},
            "سمن": {"calories": 900, "protein": 0, "carbs": 0, "fat": 100},
        },
        "dairy": {
            "حليب": {"calories": 61, "protein": 3.2, "carbs": 4.8, "fat": 3.3},
            "حليب كامل الدسم": {"calories": 61, "protein": 3.2, "carbs": 4.8, "fat": 3.3},
            "حليب قليل الدسم": {"calories": 50, "protein": 3.5, "carbs": 5.1, "fat": 1.2},
            "لبن": {"calories": 61, "protein": 3.4, "carbs": 4.7, "fat": 3.3},
            "لبن زبادي": {"calories": 61, "protein": 3.4, "carbs": 4.7, "fat": 3.3},
            "زبادي يوناني": {"calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4},
            "جبنة موزاريلا": {"calories": 280, "protein": 28, "carbs": 3.1, "fat": 17},
            "جبنة فيتا": {"calories": 264, "protein": 14, "carbs": 4, "fat": 21},
            "قشطة": {"calories": 350, "protein": 2, "carbs": 3, "fat": 37},
        },

        "others": {
            "شاي": {"calories": 1, "protein": 0, "carbs": 0.2, "fat": 0},
            "فول سوداني": {"calories": 567, "protein": 26, "carbs": 16, "fat": 49},
            "بهارات شاورما": {"calories": 277, "protein": 14, "carbs": 58, "fat": 7},
            "ريحان": {"calories": 22, "protein": 3.1, "carbs": 2.7, "fat": 0.6},
            "طحينة": {"calories": 595, "protein": 17, "carbs": 21, "fat": 53},
            "خردل": {"calories": 66, "protein": 4, "carbs": 5, "fat": 4},
            "بهارات كبسة": {"calories": 325, "protein": 12, "carbs": 58, "fat": 14},
            "بهارات": {"calories": 247, "protein": 9, "carbs": 41, "fat": 7},
            "صلصة طماطم": {"calories": 29, "protein": 1.4, "carbs": 5.6, "fat": 0.3},
            "صلصة صويا": {"calories": 53, "protein": 8, "carbs": 17, "fat": 0.5},
            "صلصة طحينة": {"calories": 595, "protein": 17, "carbs": 21, "fat": 53},
            "خل": {"calories": 18, "protein": 0, "carbs": 0.9, "fat": 0},
            "بروتين بودرة": {"calories": 400, "protein": 80, "carbs": 8, "fat": 7},
            "شيا": {"calories": 486, "protein": 17, "carbs": 42, "fat": 31},
            "ورق عنب": {"calories": 93, "protein": 5.6, "carbs": 17, "fat": 2.3},
            "جميد": {"calories": 310, "protein": 23, "carbs": 7, "fat": 21},
        }
    }
    
    def _load_recipe_database(self) -> Dict:
        return {
            "breakfast": [
               {"name": "شوفان مع التوت والمكسرات",
                "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZ8vi_scEx93nGq6qWXrat1fijcv_fP_xnOA&s?w=400",
                "ingredients": ["شوفان", "توت", "مكسرات", "عسل"]},

              {"name": "زبادي يوناني مع الجرانولا",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSGt7C3vqP5xs8GkbAdYzs7Fd-YPG6qimm8iQ&s?w=400",
               "ingredients": ["زبادي يوناني", "جرانولا", "عسل"]},

              {"name": "بيض مخفوق مع الخضار",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8QmhkV8zKCKa1E9o2zUDmf030m0w7apqRSQ&s?w=400",
               "ingredients": ["بيض", "طماطم", "سبانخ", "فلفل"]},

              {"name": "وعاء سموذي بالفواكه",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkSpiBtY6BvdFstFzVvumUV6wOixuYz3WNAA&s?w=400",
               "ingredients": ["موز", "توت", "حليب", "شوفان"]},

              {"name": "توست الأفوكادو مع البيض",
               "image": "https://images.unsplash.com/photo-1541519227354-08fa5d50c44d?w=400",
               "ingredients": ["توست", "أفوكادو", "بيض"]},

              {"name": "فول مدمس مع الطحينة",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSuyRnb2iG6twSZHYiYtmViL9gDtDLwoK0QOQ&s?w=400",
               "ingredients": ["فول", "طحينة", "ليمون"]},

              {"name": "جبنة بيضاء مع الزعتر",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRtUXjfopdC_8ojirGaTZT6zN5L4mxEL4JepA&s?w=400",
               "ingredients": ["جبنة بيضاء", "زعتر", "زيت زيتون"]},

              {"name": "شاي بالحليب مع البسكويت",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtmO_gdwNnZcQ5P3aRM_Z5Yu7RQsRTb7fGxw&s?w=400",
               "ingredients": ["شاي", "حليب", "بسكويت"]},

              {"name": "عجة بالجبن والخضار",
               "image": "https://images.unsplash.com/photo-1525351484163-7529414344d8?w=400",
               "ingredients": ["بيض", "جبنة", "طماطم", "فلفل"]},

              {"name": "حليب مع التمر واللوز",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT8Z3UyEhOKJ9202S2GNmraA9sxBgvwnAyu3A&s?w=400",
               "ingredients": ["تمر", "لوز", "حليب"]},

              {"name": "شكشوكة صحية",
               "image": "https://images.pexels.com/photos/6275105/pexels-photo-6275105.jpeg?w=400",
               "ingredients": ["بيض", "طماطم", "بصل", "فلفل", "زيت زيتون"]},

              {"name": "سندويتش جبنة قريش مع خيار",
               "image": "https://images.pexels.com/photos/12540686/pexels-photo-12540686.png?w=400",
               "ingredients": ["خبز أسمر", "جبنة قريش", "خيار"]},

              {"name": "بان كيك الشوفان الصحي",
               "image": "https://images.pexels.com/photos/14263510/pexels-photo-14263510.jpeg?w=400",
               "ingredients": ["شوفان", "موز", "بيض"]},

              {"name": "لبنة مع زيت الزيتون وزعتر",
               "image": "https://i.ibb.co/XrLBGLGg/Image-29.png?w=400",
               "ingredients": ["لبنة", "زعتر", "زيت زيتون", "خبز أسمر"]},

              {"name": "عصيدة تمر صحية",
               "image": "https://i.ibb.co/DDy45JZz/image.jpg?w=400",
               "ingredients": ["تمر", "دقيق الشوفان", "حليب"]}],
            "lunch": [
              {"name": "سلطة دجاج مشوي",
                "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpJ0fnpGWGxrWMLn0Ax6nW-KCpB0t4Jo-_Gw&s?w=400",
                "ingredients": ["دجاج مشوي", "خس", "خيار", "طماطم", "زيت زيتون", "ليمون"]},
              {"name": "وعاء الكينوا مع الخضار",
               "image": "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=400",
               "ingredients": ["كينوا", "بروكلي", "جزر", "فلفل رومي", "زيت زيتون"]},
              {"name": "شوربة العدس مع الخبز",
               "image": "https://i.ytimg.com/vi/9j66YdbkuF8/hq720.jpg",
               "ingredients": ["عدس", "بصل", "طماطم", "كمون", "خبز أسمر"]},
              {"name": "سلمون مع خضار محمصة",
               "image": "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400",
               "ingredients": ["سلمون", "بطاطا حلوة", "بروكلي", "زيت زيتون", "ثوم"]},
              {"name": "خضار سوتيه مع التوفو",
               "image": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400",
               "ingredients": ["توفو", "سبانخ", "فطر", "فلفل رومي", "صلصة صويا"]},
              {"name": "كشري مصري",
               "image": "https://i.ytimg.com/vi/cJlZpxTavp8/sddefault.jpg?w=400",
               "ingredients": ["أرز", "عدس", "مكرونة", "حمص", "صلصة طماطم"]},
              {"name": "مسقعة الباذنجان",
               "image": "https://images.unsplash.com/photo-1574484284002-952d92456975?w=400",
               "ingredients": ["باذنجان", "لحم مفروم", "طماطم", "بصل", "ثوم"]},
              {"name": "محشي ورق العنب",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTgd26As8BEHHOVP32od41NFUhdnnaes6Mz2w&s?w=400",
               "ingredients": ["ورق عنب", "أرز", "بقدونس", "طماطم", "زيت زيتون"]},
              {"name": "كبسة دجاج",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTqWPNivmfclbrDZ-fwXmTZtHYuR2OVLR5mcg&s?w=400",
               "ingredients": ["دجاج", "أرز", "بهارات كبسة", "بصل", "طماطم"]},
              {"name": "فتة باللحم",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOqD0FTqovGgO0oMXknTcqAg19SqqbEqTQkw&s?w=400",
               "ingredients": ["لحم", "خبز محمص", "لبن زبادي", "ثوم", "طحينة"]},
              {"name": "دجاج بالخردل والعسل",
               "image": "https://i.ibb.co/V0J5rGDH/image.jpg?w=400",
               "ingredients": ["دجاج", "عسل", "خردل", "ثوم", "بطاطا"]},
              {"name": "سلطة التونة الصحية",
               "image": "https://i.ibb.co/h1gR0x4P/image.jpg?w=400",
               "ingredients": ["تونة", "ذرة", "خيار", "خس", "ليمون"]},
              {"name": "مكرونة بينيه بالدجاج والخضار",
               "image": "https://i.ibb.co/zTfJbwbZ/image.jpg?w=400",
               "ingredients": ["مكرونة", "دجاج", "فلفل رومي", "بروكلي", "ثوم"]},
              {"name": "شوربة خضار بالدجاج",
               "image": "https://i.ibb.co/v4LXsGB6/image.png?w=400",
               "ingredients": ["دجاج", "جزر", "كرفس", "بصل", "بطاطا"]},
              {"name": "سمك مشوي بالليمون",
               "image": "https://i.ibb.co/NdXcWvkY/Gemini-Generated-Image-leaeheleaeheleae.png?w=400",
               "ingredients": ["سمك", "ليمون", "زيت زيتون", "بقدونس", "ثوم"]}],

            "dinner": [
              {"name": "سلمون محمص مع البطاطا الحلوة",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSf0iNacJyY8eS-oaXxn6i7q6JwZFhuCVcCUw&s?w=400",
               "ingredients": ["سلمون", "بطاطا حلوة", "بروكلي", "ثوم", "زيت زيتون"]},
              {"name": "كاري دجاج مع الخضار",
               "image": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=400",
               "ingredients": ["دجاج", "حليب جوز الهند", "كاري", "بطاطا", "جزر"] },
              {"name": "باستا نباتية مع الصلصة",
               "image": "https://images.unsplash.com/photo-1551183053-bf91a1d81141?w=400",
               "ingredients": ["مكرونة", "طماطم", "ريحان", "فطر", "زيت زيتون"]},
              {"name": "دجاج مشوي مع الكينوا",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRNBynFxHEZIK9EKwD_G3mgfOwCODQCIs0cOw&s?w=400",
               "ingredients": ["دجاج", "كينوا", "خيار", "طماطم", "ليمون"]},
              {"name": "فلفل محشي",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSNqOv9Z23Yc6DkOdIvrFPvbCvA-8AxsO33Ew&s?w=400",
               "ingredients": ["فلفل رومي", "أرز", "طماطم", "بصل", "بقدونس"]},
              {"name": "مقلوبة لحم",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfltMPNJFa3__rhaq8w6OkZUyfpMKeNhaz8g&s?w=400",
               "ingredients": ["لحم", "باذنجان", "أرز", "طماطم", "بهارات"]},
              {"name": "منسف أردني",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTlE4YQWwrTg83Dr2-G5juyc-peO6l6hhUwCw&s?w=400",
               "ingredients": ["لحم", "لبن", "جميد", "أرز", "سمن"]},
              {"name": "كباب مشوي مع الأرز",
               "image": "https://images.unsplash.com/photo-1529042410759-befb1204b468?w=400",
               "ingredients": ["لحم مفروم", "بقدونس", "بصل", "أرز", "بهارات"]},
              {"name": "شاورما دجاج",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe037vjzTy5OzUebsgdVtSx9QoC8XnjivsCg&s?w=400",
               "ingredients": ["دجاج", "ثوم", "لبن زبادي", "خل", "بهارات شاورما"]},
              {"name": "مشاوي مشكلة",
               "image": "https://images.unsplash.com/photo-1529042410759-befb1204b468?w=400",
               "ingredients": ["كباب", "شيش طاووق", "خضار مشوية", "أرز", "صلصة طحينة"]},
              {"name": "صينية دجاج وخضار صحية",
               "image": "https://i.ibb.co/JRxzCKNj/image.png?w=400",
               "ingredients": ["دجاج", "بطاطا", "كوسة", "جزر", "زيت زيتون"]},
              {"name": "سمك قاروص مشوي بالأعشاب",
               "image": "https://i.ibb.co/WWwDG16F/image.png?w=400",
               "ingredients": ["سمك", "ثوم", "ليمون", "بقدونس", "زيت زيتون"]},
              {"name": "مكرونة بصلصة الطماطم والريحان",
               "image": "https://i.ibb.co/PGfSGsF6/image.png?w=400",
               "ingredients": ["مكرونة", "طماطم", "ريحان", "ثوم", "فلفل أسود"]},
              {"name": "مرقوق صحي",
               "image": "https://i.ibb.co/XrB9bwQV/image.png?w=400",
               "ingredients": ["دقيق قمح", "لحم", "كوسة", "بطاطا", "طماطم"]},
              {"name": "أرز بالخضار والدجاج",
               "image": "https://i.ibb.co/B5jBryPc/image.png?w=400",
               "ingredients": ["أرز", "دجاج", "بازلاء", "جزر", "بصل"]}
               ],
            "snacks": [
              {"name": "مكسرات وفواكه مجففة",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSt64RwOHj0USiggQTR9cQyS4EhUga1dtQ1aQ&s?w=400",
               "ingredients": ["مكسرات", "زبيب", "مشمش مجفف", "جوز"]},
              {"name": "زبادي يوناني مع العسل",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT42dF--h9Tp7ao4ii4hrMkQp4RHk3NA0sUpw&s?w=400",
               "ingredients": ["زبادي يوناني", "عسل", "جوز"]},
              {"name": "شرائح تفاح مع زبدة اللوز",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTYIM1rUWxZRLUgHw1pBBZB0C0iaDcmRE2B6A&s?w=400",
               "ingredients": ["تفاح", "زبدة لوز"]},
              {"name": "خضار مع الحمص",
               "image": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400",
               "ingredients": ["خيار", "جزر", "حمص"]},
              {"name": "سموذي بروتين",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1w3dvPGbMEpf7S2YDw6e4_tye--e6t1C9cA&s?w=400",
               "ingredients": ["حليب", "موز", "بروتين بودرة"]},
              {"name": "تمر مع اللوز",
               "image": "https://i.ytimg.com/vi/B16gghM2-_I/sddefault.jpg?w=400",
               "ingredients": ["تمر", "لوز"]},
              {"name": "جبنة مع الزيتون",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTKUY0TFZduTnUrTJDtEiyoGbB2oeQhAYfatw&s?w=400",
               "ingredients": ["جبنة بيضاء", "زيتون"]},
              {"name": "بسكويت مع الشاي",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQXBvxY7IXU3diOzHqXaDkWr7j9nK-bTV2oRw&s?w=400",
               "ingredients": ["بسكويت", "شاي", "حليب"]},
              {"name": "فواكه طازجة",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQX39QmxqHMlly5szukexEo6Vnvh5_KM543Ww&s?w=400",
               "ingredients": ["تفاح", "موز", "عنب", "فراولة"]},
              {"name": "مكسرات محمصة",
               "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT08jDgIRPwirKS5IChEEi-BOe5gyIZWWhvJA&s?w=400",
               "ingredients": ["مكسرات", "لوز", "جوز", "فستق"]},
              {"name": "بودينغ الشيا بالفواكه",
               "image": "https://i.ibb.co/VW05yhgn/IMG-1096.jpg?w=400",
               "ingredients": ["شيا", "حليب", "عسل", "توت"]},
              {"name": "توست زبدة الفول السوداني والموز",
               "image": "https://i.ibb.co/LDWqgvG1/unnamed.jpg?w=400",
               "ingredients": ["توست", "زبدة فول سوداني", "موز"]},
              {"name": "لبنة مع خيار وزيت زيتون",
               "image": "https://i.ibb.co/Z6tZ1bZd/IMG-1097.jpg?w=400",
               "ingredients": ["لبنة", "خيار", "زيت زيتون", "زعتر"]},
              {"name": "سموذي أخضر",
               "image": "https://i.ibb.co/d4Z32Lgj/IMG-1098.jpg?w=400",
               "ingredients": ["سبانخ", "تفاح", "كيوي", "ليمون"]},
              {"name": "بروتين بار منزلي",
               "image": "https://i.ibb.co/bjK5LNLH/IMG-1099.jpg?w=400",
               "ingredients": ["شوفان", "عسل", "بروتين بودرة", "فول سوداني"]}
     ]
    }
    
    
    def _load_seasonal_foods(self) -> Dict:
        return {
            "spring": ["هليون", "خرشوف", "بازلاء", "فجل", "فراولة"],
            "summer": ["طماطم", "ذرة", "خوخ", "بطيخ", "كوسة"],
            "autumn": ["قرع", "تفاح", "قرع شتوي", "توت بري", "بطاطا حلوة"],
            "winter": ["حمضيات", "ملفوف", "جزر", "بطاطا", "قرع شتوي"]
        }
    
    def _load_medical_conditions_database(self) -> Dict:
        """قاعدة بيانات للأمراض والقيود الغذائية المناسبة لكل مرض"""
        return {
            "diabetes": {
                "name_ar": "السكري",
                "avoid": ["سكر", "عسل", "تمر", "عنب", "موز", "أرز أبيض", "خبز أبيض", "معكرونة", "عصائر"],
                "prefer": ["خضار", "بروتين", "أرز بني", "شوفان", "كينوا", "تفاح", "توت", "بروكلي", "سبانخ"],
                "restrictions": ["low_sugar", "low_carb"],
                "notes": "تجنب السكريات البسيطة، التركيز على الكربوهيدرات المعقدة والألياف"
            },
            "hypertension": {
                "name_ar": "ارتفاع ضغط الدم",
                "avoid": ["ملح", "أطعمة مملحة", "معلبات", "لحوم معالجة", "جبنة مملحة"],
                "prefer": ["خضار", "فواكه", "أسماك", "مكسرات غير مملحة", "حبوب كاملة"],
                "restrictions": ["low_sodium"],
                "notes": "تقليل الصوديوم، زيادة البوتاسيوم والمغنيسيوم"
            },
            "heart_disease": {
                "name_ar": "أمراض القلب",
                "avoid": ["دهون مشبعة", "لحوم حمراء", "أطعمة مقلية", "سكر", "ملح"],
                "prefer": ["أسماك", "خضار", "فواكه", "حبوب كاملة", "زيت زيتون", "مكسرات"],
                "restrictions": ["low_saturated_fat", "low_sodium"],
                "notes": "نظام غذائي متوسطي، تقليل الدهون المشبعة والكوليسترول"
            },
            "kidney_disease": {
                "name_ar": "أمراض الكلى",
                "avoid": ["بروتين عالي", "ملح", "بوتاسيوم عالي", "فوسفور عالي"],
                "prefer": ["خضار منخفضة البوتاسيوم", "بروتين نباتي معتدل"],
                "restrictions": ["low_protein", "low_sodium", "low_potassium"],
                "notes": "تقليل البروتين والصوديوم والبوتاسيوم حسب حالة الكلى"
            },
            "celiac": {
                "name_ar": "الداء البطني (حساسية الجلوتين)",
                "avoid": ["قمح", "شعير", "جاودار", "خبز", "معكرونة", "كعك"],
                "prefer": ["أرز", "كينوا", "بطاطا", "بطاطا حلوة", "خضار", "فواكه"],
                "restrictions": ["gluten_free"],
                "notes": "تجنب جميع الأطعمة المحتوية على الجلوتين"
            },
            "ibs": {
                "name_ar": "متلازمة القولون العصبي",
                "avoid": ["بقوليات", "كرنب", "بروكلي", "أطعمة دهنية", "كافيين"],
                "prefer": ["أرز", "بطاطا", "خضار مطبوخة", "بروتين خفيف"],
                "restrictions": ["low_fodmap"],
                "notes": "نظام غذائي منخفض FODMAP لتقليل أعراض القولون"
            }
        }
    
    def enhance_user_profile(self, basic_info: str = None, age: int = None, 
                            gender: str = None, weight: float = None, 
                            height: float = None, activity_level: str = None,
                            dietary_restrictions: List[str] = None,
                            goals: List[str] = None,
                            medical_conditions: List[str] = None) -> UserProfile:
        """
        تحسين ملف المستخدم من النص أو من المعاملات المباشرة.
        إذا تم تمرير المعاملات مباشرة، سيتم استخدامها بدلاً من استخراجها من النص.
        """
        # إذا تم تمرير البيانات مباشرة، استخدمها
        if age is not None and weight is not None and height is not None:
            # استخدام البيانات الممررة مباشرة
            final_age = age
            final_weight = weight
            final_height = height
            final_gender = gender or "other"
            final_activity_level = activity_level or "moderate"
            final_dietary_restrictions = dietary_restrictions or []
            final_goals = goals or ["maintenance"]
            final_medical_conditions = medical_conditions or []
        elif basic_info:
            # استخراج المعلومات من النص
            final_age = self._extract_age(basic_info)
            final_gender = self._extract_gender(basic_info)
            final_weight = self._extract_weight(basic_info)
            final_height = self._extract_height(basic_info)
            
            # تحديد المستوى النشاط
            final_activity_level = self._determine_activity_level(basic_info)
            
            # تحديد القيود الغذائية
            final_dietary_restrictions = self._extract_dietary_restrictions(basic_info)
            
            # تحديد الأهداف
            final_goals = self._extract_goals(basic_info)
            
            # استخراج الأمراض
            final_medical_conditions = self._extract_medical_conditions(basic_info)
        else:
            # استخدام القيم الافتراضية إذا لم يتم توفير أي بيانات
            final_age = 30
            final_gender = "other"
            final_weight = 70.0
            final_height = 170.0
            final_activity_level = "moderate"
            final_dietary_restrictions = []
            final_goals = ["maintenance"]
            final_medical_conditions = []
        
        # إضافة القيود الغذائية من الأمراض
        if final_medical_conditions:
            for condition in final_medical_conditions:
                if condition in self.medical_conditions_db:
                    condition_data = self.medical_conditions_db[condition]
                    # إضافة القيود الغذائية من المرض إلى القيود العامة
                    for restriction in condition_data.get("restrictions", []):
                        if restriction not in final_dietary_restrictions:
                            final_dietary_restrictions.append(restriction)
        
        return UserProfile(
            age=final_age,
            gender=final_gender,
            weight=final_weight,
            height=final_height,
            activity_level=final_activity_level,
            dietary_restrictions=final_dietary_restrictions,
            allergies=[],
            goals=final_goals,
            preferences=[],
            budget="medium",
            cooking_skill="intermediate",
            available_time="moderate",
            medical_conditions=final_medical_conditions
        )
    
    def _extract_age(self, text: str) -> int:
        # Normalize numbers and common variants (including Arabic words and Arabic-Indic digits)
        t = self._normalize_numbers(text)

        # Common patterns: '28 years', '28 yrs', '28 سنة', 'عمر: 28', 'age: 28'
        age_match = re.search(r'(\d{1,3})\s*(?:years?|yrs?|سنة|عام|أعوام)\b', t, re.IGNORECASE)
        if age_match:
            return int(age_match.group(1))

        age_match = re.search(r'age[:\s،]*?(\d{1,3})\b', t, re.IGNORECASE)
        if age_match:
            return int(age_match.group(1))

        # Arabic pattern with colon or comma (including Arabic comma)
        age_match = re.search(r'عمر[:\s،]*?(\d{1,3})\b', t, re.IGNORECASE)
        if age_match:
            return int(age_match.group(1))

        # Pattern: "عمر: 28،" or "عمر: 28"
        age_match = re.search(r'عمر[:\s]*(\d{1,3})[،,]', t, re.IGNORECASE)
        if age_match:
            return int(age_match.group(1))

        # If nothing explicit found, keep sensible default
        return 30
    
    def _extract_gender(self, text: str) -> str:
        if re.search(r'\bfemale\b', text, re.IGNORECASE):
            return "female"
        elif re.search(r'\bmale\b', text, re.IGNORECASE):
            return "male"
        return "other"
    
    def _extract_weight(self, text: str) -> float:
        t = self._normalize_numbers(text)

        # Match kg with various spellings (English and Arabic)
        weight_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:kg|kgs|kilogram|kilograms|كجم|كغ|كيلوغرام|كيلو)\b', t, re.IGNORECASE)
        if weight_match:
            return float(weight_match.group(1).replace(',', '.'))

        # 'weight: 70' or 'weight 70'
        weight_match = re.search(r'weight[:\s،]*?(\d+(?:[.,]\d+)?)\b', t, re.IGNORECASE)
        if weight_match:
            return float(weight_match.group(1).replace(',', '.'))

        # Arabic label 'وزن' or 'الوزن' with colon or comma
        weight_match = re.search(r'وزن[:\s،]*?(\d+(?:[.,]\d+)?)\b', t, re.IGNORECASE)
        if weight_match:
            return float(weight_match.group(1).replace(',', '.'))

        # Pattern: "وزن: 70 كجم" or "وزن: 70،"
        weight_match = re.search(r'وزن[:\s]*(\d+(?:[.,]\d+)?)\s*(?:كجم|كغ|كيلو)', t, re.IGNORECASE)
        if weight_match:
            return float(weight_match.group(1).replace(',', '.'))

        # As a last attempt, pick a plausible weight number if present (20-300 kg)
        nums = re.findall(r'(\d+(?:[.,]\d+)?)', t)
        for n in nums:
            try:
                val = float(n.replace(',', '.'))
            except:
                continue
            if 20 <= val <= 300:
                return val

        return 70.0
    
    def _extract_height(self, text: str) -> float:
        t = self._normalize_numbers(text)

        # Match cm with English and Arabic words
        height_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:cm|centimeters|سم)\b', t, re.IGNORECASE)
        if height_match:
            return float(height_match.group(1).replace(',', '.'))

        # 'height: 165' or Arabic 'طول: 165' / 'الطول'
        height_match = re.search(r'height[:\s،]*?(\d+(?:[.,]\d+)?)\b', t, re.IGNORECASE)
        if height_match:
            return float(height_match.group(1).replace(',', '.'))

        # Arabic pattern with colon or comma
        height_match = re.search(r'(?:طول|الطول)[:\s،]*?(\d+(?:[.,]\d+)?)\b', t, re.IGNORECASE)
        if height_match:
            return float(height_match.group(1).replace(',', '.'))

        # Pattern: "طول: 170 سم" or "طول: 170،"
        height_match = re.search(r'طول[:\s]*(\d+(?:[.,]\d+)?)\s*(?:سم|cm)', t, re.IGNORECASE)
        if height_match:
            return float(height_match.group(1).replace(',', '.'))

        # As a last attempt, pick a plausible height number if present (100-250 cm)
        nums = re.findall(r'(\d+(?:[.,]\d+)?)', t)
        for n in nums:
            try:
                val = float(n.replace(',', '.'))
            except:
                continue
            if 100 <= val <= 250:
                return val

        return 170.0

    def _normalize_numbers(self, text: str) -> str:
        """Convert Arabic-Indic digits to Western digits, normalize commas to dots, and collapse whitespace.

        This helps parsing inputs like '٦٠ كجم', '165 سم', '60,5 kg' or Arabic labels.
        """
        if not text:
            return ""

        # Translate Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) to Western digits
        arabic_indic = '٠١٢٣٤٥٦٧٨٩'
        western = '0123456789'
        trans = str.maketrans(arabic_indic, western)
        try:
            txt = text.translate(trans)
        except Exception:
            txt = text

        # Replace Arabic comma (،) with regular comma first, then replace comma with dot for decimal numbers
        # But preserve Arabic comma for pattern matching - we'll handle it in regex patterns
        # For now, just normalize regular commas to dots for decimals
        # Note: Arabic comma (،) will be handled in regex patterns directly

        # Normalize whitespace
        txt = re.sub(r'\s+', ' ', txt).strip()

        return txt
    
    def _determine_activity_level(self, text: str) -> str:
        if re.search(r'\bactive\b|\bexercise\b|\bworkout\b', text, re.IGNORECASE):
            return "high"
        elif re.search(r'\bmoderate\b|\bregular\b', text, re.IGNORECASE):
            return "moderate"
        return "low"
    
    def _extract_dietary_restrictions(self, text: str) -> List[str]:
        restrictions = []
        if re.search(r'\bvegetarian\b', text, re.IGNORECASE):
            restrictions.append("vegetarian")
        if re.search(r'\bvegan\b', text, re.IGNORECASE):
            restrictions.append("vegan")
        if re.search(r'\bgluten.free\b', text, re.IGNORECASE):
            restrictions.append("gluten_free")
        if re.search(r'\bdairy.free\b', text, re.IGNORECASE):
            restrictions.append("dairy_free")
        return restrictions
    
    def _extract_goals(self, text: str) -> List[str]:
        goals = []
        if re.search(r'\bweight\s*loss\b|\blose\s*weight\b', text, re.IGNORECASE):
            goals.append("weight_loss")
        if re.search(r'\bweight\s*gain\b|\bgain\s*weight\b', text, re.IGNORECASE):
            goals.append("weight_gain")
        if re.search(r'\bmuscle\s*building\b|\bbuild\s*muscle\b', text, re.IGNORECASE):
            goals.append("muscle_building")
        if re.search(r'\bmaintain\b|\bmaintenance\b', text, re.IGNORECASE):
            goals.append("maintenance")
        return goals if goals else ["maintenance"]
    
    def _extract_medical_conditions(self, text: str) -> List[str]:
        """استخراج الأمراض من النص"""
        conditions = []
        text_lower = text.lower()
        
        # البحث عن الأمراض في قاعدة البيانات
        condition_keywords = {
            "diabetes": [r'\bdiabetes\b', r'\bسكري\b', r'\bسكر\b', r'\bdiabetic\b'],
            "hypertension": [r'\bhypertension\b', r'\bhigh\s*blood\s*pressure\b', r'\bضغط\s*دم\b', r'\bضغط\s*عالي\b'],
            "heart_disease": [r'\bheart\s*disease\b', r'\bcardiac\b', r'\bأمراض\s*قلب\b', r'\bقلب\b'],
            "kidney_disease": [r'\bkidney\s*disease\b', r'\brenal\b', r'\bأمراض\s*كلى\b', r'\bكلى\b'],
            "celiac": [r'\bceliac\b', r'\bgluten\s*sensitivity\b', r'\bحساسية\s*جلوتين\b'],
            "ibs": [r'\bibs\b', r'\birritable\s*bowel\b', r'\bقولون\s*عصبي\b', r'\bمتلازمة\s*قولون\b']
        }
        
        for condition_key, patterns in condition_keywords.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    if condition_key not in conditions:
                        conditions.append(condition_key)
                    break
        
        return conditions
    
    def calculate_calorie_needs(self, profile: UserProfile) -> int:
        # معادلة Harris-Benedict
        if profile.gender.lower() == "male":
            bmr = 88.362 + (13.397 * profile.weight) + (4.799 * profile.height) - (5.677 * profile.age)
        else:
            bmr = 447.593 + (9.247 * profile.weight) + (3.098 * profile.height) - (4.330 * profile.age)
        
        # مضاعف النشاط
        activity_multipliers = {
            "low": 1.2,
            "moderate": 1.55,
            "high": 1.725
        }
        
        tdee = bmr * activity_multipliers.get(profile.activity_level, 1.55)
        
        # تعديل حسب الأهداف
        if "weight_loss" in profile.goals:
            tdee *= 0.8  # عجز 20%
        elif "weight_gain" in profile.goals:
            tdee *= 1.2  # فائض 20%
        
        return int(tdee)
    
    def generate_enhanced_meal_plan(self, profile: UserProfile, days: int = 7) -> Dict:
        calorie_needs = self.calculate_calorie_needs(profile)
        current_season = self._get_current_season()
        
        meal_plan = {
            "user_profile": profile,
            "daily_calories": calorie_needs,
            "season": current_season,
            "days": []
        }
        
        for day in range(days):
            daily_plan = self._generate_daily_plan(profile, calorie_needs, current_season, day)
            meal_plan["days"].append(daily_plan)
        
        return meal_plan
    
    def _get_current_season(self) -> str:
        """تحديد الموسم الحالي"""
        month = datetime.now().month
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "winter"
    
    def _generate_daily_plan(self, profile: UserProfile, calories: int, season: str, day_index: int = 0) -> Dict:
        # توزيع السعرات على الوجبات
        meal_distribution = {
            "breakfast": 0.25,
            "lunch": 0.35,
            "dinner": 0.30,
            "snacks": 0.10
        }
        
        daily_plan = {
            "date": (datetime.now() + timedelta(days=day_index)).strftime("%Y-%m-%d"),
            "total_calories": calories,
            "meals": {}
        }
        
        for meal_type, ratio in meal_distribution.items():
            meal_calories = int(calories * ratio)
            meal = self._generate_meal(meal_type, meal_calories, profile, season)
            daily_plan["meals"][meal_type] = meal
        
        return daily_plan
    
    def _generate_meal(self, meal_type: str, calories: int, profile: UserProfile, season: str) -> Dict:
        # اختيار وصفة مناسبة
        available_recipes = self.recipe_database.get(meal_type, [])
        
        # تصفية حسب القيود الغذائية والأمراض
        filtered_recipes = self._filter_by_dietary_restrictions(
            available_recipes, 
            profile.dietary_restrictions,
            profile.medical_conditions
        )
        
        # اختيار وصفة عشوائية
        if filtered_recipes:
            selected_recipe_data = random.choice(filtered_recipes)
        elif available_recipes:
            selected_recipe_data = available_recipes[0]
        else:
            selected_recipe_data = {"name": f"{meal_type.title()} meal", "image": "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400"}
        
        # استخراج اسم الوصفة والصورة
        if isinstance(selected_recipe_data, dict):
            selected_recipe_name = selected_recipe_data["name"]
            recipe_image = selected_recipe_data["image"]
        else:
            selected_recipe_name = selected_recipe_data
            recipe_image = "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400"
        
        # إضافة مكونات موسمية
        seasonal_ingredients = self.seasonal_foods.get(season, [])
        
        # إضافة مكونات مناسبة للأمراض
        preferred_ingredients = []
        medical_notes = []
        if profile.medical_conditions:
            for condition in profile.medical_conditions:
                if condition in self.medical_conditions_db:
                    condition_data = self.medical_conditions_db[condition]
                    preferred = condition_data.get("prefer", [])
                    preferred_ingredients.extend(preferred[:2])  # أول مكونين مفضلين
                    medical_notes.append(condition_data.get("notes", ""))
        
        meal = {
            "name": selected_recipe_name,
            "image": recipe_image,
            "estimated_calories": calories,
            "ingredients": self._generate_ingredients(selected_recipe_name, seasonal_ingredients, preferred_ingredients),
            "nutritional_info": self._calculate_nutrition(selected_recipe_name, calories, profile),
            "seasonal_ingredients": seasonal_ingredients[:3],  # أول 3 مكونات موسمية
            "medical_considerations": medical_notes if medical_notes else None
        }
        
        return meal
    def _generate_ingredients(self, recipe_name: str, seasonal_ingredients: List[str], preferred_ingredients: List[str]) -> List[str]:
     ingredients = []

    # مكونات أساسية افتراضية بناءً على اسم الوصفة
     base_ingredients = self.recipe_database.get(recipe_name, [])
     ingredients.extend(base_ingredients[:4])  # ناخذ أول 4 فقط

    # نضيف مكونين موسميين
     if seasonal_ingredients:
        ingredients.extend(seasonal_ingredients[:2])

    # نضيف مكونات مناسبة لمرض المستخدم
     if preferred_ingredients:
        ingredients.extend(preferred_ingredients[:2])

     return ingredients[:6]
    
    def _filter_by_dietary_restrictions(self, recipes: List, restrictions: List[str], medical_conditions: List[str] = None) -> List:
        if not restrictions and not medical_conditions:
            return recipes
        
        filtered = []
        for recipe in recipes:
            # استخراج اسم الوصفة
            if isinstance(recipe, dict):
                recipe_name = recipe["name"]
            else:
                recipe_name = recipe
            
            recipe_lower = recipe_name.lower()
            skip_recipe = False
            
            # فحص القيود الغذائية
            if "vegetarian" in restrictions and any(meat in recipe_lower for meat in ["دجاج", "لحم", "سمك", "سلمون"]):
                skip_recipe = True
            if "vegan" in restrictions and any(animal in recipe_lower for animal in ["بيض", "جبنة", "حليب", "زبادي"]):
                skip_recipe = True
            if "gluten_free" in restrictions and any(gluten in recipe_lower for gluten in ["خبز", "معكرونة", "كعك", "قمح"]):
                skip_recipe = True
            
            # فحص الأمراض
            if medical_conditions:
                for condition in medical_conditions:
                    if condition in self.medical_conditions_db:
                        condition_data = self.medical_conditions_db[condition]
                        # تجنب الأطعمة الممنوعة للمرض
                        avoid_foods = condition_data.get("avoid", [])
                        for avoid_food in avoid_foods:
                            if avoid_food.lower() in recipe_lower:
                                skip_recipe = True
                                break
                        if skip_recipe:
                            break
            
            if not skip_recipe:
                filtered.append(recipe)
        
        return filtered if filtered else recipes
    
    
    def _calculate_nutrition(self, recipe: str, calories: int, profile: UserProfile = None) -> Dict:
        # توزيع السعرات على المغذيات الكبرى (قيم افتراضية)
        protein_ratio = 0.25
        carb_ratio = 0.45
        fat_ratio = 0.30
        
        # تعديل النسب حسب الأمراض
        if profile and profile.medical_conditions:
            for condition in profile.medical_conditions:
                if condition == "diabetes":
                    # تقليل الكربوهيدرات للسكري
                    carb_ratio = 0.35
                    protein_ratio = 0.30
                    fat_ratio = 0.35
                elif condition == "kidney_disease":
                    # تقليل البروتين لأمراض الكلى
                    protein_ratio = 0.15
                    carb_ratio = 0.55
                    fat_ratio = 0.30
                elif condition == "heart_disease":
                    # تقليل الدهون لأمراض القلب
                    fat_ratio = 0.20
                    carb_ratio = 0.50
                    protein_ratio = 0.30
        
        nutrition = {
            "calories": calories,
            "protein_g": round(calories * protein_ratio / 4),
            "carbs_g": round(calories * carb_ratio / 4),
            "fat_g": round(calories * fat_ratio / 9),
            "fiber_g": round(calories * 0.02),
            "sugar_g": round(calories * 0.05)
        }
        
        # تقليل السكر للأمراض التي تتطلب ذلك
        if profile and profile.medical_conditions:
            if "diabetes" in profile.medical_conditions:
                nutrition["sugar_g"] = round(calories * 0.01)  # تقليل السكر بشكل كبير
        
        return nutrition
    
    def create_enhanced_dataset(self, original_dataset) -> Dict:
        enhanced_data = {
            "train": [],
            "validation": []
        }
        
        for split in ["train", "validation"]:
            if split in original_dataset:
                for example in original_dataset[split]:
                    enhanced_input = self._enhance_input(example["input"])
                    enhanced_output = self._enhance_output(example["output"])
                    
                    enhanced_data[split].append({
                        "input": enhanced_input,
                        "output": enhanced_output
                    })
        
        return enhanced_data
    
    def _enhance_input(self, input_text: str) -> str:
        enhanced = f"""
        User Request: {input_text}
        Current Date: {datetime.now().strftime('%Y-%m-%d')}
        Season: {self._get_current_season()}
        Context: Please provide a detailed, personalized meal plan with specific recipes, nutritional information, preparation instructions, and shopping list.
        """
        return enhanced.strip()
    
    def _enhance_output(self, output_text: str) -> str:
        # إضافة معلومات غذائية إضافية
        enhanced = f"""
        {output_text}
        
        Additional Information:
        - Preparation time for each meal
        - Nutritional breakdown per meal
        - Seasonal ingredient recommendations
        - Shopping list with quantities
        - Meal prep tips and storage instructions
        """
        return enhanced.strip()
    
    def _find_recipe_image(self, meal_name: str):
     """إيجاد صورة الوصفة من قاعدة البيانات إذا كانت موجودة."""
     name = meal_name.lower().strip()

     for category, recipes in self.recipe_database.items():
        for recipe in recipes:
            rname = recipe.get("name", "").lower().strip()
            if rname == name:
                return recipe.get("image")

     return None
    
    def _generate_recipe_tags(self, recipe):
       tags = []
       ingredients = recipe.get("ingredients", [])
       name = recipe.get("name", "")

    # --- التاقات العامة ---
       if any(w in name for w in ["تمر", "مكسرات", "لوز", "فواكه"]):
        tags.append("وجبة خفيفة")

       if any(w in name for w in ["خبز", "قمح", "مكرونة"]):
        tags.append("يحتوي جلوتين")
       else:
        tags.append("خالي جلوتين")

    # --- استخدام قاعدة بيانات الأمراض الصحيحة ---
       ing_text = " ".join(ingredients)

       for disease, rules in self.medical_conditions_db.items():
        # تجنب الأطعمة للمصابين
           has_avoid = any(bad in ing_text for bad in rules.get("avoid", []))
           has_prefer = any(good in ing_text for good in rules.get("prefer", []))

           if has_avoid:
              tags.append(f"غير مناسب لمرضى {rules['name_ar']}")
              continue  # ❗ لا نسمح بإضافة "مناسب" لو فيه avoid

           if has_prefer:
              tags.append(f"مناسب لمرضى {rules['name_ar']}")

       return list(set(tags))
    
    def _attach_tags_to_recipes(self):
     for category, recipes in self.recipe_database.items():
        for recipe in recipes:

            # لا نحسب nutrition من المكونات
            # نأخذ nutrition الموجود أساساً داخل recipe_database
            if "nutrition" not in recipe:
                recipe["nutrition"] = {
                    "calories": recipe.get("calories", 0),
                    "protein": recipe.get("protein", 0),
                    "carbs": recipe.get("carbs", 0),
                    "fat": recipe.get("fat", 0)
                }

            # نحدد السعرات
            if "calories" not in recipe:
                recipe["calories"] = recipe["nutrition"].get("calories", 0)

            # التاقات
            recipe["tags"] = self._generate_recipe_tags(recipe)

def main():
    enhancer = DataEnhancer()
    
    # اختبار تحسين ملف المستخدم
    basic_info = "Female, 28 years old, 60kg, 165cm, vegetarian, active lifestyle"
    profile = enhancer.enhance_user_profile(basic_info)
    
    print("👤 ملف المستخدم المحسن:")
    print(f"العمر: {profile.age}")
    print(f"الجنس: {profile.gender}")
    print(f"الوزن: {profile.weight} kg")
    print(f"الطول: {profile.height} cm")
    print(f"مستوى النشاط: {profile.activity_level}")
    print(f"القيود الغذائية: {profile.dietary_restrictions}")
    print(f"الأهداف: {profile.goals}")
    
    # حساب احتياجات السعرات
    calories = enhancer.calculate_calorie_needs(profile)
    print(f"\n احتياجات السعرات اليومية: {calories} سعرة حرارية")
    
    # توليد خطة وجبات
    meal_plan = enhancer.generate_enhanced_meal_plan(profile, days=3)
    print(f"\n خطة الوجبات لمدة 3 أيام:")
    print(json.dumps(meal_plan, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
