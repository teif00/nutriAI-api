import os
import torch
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMealPlanner:
    
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "auto"):
        self.device = self._setup_device(device)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.config = None
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        logger.info(f" تم تهيئة النموذج على الجهاز: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_tokenizer_and_config(self, vocab_size: int = 32000):
        logger.info(" جاري تحميل Tokenizer...")
        
        # استخدام T5 tokenizer محسن
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # تكوين محسن للنموذج
        self.config = AutoConfig.from_pretrained(self.model_name)
        
        # تحسين المعاملات
        self.config.update({
            "vocab_size": self.tokenizer.vocab_size,
            "d_model": 512,              # زيادة أبعاد التضمين
            "d_ff": 1024,                # زيادة أبعاد الطبقة المخفية
            "num_layers": 6,             # زيادة عدد الطبقات
            "num_heads": 8,              # زيادة عدد رؤوس الانتباه
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-6,
            "relative_attention_num_buckets": 32,
            "relative_attention_max_distance": 128,
            "decoder_start_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32
        })
        
        logger.info(" تم تحميل Tokenizer و Config بنجاح")
    
    def create_enhanced_model(self):
        logger.info(" جاري إنشاء النموذج المحسن...")
        
        if self.config is None:
            self.load_tokenizer_and_config()
        
        # إنشاء النموذج من التكوين
        self.model = AutoModelForSeq2SeqLM.from_config(self.config)
        self.model = self.model.to(self.device)
        
        # تحسين الذاكرة
        if self.device.type == "cuda":
            self.model = self.model.half()  # استخدام float16 لتوفير الذاكرة
        
        logger.info(" تم إنشاء النموذج المحسن بنجاح")
    
    def load_and_preprocess_data(self, dataset_name: str = "sridhar52/Augmented_Meal_Planner_data", 
                                sample_fraction: float = 0.8):
        """تحميل ومعالجة البيانات مع تحسينات"""
        logger.info(" جاري تحميل البيانات...")
        
        # تحميل البيانات
        dataset = load_dataset(dataset_name)
        logger.info(f" تم تحميل {len(dataset['train'])} عينة تدريب")
        
        # تحسين حجم البيانات
        if sample_fraction < 1.0:
            n_samples = int(len(dataset["train"]) * sample_fraction)
            dataset["train"] = dataset["train"].shuffle(seed=42).select(range(n_samples))
            logger.info(f" تم تقليل البيانات إلى {n_samples} عينة")
        
        # إنشاء مجموعة التحقق
        if "validation" not in dataset:
            val_size = min(500, len(dataset["train"]) // 5)
            dataset["validation"] = dataset["train"].select(range(val_size))
            dataset["train"] = dataset["train"].select(range(val_size, len(dataset["train"])))
            logger.info(f" تم إنشاء {val_size} عينة للتحقق")
        
        return dataset
    
    def enhanced_preprocessing(self, examples: Dict) -> Dict:
        inputs = []
        targets = []
        
        for i, (input_text, output_text) in enumerate(zip(examples["input"], examples["output"])):
            # إضافة معلومات إضافية للسياق
            enhanced_input = self._enhance_input_context(input_text, i)
            inputs.append(enhanced_input)
            targets.append(output_text)
        
        # Tokenization محسن
        model_inputs = self.tokenizer(
            inputs, 
            max_length=256,  # زيادة الطول الأقصى
            truncation=True, 
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            targets, 
            max_length=256, 
            truncation=True, 
            padding="max_length",
            return_tensors="pt"
        )
        
        # معالجة التسميات
        model_inputs["labels"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        
        return model_inputs
    
    def _enhance_input_context(self, input_text: str, index: int) -> str:
        # إضافة معلومات زمنية
        current_time = datetime.now()
        time_context = f"الوقت الحالي: {current_time.strftime('%Y-%m-%d %H:%M')}"
        
        # إضافة معلومات إضافية
        enhanced_context = f"""
        ملف المستخدم: {input_text}
        {time_context}
        رقم الطلب: {index}
        السياق: أنشئ خطة وجبات مفصلة ومخصصة مع معلومات غذائية باللغة العربية.
        """
        
        return enhanced_context.strip()
    
    def setup_training_args(self, output_dir: str = "./enhanced_mealplanner_model") -> Seq2SeqTrainingArguments:
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=3e-4,  # معدل تعلم محسن
            per_device_train_batch_size=8,  # زيادة حجم الدفعة
            per_device_eval_batch_size=4,
            num_train_epochs=8,  # زيادة عدد العصور
            weight_decay=0.01,
            warmup_steps=100,  # إضافة تسخين
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,  # تحسين تحميل البيانات
            logging_dir=f"{output_dir}/logs",
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            remove_unused_columns=False,
            label_smoothing_factor=0.1,  # إضافة تنعيم التسميات
            gradient_accumulation_steps=2,  # تراكم التدرجات
            lr_scheduler_type="cosine",  # جدولة معدل التعلم
            warmup_ratio=0.1,
        )
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        # فك تشفير التوقعات
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # حساب مقاييس بسيطة
        exact_match = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred.strip() == label.strip())
        exact_match_ratio = exact_match / len(decoded_preds)
        
        return {
            "exact_match": exact_match_ratio,
            "total_samples": len(decoded_preds)
        }
    
    def train_model(self, dataset, output_dir: str = "./enhanced_mealplanner_model"):
        logger.info(" بدء تدريب النموذج المحسن...")
        
        # معالجة البيانات
        tokenized_datasets = dataset.map(
            self.enhanced_preprocessing,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=4  # معالجة متوازية
        )
        
        # إعداد Data Collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model,
            padding=True
        )
        
        # إعداد معاملات التدريب
        training_args = self.setup_training_args(output_dir)
        
        # إعداد المدرب
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # بدء التدريب
        train_result = trainer.train()
        
        # حفظ النموذج
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(" تم تدريب النموذج بنجاح!")
        return train_result
    
    def generate_meal_plan(self, user_info: str, goal: str, 
                          max_length: int = 200, temperature: float = 0.7) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("يجب تحميل النموذج أولاً!")
        
        # تحسين الـ prompt
        enhanced_prompt = f"""
        ملف المستخدم: {user_info}
        الهدف: {goal}
        التاريخ الحالي: {datetime.now().strftime('%Y-%m-%d')}
        السياق: أنشئ خطة وجبات مفصلة ومخصصة مع وصفات محددة ومعلومات غذائية وتعليمات التحضير باللغة العربية.
        """
        
        # Tokenization
        inputs = self.tokenizer(
            enhanced_prompt, 
            return_tensors="pt", 
            max_length=256, 
            truncation=True
        ).to(self.device)
        
        # توليد النص
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_beams=4,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # فك تشفير النتيجة
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def batch_generate(self, user_requests: List[Tuple[str, str]]) -> List[str]:
        results = []
        for user_info, goal in user_requests:
            try:
                result = self.generate_meal_plan(user_info, goal)
                results.append(result)
            except Exception as e:
                logger.error(f"خطأ في توليد الخطة: {e}")
                results.append("عذراً، حدث خطأ في توليد خطة الوجبات.")
        
        return results

def main():
    print(" بدء تشغيل النموذج المحسن لتصميم الوجبات...")
    
    # إنشاء النموذج
    meal_planner = EnhancedMealPlanner()
    
    # تحميل Tokenizer و Config
    meal_planner.load_tokenizer_and_config()
    
    # إنشاء النموذج
    meal_planner.create_enhanced_model()
    
    # تحميل البيانات
    dataset = meal_planner.load_and_preprocess_data()
    
    # تدريب النموذج
    train_result = meal_planner.train_model(dataset)
    
    print(" تم تدريب النموذج المحسن بنجاح!")
    
    # اختبار النموذج
    test_result = meal_planner.generate_meal_plan(
        "Female, 28 years old, 60kg, vegetarian, active lifestyle",
        "Create a 1200-calorie vegetarian meal plan for weight loss"
    )
    
    print("\n خطة الوجبات المولدة:")
    print(test_result)

if __name__ == "__main__":
    main()

