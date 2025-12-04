"""
محسن الأداء للنموذج
Performance Optimizer for Enhanced Meal Planner
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import psutil
import gc
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """محسن الأداء للنموذج مع تقنيات متقدمة"""
    
    def __init__(self, model, device: str = "auto"):
        """
        تهيئة محسن الأداء
        
        Args:
            model: النموذج المراد تحسينه
            device: الجهاز المستخدم
        """
        self.model = model
        self.device = self._setup_device(device)
        self.optimizer = None
        self.scheduler = None
        self.mixed_precision = False
        self.gradient_accumulation_steps = 1
        
        # إحصائيات الأداء
        self.performance_stats = {
            "training_time": 0,
            "memory_usage": [],
            "gpu_utilization": [],
            "throughput": 0
        }
    
    def _setup_device(self, device: str) -> torch.device:
        """إعداد الجهاز المناسب"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def setup_mixed_precision(self, enabled: bool = True):
        """إعداد التدريب بخلط الدقة"""
        self.mixed_precision = enabled and self.device.type == "cuda"
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("✅ تم تفعيل التدريب بخلط الدقة (Mixed Precision)")
        else:
            logger.info("ℹ️ التدريب بخلط الدقة غير متاح أو معطل")
    
    def setup_gradient_accumulation(self, steps: int = 4):
        """إعداد تراكم التدرجات"""
        self.gradient_accumulation_steps = steps
        logger.info(f"✅ تم إعداد تراكم التدرجات لـ {steps} خطوات")
    
    def setup_optimizer(self, learning_rate: float = 3e-4, weight_decay: float = 0.01):
        """إعداد محسن متقدم"""
        # استخدام AdamW مع معاملات محسنة
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        logger.info(f"✅ تم إعداد محسن AdamW بمعدل تعلم {learning_rate}")
    
    def setup_scheduler(self, num_training_steps: int, warmup_steps: int = 100):
        """إعداد جدولة معدل التعلم"""
        from transformers import get_cosine_schedule_with_warmup
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"✅ تم إعداد جدولة معدل التعلم مع {warmup_steps} خطوات تسخين")
    
    def setup_dataloader_optimization(self, dataloader: DataLoader):
        """تحسين تحميل البيانات"""
        # تحسين عدد العمال
        if hasattr(dataloader, 'num_workers'):
            optimal_workers = min(psutil.cpu_count(), 8)
            dataloader.num_workers = optimal_workers
            logger.info(f"✅ تم تحسين عدد عمال تحميل البيانات إلى {optimal_workers}")
        
        # تحسين حجم الدفعة
        if hasattr(dataloader, 'batch_size'):
            current_batch_size = dataloader.batch_size
            if self.device.type == "cuda":
                # زيادة حجم الدفعة للـ GPU
                optimal_batch_size = min(current_batch_size * 2, 32)
                dataloader.batch_size = optimal_batch_size
                logger.info(f"✅ تم تحسين حجم الدفعة إلى {optimal_batch_size}")
    
    @contextmanager
    def memory_monitoring(self):
        """مراقبة استخدام الذاكرة"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            initial_cached = torch.cuda.memory_reserved()
            
            try:
                yield
            finally:
                final_memory = torch.cuda.memory_allocated()
                final_cached = torch.cuda.memory_reserved()
                
                memory_used = (final_memory - initial_memory) / 1024**3  # GB
                cached_used = (final_cached - initial_cached) / 1024**3  # GB
                
                self.performance_stats["memory_usage"].append({
                    "memory_used_gb": memory_used,
                    "cached_used_gb": cached_used,
                    "timestamp": time.time()
                })
                
                logger.info(f"💾 استخدام الذاكرة: {memory_used:.2f} GB (مخزن مؤقت: {cached_used:.2f} GB)")
        else:
            yield
    
    def optimize_model_for_inference(self):
        """تحسين النموذج للاستدلال"""
        logger.info("🔄 جاري تحسين النموذج للاستدلال...")
        
        # تحويل إلى وضع التقييم
        self.model.eval()
        
        # تحسين الذاكرة
        if self.device.type == "cuda":
            # تحويل إلى half precision
            self.model = self.model.half()
            
            # تحسين الذاكرة
            torch.cuda.empty_cache()
            
            # تفعيل cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # تجميد المعاملات
        for param in self.model.parameters():
            param.requires_grad = False
        
        logger.info("✅ تم تحسين النموذج للاستدلال")
    
    def optimize_model_for_training(self):
        """تحسين النموذج للتدريب"""
        logger.info("🔄 جاري تحسين النموذج للتدريب...")
        
        # تحويل إلى وضع التدريب
        self.model.train()
        
        # تفعيل التدرجات
        for param in self.model.parameters():
            param.requires_grad = True
        
        # تحسين الذاكرة للتدريب
        if self.device.type == "cuda":
            # استخدام float16 للتدريب
            self.model = self.model.half()
            
            # تحسين cuDNN
            torch.backends.cudnn.benchmark = True
        
        logger.info("✅ تم تحسين النموذج للتدريب")
    
    def train_step_optimized(self, batch: Dict, step: int) -> Dict:
        """خطوة تدريب محسنة"""
        start_time = time.time()
        
        with self.memory_monitoring():
            # نقل البيانات للجهاز
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(self.device)
            
            # التدريب بخلط الدقة
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                # تراكم التدرجات
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # تطبيق التدرجات
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # تحديث جدولة معدل التعلم
                    if self.scheduler:
                        self.scheduler.step()
            else:
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss / self.gradient_accumulation_steps
                
                # تراكم التدرجات
                loss.backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # تطبيق التدرجات
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # تحديث جدولة معدل التعلم
                    if self.scheduler:
                        self.scheduler.step()
        
        # حساب الإحصائيات
        step_time = time.time() - start_time
        throughput = batch["input_ids"].size(0) / step_time
        
        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "step_time": step_time,
            "throughput": throughput,
            "learning_rate": self.optimizer.param_groups[0]["lr"] if self.optimizer else 0
        }
    
    def generate_optimized(self, input_text: str, max_length: int = 200, 
                          temperature: float = 0.7, num_beams: int = 4) -> str:
        """توليد محسن للنص"""
        start_time = time.time()
        
        with self.memory_monitoring():
            # تحضير المدخلات
            inputs = self.model.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True
            ).to(self.device)
            
            # توليد النص مع تحسينات
            with torch.no_grad():
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=max_length,
                            temperature=temperature,
                            num_beams=num_beams,
                            no_repeat_ngram_size=2,
                            do_sample=True,
                            top_p=0.9,
                            top_k=50,
                            early_stopping=True,
                            pad_token_id=self.model.tokenizer.pad_token_id,
                            eos_token_id=self.model.tokenizer.eos_token_id,
                            use_cache=True  # استخدام التخزين المؤقت
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        num_beams=num_beams,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        early_stopping=True,
                        pad_token_id=self.model.tokenizer.pad_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id,
                        use_cache=True
                    )
            
            # فك تشفير النتيجة
            generated_text = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        logger.info(f"⚡ وقت التوليد: {generation_time:.2f} ثانية")
        
        return generated_text
    
    def batch_generate_optimized(self, input_texts: List[str], 
                                batch_size: int = 4) -> List[str]:
        """توليد دفعات محسن"""
        results = []
        
        # تقسيم النصوص إلى دفعات
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i + batch_size]
            
            with self.memory_monitoring():
                # تحضير الدفعة
                inputs = self.model.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # توليد النصوص
                with torch.no_grad():
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model.generate(
                                **inputs,
                                max_length=200,
                                temperature=0.7,
                                num_beams=4,
                                no_repeat_ngram_size=2,
                                do_sample=True,
                                early_stopping=True,
                                pad_token_id=self.model.tokenizer.pad_token_id,
                                eos_token_id=self.model.tokenizer.eos_token_id,
                                use_cache=True
                            )
                    else:
                        outputs = self.model.generate(
                            **inputs,
                            max_length=200,
                            temperature=0.7,
                            num_beams=4,
                            no_repeat_ngram_size=2,
                            do_sample=True,
                            early_stopping=True,
                            pad_token_id=self.model.tokenizer.pad_token_id,
                            eos_token_id=self.model.tokenizer.eos_token_id,
                            use_cache=True
                        )
                
                # فك تشفير النتائج
                batch_results = self.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(batch_results)
        
        return results
    
    def profile_model(self, input_text: str, num_runs: int = 10) -> Dict:
        """تحليل أداء النموذج"""
        logger.info(f"🔍 تحليل أداء النموذج مع {num_runs} تشغيل...")
        
        times = []
        memory_usage = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated()
            
            # تشغيل النموذج
            _ = self.generate_optimized(input_text)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                memory_usage.append((end_memory - start_memory) / 1024**3)  # GB
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # حساب الإحصائيات
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        
        profile_results = {
            "average_time": avg_time,
            "std_time": std_time,
            "min_time": np.min(times),
            "max_time": np.max(times),
            "average_memory_gb": avg_memory,
            "throughput_per_second": 1.0 / avg_time,
            "device": str(self.device),
            "mixed_precision": self.mixed_precision
        }
        
        logger.info(f"📊 نتائج التحليل:")
        logger.info(f"   متوسط الوقت: {avg_time:.3f} ± {std_time:.3f} ثانية")
        logger.info(f"   متوسط الذاكرة: {avg_memory:.2f} GB")
        logger.info(f"   الإنتاجية: {1.0/avg_time:.2f} نص/ثانية")
        
        return profile_results
    
    def cleanup_memory(self):
        """تنظيف الذاكرة"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        logger.info("🧹 تم تنظيف الذاكرة")
    
    def get_performance_summary(self) -> Dict:
        """ملخص أداء النموذج"""
        summary = {
            "device": str(self.device),
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer": type(self.optimizer).__name__ if self.optimizer else None,
            "scheduler": type(self.scheduler).__name__ if self.scheduler else None,
            "performance_stats": self.performance_stats
        }
        
        return summary

class OptimizedDataset(Dataset):
    """مجموعة بيانات محسنة"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenization محسن
        inputs = self.tokenizer(
            item["input"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            item["output"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }

def main():
    """اختبار محسن الأداء"""
    print("🚀 اختبار محسن الأداء...")
    
    # إنشاء نموذج وهمي للاختبار
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    # إنشاء محسن الأداء
    optimizer = PerformanceOptimizer(model)
    
    # إعداد التحسينات
    optimizer.setup_mixed_precision(enabled=True)
    optimizer.setup_gradient_accumulation(steps=4)
    optimizer.setup_optimizer()
    
    # تحسين النموذج للاستدلال
    optimizer.optimize_model_for_inference()
    
    # اختبار التوليد
    test_input = "Create a healthy meal plan for a 30-year-old woman"
    result = optimizer.generate_optimized(test_input)
    
    print(f"✅ النتيجة: {result}")
    
    # تحليل الأداء
    profile = optimizer.profile_model(test_input)
    print(f"📊 ملخص الأداء: {profile}")

if __name__ == "__main__":
    main()
