"""
Opus-MT Fine-tuning Script for Star Citizen Localization
支持从 INI 文件加载数据集并进行分类优先级训练
"""

import os
import re
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# ==================== 配置区域 ====================
@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "Helsinki-NLP/opus-mt-zh-en"  # 可更换为其他模型
    
    # 数据集配置
    dataset_folder: str = "dataset"
    source_file: str = "chinese_(simplified).ini"  # 源语言文件
    target_file: str = "english.ini"  # 目标语言文件
    source_lang: str = "zh"  # 源语言代码
    target_lang: str = "en"  # 目标语言代码
    
    # ScWeb 数据配置
    scweb_folder: str = "dataset/ScWeb_Chinese_Translate"  # ScWeb 数据文件夹
    scweb_mode: str = "off"  # off, k2v (key->value, en->zh), v2k (value->key, zh->en)
    
    # 训练配置
    output_dir: str = "./results"
    max_length: int = 128
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    save_steps: int = 1000
    eval_steps: int = 500
    
    # 数据清洗配置
    max_text_length: int = 100  # 优先训练短文本
    min_text_length: int = 2
    
    # 优先级配置（是否启用分类训练）
    use_priority_training: bool = True
    priority_ratio: float = 0.7  # 优先类别占比

    # 测试配置
    fixed_sentences: List[str] = field(default_factory=list)
    templates: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_path: str) -> "TrainingConfig":
        """从 JSON 文件加载配置"""
        if not os.path.exists(json_path):
            return cls()
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 过滤未知的配置项
            valid_keys = cls.__dataclass_fields__.keys()
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            
            print(f"Loaded config from {json_path}")
            return cls(**filtered_dict)
        except Exception as e:
            print(f"Error loading config from {json_path}: {e}")
            return cls()

    def save_to_json(self, json_path: str):
        """保存配置到 JSON 文件"""
        config_dict = {k: v for k, v in self.__dict__.items()}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

# ==================== 数据清洗和分类 ====================
class DataCleaner:
    """数据清洗器"""
    
    # 复杂占位符模式（需要完全过滤的）
    COMPLEX_PLACEHOLDER_PATTERNS = [
        r'~mission\([^)]+\)',  # ~mission(location|address)
        r'~\w+\([^)]+\)',  # ~function(params)
        r'\{[^}]*\|[^}]*\}',  # {option1|option2}
        r'<%[^>]+%>',  # <%code%>
    ]
    
    # 简单占位符模式（可以容忍的）
    SIMPLE_PLACEHOLDER_PATTERNS = [
        r'%\w+',  # %VAR
        r'\$\w+',  # $VAR
        r'@\w+',  # @VAR
    ]
    
    @staticmethod
    def has_complex_placeholder(text: str) -> bool:
        """检查是否包含复杂占位符"""
        if not text:
            return False
        
        for pattern in DataCleaner.COMPLEX_PLACEHOLDER_PATTERNS:
            if re.search(pattern, text):
                return True
        return False
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清洗文本（仅用于已验证的文本）"""
        if not text:
            return ""
        
        # 去除首尾空格
        text = text.strip()
        
        # 移除引号
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        
        return text
    
    @staticmethod
    def is_valid_pair(source: str, target: str, min_len: int = 2, max_len: int = 100) -> bool:
        """验证文本对是否有效"""
        # 检查是否包含复杂占位符 - 直接拒绝
        if DataCleaner.has_complex_placeholder(source) or \
           DataCleaner.has_complex_placeholder(target):
            return False
        
        # 清理后检查
        source_clean = source.strip()
        target_clean = target.strip()
        
        # 检查长度
        if len(source_clean) < min_len or len(target_clean) < min_len:
            return False
        
        if len(source_clean) > max_len or len(target_clean) > max_len:
            return False
        
        # 检查是否为空
        if not source_clean or not target_clean:
            return False
        
        # 检查简单占位符数量（不超过2个）
        simple_placeholder_count = 0
        for pattern in DataCleaner.SIMPLE_PLACEHOLDER_PATTERNS:
            simple_placeholder_count += len(re.findall(pattern, source_clean))
            simple_placeholder_count += len(re.findall(pattern, target_clean))
        
        if simple_placeholder_count > 2:
            return False
        
        return True


class CategoryClassifier:
    """分类器 - 基于 Star Citizen 的键名规则"""
    
    # 优先类别（没有 _opt 后缀的）
    PRIORITY_CATEGORIES = {
        "location": [
            r"^Bacchus(?!.*_Desc).*",
            r"^Cano(?!.*_Desc).*",
            r"^Castra(?!.*_Desc).*",
            r"^Delamar(?!.*_Desc).*",
            r"^Ellis(?!.*_Desc).*",
            r"^Goss(?!.*_Desc).*",
            r"^Hadrian(?!.*_Desc).*",
            r"^Levski_Shop_Teach(?!.*_Desc).*",
            r"^Magnus(?!.*_Desc).*",
            r"^Nyx(?!.*_Desc).*",
            r"^Oso(?!.*_Desc).*",
            r"^Pyro(?!.*_Desc).*",
            r"^Stanton(?!.*_Desc).*",
            r"^Taranis(?!.*_Desc).*",
            r"^Tarpits(?!.*_Desc).*",
            r"^Tayac(?!.*_Desc).*",
            r"^Terra(?!.*_Desc).*",
            r"^Virgil(?!.*_Desc).*",
        ],
        "thing": [
            r"^item_Name.*",
        ],
        "vehicle": [
            r"^vehicle_Name.*",
        ],
        "subtitle": [
            r"^DXSH_",
            r"^Dlg_SC_.*",
            r"^FW22_NT_Datapad_.*",
            r"^FleetWeek2950_.*",
            r"^GenResponse_.*",
            r"^GenericLanding_.*",
            r"^IT_Shared_.*",
            r"^Imperilled_.*",
            r"^MKTG_CUSTOMS1_CV_Access_.*",
            r"^PH_PU_.*",
            r"^PU_.*",
            r"^Pacheco_.*",
            r"^SC_ac_.*",
            r"^SC_lz_.*",
            r"^SM_SIMANN1_.*",
            r"^contract_.*",
            r"^covalex_.*",
            r"^covalexrand_.*",
            r"^covalexspec_.*",
        ],
    }
    
    @staticmethod
    def classify_key(key: str) -> Tuple[str, bool]:
        """
        分类键名
        返回: (category_name, is_priority)
        """
        for category, patterns in CategoryClassifier.PRIORITY_CATEGORIES.items():
            for pattern in patterns:
                if re.match(pattern, key):
                    return category, True
        
        return "other", False


# ==================== 数据集加载 ====================
class IniDatasetLoader:
    """INI 数据集加载器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.cleaner = DataCleaner()
        self.classifier = CategoryClassifier()
    
    def load_ini_file(self, filepath: str) -> Dict[str, str]:
        """加载 INI 文件（支持无 section 头部的格式）"""
        data = {}
        
        # 尝试不同的编码
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    for line_num, line in enumerate(f, 1):
                        # 移除 BOM 和空格
                        line = line.strip()
                        
                        # 跳过空行和注释
                        if not line or line.startswith(';') or line.startswith('#') or line.startswith('//'):
                            continue
                        
                        # 查找等号
                        if '=' in line:
                            # 分割键值对（只分割第一个等号）
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = parts[1].strip()
                                
                                # 移除值两端的引号（如果有）
                                if value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]
                                elif value.startswith("'") and value.endswith("'"):
                                    value = value[1:-1]
                                
                                data[key] = value
                
                # 成功读取，跳出编码循环
                if data:
                    break
                    
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Warning: Error reading file with {encoding} encoding: {e}")
                continue
        
        if not data:
            raise ValueError(f"Failed to load data from {filepath}. Please check file format and encoding.")
        
        return data
    
    def load_scweb_data(self) -> List[Dict]:
        """
        加载 ScWeb_Chinese_Translate 数据
        根据 scweb_mode 决定数据方向:
        - k2v: key->value (英文->中文)
        - v2k: value->key (中文->英文)
        返回: 样本列表
        """
        if self.config.scweb_mode == "off":
            return []
        
        scweb_folder = self.config.scweb_folder
        if not os.path.exists(scweb_folder):
            print(f"Warning: ScWeb folder not found: {scweb_folder}")
            return []
        
        samples = []
        filtered_count = 0
        
        # 遍历所有 JSON 文件
        for filename in os.listdir(scweb_folder):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(scweb_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for key, value in data.items():
                    # 根据模式决定 source 和 target
                    if self.config.scweb_mode == "k2v":
                        # key->value: 英文->中文
                        source_text = self.cleaner.clean_text(key)
                        target_text = self.cleaner.clean_text(value)
                    elif self.config.scweb_mode == "v2k":
                        # value->key: 中文->英文
                        source_text = self.cleaner.clean_text(value)
                        target_text = self.cleaner.clean_text(key)
                    else:
                        continue
                    
                    # 验证文本对
                    if not self.cleaner.is_valid_pair(
                        source_text,
                        target_text,
                        self.config.min_text_length,
                        self.config.max_text_length
                    ):
                        filtered_count += 1
                        continue
                    
                    sample = {
                        "key": f"scweb_{filename}_{key[:20]}",
                        "source": source_text,
                        "target": target_text,
                        "source_length": len(source_text),
                        "target_length": len(target_text),
                        "type": "scweb",
                        "category": "scweb"
                    }
                    samples.append(sample)
                    
            except Exception as e:
                print(f"Warning: Error loading ScWeb file {filepath}: {e}")
                continue
        
        print(f"\nScWeb Dataset Statistics:")
        print(f"  Mode: {self.config.scweb_mode}")
        print(f"  Valid samples: {len(samples)}")
        print(f"  Filtered samples: {filtered_count}")
        
        return samples
    
    def load_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """
        加载数据集（包含原始数据 + ScWeb 数据）
        返回: (priority_data, other_data)
        """
        source_path = os.path.join(self.config.dataset_folder, self.config.source_file)
        target_path = os.path.join(self.config.dataset_folder, self.config.target_file)
        
        print(f"Loading source file: {source_path}")
        print(f"Loading target file: {target_path}")
        
        source_data = self.load_ini_file(source_path)
        target_data = self.load_ini_file(target_path)
        
        priority_samples = []
        other_samples = []
        filtered_count = 0
        
        # 匹配键值对
        for key in source_data.keys():
            if key not in target_data:
                continue
            
            source_text = self.cleaner.clean_text(source_data[key])
            target_text = self.cleaner.clean_text(target_data[key])
            
            # 验证文本对
            if not self.cleaner.is_valid_pair(
                source_text, 
                target_text,
                self.config.min_text_length,
                self.config.max_text_length
            ):
                filtered_count += 1
                continue
            
            sample = {
                "key": key,
                "source": source_text,
                "target": target_text,
                "source_length": len(source_text),
                "target_length": len(target_text),
                "type": "original"
            }
            
            # 分类
            category, is_priority = self.classifier.classify_key(key)
            sample["category"] = category
            
            if is_priority:
                priority_samples.append(sample)
            else:
                other_samples.append(sample)
        
        print(f"\nOriginal Dataset Statistics:")
        print(f"  Valid samples: {len(priority_samples) + len(other_samples)}")
        print(f"  Filtered samples: {filtered_count}")
        print(f"  Priority samples: {len(priority_samples)}")
        print(f"  Other samples: {len(other_samples)}")
        
        # 加载 ScWeb 数据
        scweb_samples = self.load_scweb_data()
        if scweb_samples:
            other_samples.extend(scweb_samples)
        
        print(f"\nFinal Dataset Statistics:")
        print(f"  Total samples: {len(priority_samples) + len(other_samples)}")
        print(f"  Priority samples: {len(priority_samples)}")
        print(f"  Other samples (incl. ScWeb): {len(other_samples)}")
        
        # 统计各类别数量
        if priority_samples:
            categories = {}
            for sample in priority_samples:
                cat = sample["category"]
                categories[cat] = categories.get(cat, 0) + 1
            print(f"\nPriority Categories:")
            for cat, count in sorted(categories.items()):
                print(f"    {cat}: {count}")
        
        return priority_samples, other_samples


# ==================== PyTorch 数据集 ====================
class TranslationDataset(Dataset):
    """翻译数据集"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize source text
        model_inputs = self.tokenizer(
            sample["source"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize target text using text_target parameter
        labels = self.tokenizer(
            text_target=sample["target"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs


# ==================== 训练器 ====================
class OpusMTTrainer:
    """Opus-MT 训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\nInitializing trainer...")
        print(f"  Device: {self.device}")
        if self.device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 加载模型和分词器
        print(f"  Loading model: {config.model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(config.model_name)
        self.model = MarianMTModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        
        # 数据加载器
        self.loader = IniDatasetLoader(config)
    
    def prepare_datasets(self):
        """准备训练和验证数据集"""
        priority_data, other_data = self.loader.load_dataset()
        
        # 根据配置决定数据集组成
        if self.config.use_priority_training and priority_data:
            # 优先训练模式：按比例混合
            num_priority = int(len(priority_data) * self.config.priority_ratio)
            num_other = len(priority_data) - num_priority
            
            # 随机采样
            random.shuffle(priority_data)
            random.shuffle(other_data)
            
            train_data = priority_data[:num_priority] + other_data[:num_other]
            random.shuffle(train_data)
            
            print(f"\nPriority Training Mode:")
            print(f"  Priority samples: {num_priority}")
            print(f"  Other samples: {num_other}")
        else:
            # 全部数据训练
            train_data = priority_data + other_data
            random.shuffle(train_data)
            print(f"\nStandard Training Mode:")
            print(f"  Total samples: {len(train_data)}")
        
        # 分割训练集和验证集
        split_idx = int(len(train_data) * 0.9)
        train_samples = train_data[:split_idx]
        val_samples = train_data[split_idx:]
        
        print(f"\nDataset Split:")
        print(f"  Training: {len(train_samples)}")
        print(f"  Validation: {len(val_samples)}")
        
        # 创建 PyTorch 数据集
        self.train_dataset = TranslationDataset(
            train_samples, 
            self.tokenizer, 
            self.config.max_length
        )
        self.val_dataset = TranslationDataset(
            val_samples, 
            self.tokenizer, 
            self.config.max_length
        )
        
        return self.train_dataset, self.val_dataset
    
    def train(self):
        """开始训练"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        # 准备数据集
        train_dataset, val_dataset = self.prepare_datasets()
        
        # 训练参数
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.config.num_epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),  # 混合精度训练
            save_steps=self.config.save_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_accumulation_steps=2,  # RTX 4090 可以减少这个值
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # 创建训练器
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        
        # 开始训练
        print("\nTraining started...")
        trainer.train()
        
        # 保存最终模型
        print("\nSaving final model...")
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        print(f"\nTraining completed!")
        print(f"Final model saved to: {final_model_path}")
        
        return trainer
    
    def test_translation(self, texts: List[str], model_path: Optional[str] = None):
        """测试翻译"""
        if model_path:
            model = MarianMTModel.from_pretrained(model_path)
            tokenizer = MarianTokenizer.from_pretrained(model_path)
        else:
            model = self.model
            tokenizer = self.tokenizer
        
        model.to(self.device)
        model.eval()
        
        print("\n" + "="*50)
        print("Translation Test")
        print("="*50)
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=self.config.max_length)
            
            translation = tokenizer.decode(translated[0], skip_special_tokens=True)
            print(f"\nSource: {text}")
            print(f"Translation: {translation}")


# ==================== 主函数 ====================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Opus-MT Training Script")
    parser.add_argument("--config", type=str, default="config/zh-en.json", help="Path to configuration file")
    args = parser.parse_args()
    
    # 创建配置
    if os.path.exists(args.config):
        config = TrainingConfig.from_json(args.config)
    else:
        print(f"Config file not found: {args.config}, using defaults.")
        config = TrainingConfig()
    
    # 保存配置 (保存一份副本到输出目录)
    os.makedirs(config.output_dir, exist_ok=True)
    config.save_to_json(os.path.join(config.output_dir, "training_config.json"))
    
    # 创建训练器
    trainer = OpusMTTrainer(config)
    
    # 开始训练
    trainer.train()
    
    # 测试翻译
    # 使用配置中的测试句子和模板
    test_texts = []
    if config.fixed_sentences:
        test_texts.extend(config.fixed_sentences)
    
    if config.templates:
        # 仅取前几个模板作为示例
        test_texts.extend(config.templates[:5])
    
    # 如果配置中没有，则跳过测试
    if not test_texts:
        print("\nSkipping translation test (no test sentences configured).")
        return

    trainer.test_translation(
        test_texts,
        model_path=os.path.join(config.output_dir, "final_model")
    )


if __name__ == "__main__":
    main()
