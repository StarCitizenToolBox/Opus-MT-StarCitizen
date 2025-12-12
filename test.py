"""
测试翻译模型 - 使用真实游戏数据生成测试句子
从 INI 文件中提取地点、载具、物品等，生成常见的游戏场景对话
支持对比原始模型、微调模型和量化模型
支持交互式 CLI 模式
"""

import os
import re
import json
import random
import time
import sys
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from transformers import MarianMTModel, MarianTokenizer
import torch

try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: optimum 未安装，无法加载 ONNX 量化模型")

# ==================== 配置 ====================
@dataclass
class TestConfig:
    """测试配置"""
    model_name: str = "Helsinki-NLP/opus-mt-zh-en"  # 原始基础模型
    finetuned_model_path: str = "./results/final_model"  # 微调后的模型路径
    quantized_model_path: str = "./results/final_model/onnx"  # 量化模型路径（与原始 ONNX 在同一目录）
    
    dataset_folder: str = "dataset"
    source_file: str = "chinese_(simplified).ini"
    
    # 生成配置
    num_test_sentences: int = 20  # 生成测试句子数量
    max_length: int = 128
    
    # 对比配置
    compare_models: bool = True  # 是否进行模型对比
    load_base_model: bool = True  # 是否加载原始模型
    load_quantized: bool = True  # 是否加载量化模型

    # ScWeb 数据配置
    scweb_folder: str = "dataset/ScWeb_Chinese_Translate"  # ScWeb 数据文件夹
    scweb_mode: str = "off"  # off, k2v (key->value, en->zh), v2k (value->key, zh->en)
    scweb_test_count: int = 100  # 从 ScWeb 数据中抽取的测试句子数量

    # 测试数据配置
    fixed_sentences: List[str] = field(default_factory=list)
    templates: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_path: str) -> "TestConfig":
        """从 JSON 文件加载配置"""
        if not os.path.exists(json_path):
            return cls()
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            valid_keys = cls.__dataclass_fields__.keys()
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            
            print(f"Loaded config from {json_path}")
            return cls(**filtered_dict)
        except Exception as e:
            print(f"Error loading config from {json_path}: {e}")
            return cls()


# ==================== 数据提取器 ====================
class GameDataExtractor:
    """从 INI 文件提取游戏数据"""
    
    # 分类正则表达式（与 train.py 中的 PRIORITY_CATEGORIES 保持同步）
    CATEGORIES = {
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
            # 额外的地点匹配模式
            r"^(Crusader|ArcCorp|Hurston|microTech|Orison|Area18|Lorville|NewBabbage)(?!.*_Desc).*",
            r"^(Port_Olisar|GrimHex|Everus|Baijini|Seraphim).*",
            r".*(?:Station|Port|Outpost|Settlement)(?!.*_Desc).*",
        ],
        "vehicle": [
            r"^vehicle_Name.*",
        ],
        "item": [
            r"^item_Name.*",
        ],
        "thing": [  # 与 train.py 保持一致的别名
            r"^item_Name.*",
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
        "mission": [
            r".*(bounty|Bounty|mission|Mission|contract|Contract).*",
        ],
    }
    
    def __init__(self, ini_path: str):
        self.ini_path = ini_path
        self.data = self._load_ini()
        self.categorized_data = self._categorize_data()
    
    def _load_ini(self) -> Dict[str, str]:
        """加载 INI 文件"""
        data = {}
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
        
        for encoding in encodings:
            try:
                with open(self.ini_path, 'r', encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith(';') or line.startswith('#'):
                            continue
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            
                            # 过滤占位符过多和过长的文本
                            if value and len(value) < 50:
                                # 简单过滤包含太多占位符的
                                placeholder_count = len(re.findall(r'[%~@\$\{\[\<]', value))
                                if placeholder_count < 3:
                                    data[key] = value
                
                if data:
                    break
            except:
                continue
        
        return data
    
    def _categorize_data(self) -> Dict[str, List[str]]:
        """分类数据"""
        categorized = {cat: [] for cat in self.CATEGORIES.keys()}
        categorized["other"] = []
        
        for key, value in self.data.items():
            classified = False
            for category, patterns in self.CATEGORIES.items():
                for pattern in patterns:
                    if re.match(pattern, key, re.IGNORECASE):
                        categorized[category].append(value)
                        classified = True
                        break
                if classified:
                    break
            
            if not classified:
                categorized["other"].append(value)
        
        # 去重
        for cat in categorized:
            categorized[cat] = list(set(categorized[cat]))
        
        return categorized
    
    def get_random_items(self, category: str, count: int = 1) -> List[str]:
        """随机获取某类别的项目"""
        items = self.categorized_data.get(category, [])
        if not items:
            return [f"<{category}>"] * count
        return random.choices(items, k=min(count, len(items)))
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n数据提取统计:")
        print("=" * 50)
        for category, items in self.categorized_data.items():
            if items:
                print(f"  {category}: {len(items)} 项")
                print(f"    示例: {', '.join(items[:3])}")
        print("=" * 50)


# ==================== ScWeb 数据加载器 ====================
class ScWebDataLoader:
    """加载 ScWeb_Chinese_Translate 数据用于测试"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.data: List[Dict[str, str]] = []
        
        if config.scweb_mode != "off":
            self._load_data()
    
    def _load_data(self):
        """加载 ScWeb 数据"""
        scweb_folder = self.config.scweb_folder
        if not os.path.exists(scweb_folder):
            print(f"Warning: ScWeb folder not found: {scweb_folder}")
            return
        
        # 遍历所有 JSON 文件
        for filename in os.listdir(scweb_folder):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(scweb_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                for key, value in json_data.items():
                    # 过滤过长或过短的文本
                    if len(key) < 2 or len(value) < 2:
                        continue
                    if len(key) > 100 or len(value) > 100:
                        continue
                    
                    # 根据模式决定 source 和 target
                    if self.config.scweb_mode == "k2v":
                        # key->value: 英文->中文
                        self.data.append({
                            "source": key,
                            "target": value,
                            "file": filename
                        })
                    elif self.config.scweb_mode == "v2k":
                        # value->key: 中文->英文
                        self.data.append({
                            "source": value,
                            "target": key,
                            "file": filename
                        })
                        
            except Exception as e:
                print(f"Warning: Error loading ScWeb file {filepath}: {e}")
                continue
        
        print(f"\nScWeb 数据加载完成:")
        print(f"  模式: {self.config.scweb_mode}")
        print(f"  总条目: {len(self.data)}")
    
    def get_random_samples(self, count: int) -> List[Dict[str, str]]:
        """随机获取测试样本"""
        if not self.data:
            return []
        return random.sample(self.data, min(count, len(self.data)))
    
    def get_test_sentences(self, count: int) -> List[str]:
        """获取测试句子（仅返回 source）"""
        samples = self.get_random_samples(count)
        return [s["source"] for s in samples]
    
    def get_test_pairs(self, count: int) -> List[Tuple[str, str]]:
        """获取测试对（source, target）用于对比"""
        samples = self.get_random_samples(count)
        return [(s["source"], s["target"]) for s in samples]


# ==================== 测试句子生成器 ====================
class TestSentenceGenerator:
    """生成测试句子"""
    
    def __init__(self, extractor: GameDataExtractor, config: TestConfig):
        self.extractor = extractor
        self.config = config
    
    def generate_sentence(self, template: str = None) -> str:
        """生成一个测试句子"""
        if template is None:
            template = random.choice(self.config.templates)
        
        # 找出模板中需要的类别
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # 替换占位符
        sentence = template
        for placeholder in placeholders:
            replacement = self.extractor.get_random_items(placeholder, 1)[0]
            sentence = sentence.replace(f"{{{placeholder}}}", replacement, 1)
        
        return sentence
    
    def generate_batch(self, count: int) -> List[str]:
        """批量生成测试句子（包含固定句子）"""
        sentences = []
        
        # 首先添加所有固定句子
        sentences.extend(self.config.fixed_sentences)
        
        # 然后生成随机句子
        remaining = count - len(self.config.fixed_sentences)
        if remaining > 0:
            templates = self.config.templates.copy()
            random.shuffle(templates)
            
            for i in range(remaining):
                template = templates[i % len(templates)]
                sentence = self.generate_sentence(template)
                sentences.append(sentence)
        
        return sentences


# ==================== 翻译器（支持多模型对比）====================
class ModelWrapper:
    """模型包装器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def translate(self, text: str, max_length: int = 128) -> Tuple[str, float]:
        """翻译文本，返回 (translation, time_ms)"""
        raise NotImplementedError


class PyTorchModelWrapper(ModelWrapper):
    """PyTorch 模型包装器"""
    
    def __init__(self, name: str, model_path: str):
        super().__init__(name)
        print(f"\n加载 {name}: {model_path}")
        
        self.tokenizer = MarianTokenizer.from_pretrained(model_path)
        self.model = MarianMTModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 预热
        _ = self.translate("测试", 128)
        
        print(f"✅ {name} 加载成功")
    
    def translate(self, text: str, max_length: int = 128) -> Tuple[str, float]:
        """翻译文本"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            translated = self.model.generate(**inputs, max_length=max_length)
        time_ms = (time.time() - start_time) * 1000
        
        translation = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return translation, time_ms


class ONNXModelWrapper(ModelWrapper):
    """ONNX 量化模型包装器"""
    
    def __init__(self, name: str, model_path: str):
        super().__init__(name)
        
        if not ONNX_AVAILABLE:
            raise ImportError("需要安装 optimum: pip install optimum[onnxruntime]")
        
        print(f"\n加载 {name}: {model_path}")
        
        self.tokenizer = MarianTokenizer.from_pretrained(model_path)
        
        # 加载量化的 ONNX 模型（会自动查找 _q4f16.onnx 文件）
        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            model_path,
            provider="CPUExecutionProvider"
        )
        
        # 预热
        _ = self.translate("测试", 128)
        
        print(f"✅ {name} 加载成功")
    
    def translate(self, text: str, max_length: int = 128) -> Tuple[str, float]:
        """翻译文本"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        start_time = time.time()
        translated = self.model.generate(**inputs, max_length=max_length)
        time_ms = (time.time() - start_time) * 1000
        
        translation = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return translation, time_ms


class MultiModelTester:
    """多模型对比测试器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.models: List[ModelWrapper] = []
        self._load_models()
    
    def _load_models(self):
        """加载所有配置的模型"""
        print("\n" + "=" * 80)
        print("加载模型")
        print("=" * 80)
        
        # 加载原始基础模型
        if self.config.load_base_model:
            try:
                model = PyTorchModelWrapper("原始模型 (Base)", self.config.model_name)
                self.models.append(model)
            except Exception as e:
                print(f"⚠️  加载原始模型失败: {e}")
        
        # 加载微调模型
        if os.path.exists(self.config.finetuned_model_path):
            try:
                model = PyTorchModelWrapper("微调模型 (Fine-tuned)", self.config.finetuned_model_path)
                self.models.append(model)
            except Exception as e:
                print(f"⚠️  加载微调模型失败: {e}")
        else:
            print(f"⚠️  微调模型不存在: {self.config.finetuned_model_path}")
        
        # 加载量化模型
        if self.config.load_quantized and os.path.exists(self.config.quantized_model_path):
            try:
                model = ONNXModelWrapper("量化模型 (Quantized)", self.config.quantized_model_path)
                self.models.append(model)
            except Exception as e:
                print(f"⚠️  加载量化模型失败: {e}")
        
        if not self.models:
            raise ValueError("没有成功加载任何模型！")
        
        print(f"\n✅ 共加载 {len(self.models)} 个模型")
    
    def compare_translation(self, texts: List[str], max_length: int = 128):
        """对比翻译多个文本"""
        print("\n" + "=" * 80)
        print("模型对比测试")
        print("=" * 80)
        
        results = []
        
        for i, text in enumerate(texts, 1):
            print(f"\n{'=' * 80}")
            print(f"[{i}/{len(texts)}] 测试句子")
            print(f"{'=' * 80}")
            print(f"中文: {text}")
            print(f"{'-' * 80}")
            
            translations = {}
            times = {}
            
            for model in self.models:
                try:
                    translation, time_ms = model.translate(text, max_length)
                    translations[model.name] = translation
                    times[model.name] = time_ms
                    
                    print(f"\n{model.name}:")
                    print(f"  翻译: {translation}")
                    print(f"  耗时: {time_ms:.2f} ms")
                except Exception as e:
                    print(f"\n{model.name}: ❌ 翻译失败 - {e}")
            
            results.append({
                "source": text,
                "translations": translations,
                "times": times
            })
        
        return results
    
    def save_comparison_results(self, results: List[Dict], output_file: str = "comparison_results.txt"):
        """保存对比结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("模型对比测试结果\n")
            f.write("=" * 100 + "\n\n")
            
            for i, result in enumerate(results, 1):
                result_type = result.get('type', 'template')
                if result_type == 'scweb':
                    f.write(f"[{i}] ScWeb 测试句子\n")
                else:
                    f.write(f"[{i}] 测试句子\n")
                f.write(f"{'-' * 100}\n")
                f.write(f"源文本: {result['source']}\n")
                
                # 如果有参考翻译，显示它
                if 'reference' in result:
                    f.write(f"参考翻译: {result['reference']}\n")
                f.write("\n")
                
                for model_name, translation in result['translations'].items():
                    time_ms = result['times'].get(model_name, 0)
                    f.write(f"{model_name}:\n")
                    f.write(f"  翻译: {translation}\n")
                    f.write(f"  耗时: {time_ms:.2f} ms\n\n")
                
                f.write("=" * 100 + "\n\n")
            
            # 统计平均耗时
            f.write("\n平均推理时间:\n")
            f.write("-" * 100 + "\n")
            
            if results:
                for model in self.models:
                    avg_time = sum(r['times'].get(model.name, 0) for r in results) / len(results)
                    f.write(f"{model.name}: {avg_time:.2f} ms\n")
        
        print(f"\n对比结果已保存到: {output_file}")


# ==================== 单模型测试器（向后兼容）====================
class TranslationTester:
    """翻译测试器（单模型，向后兼容）"""
    
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n加载模型: {model_path}")
        print(f"设备: {self.device}")
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_path)
            self.model = MarianMTModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("模型加载成功！\n")
        except Exception as e:
            print(f"加载微调模型失败: {e}")
            print("尝试加载原始模型...\n")
            base_model = "Helsinki-NLP/opus-mt-zh-en"
            self.tokenizer = MarianTokenizer.from_pretrained(base_model)
            self.model = MarianMTModel.from_pretrained(base_model)
            self.model.to(self.device)
            self.model.eval()
    
    def translate(self, text: str, max_length: int = 128) -> str:
        """翻译单个句子"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            translated = self.model.generate(**inputs, max_length=max_length)
        
        translation = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return translation
    
    def translate_batch(self, texts: List[str], max_length: int = 128) -> List[Tuple[str, str]]:
        """批量翻译"""
        results = []
        print("\n" + "=" * 80)
        print("翻译测试结果")
        print("=" * 80)
        
        for i, text in enumerate(texts, 1):
            translation = self.translate(text, max_length)
            results.append((text, translation))
            
            print(f"\n[{i}/{len(texts)}]")
            print(f"中文: {text}")
            print(f"英文: {translation}")
        
        print("\n" + "=" * 80)
        return results
    
    def save_results(self, results: List[Tuple[str, str]], output_file: str = "test_results.txt"):
        """保存测试结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("翻译测试结果\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (source, target) in enumerate(results, 1):
                f.write(f"[{i}]\n")
                f.write(f"中文: {source}\n")
                f.write(f"英文: {target}\n\n")
        
        print(f"\n结果已保存到: {output_file}")


# ==================== CLI 交互模式 ====================
class InteractiveCLI:
    """交互式命令行翻译工具"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.tester = None
    
    def start(self):
        """启动交互式 CLI"""
        print("\n" + "=" * 80)
        print("交互式翻译对比工具")
        print("=" * 80)
        print("\n加载模型，请稍候...")
        
        # 加载模型
        self.tester = MultiModelTester(self.config)
        
        if not self.tester.models:
            print("\n❌ 没有可用的模型，退出")
            return
        
        print("\n" + "=" * 80)
        print("已加载模型:")
        for i, model in enumerate(self.tester.models, 1):
            print(f"  [{i}] {model.name}")
        print("=" * 80)
        
        print("\n使用说明:")
        print("  - 输入中文句子进行翻译")
        print("  - 输入 'exit' 或 'quit' 退出")
        print("  - 输入 'clear' 清屏")
        print("=" * 80)
        
        # 主循环
        while True:
            try:
                print("\n")
                user_input = input("请输入中文 > ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n再见！")
                    break
                
                if user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                # 翻译
                self._translate_and_compare(user_input)
                
            except KeyboardInterrupt:
                print("\n\n收到中断信号，退出...")
                break
            except EOFError:
                print("\n\n收到 EOF，退出...")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
    
    def _translate_and_compare(self, text: str):
        """翻译并对比"""
        print("\n" + "-" * 80)
        print(f"源文本: {text}")
        print("-" * 80)
        
        for model in self.tester.models:
            try:
                translation, time_ms = model.translate(text, self.config.max_length)
                print(f"\n[{model.name}]")
                print(f"  翻译: {translation}")
                print(f"  耗时: {time_ms:.2f} ms")
            except Exception as e:
                print(f"\n[{model.name}]")
                print(f"  ❌ 翻译失败: {e}")
        
        print("-" * 80)


# ==================== 主函数 ====================
# ==================== 主函数 ====================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Opus-MT Test Script")
    parser.add_argument("--config", type=str, default="config/zh-en.json", help="Path to configuration file")
    parser.add_argument("--cli", action="store_true", help="Enable interactive CLI mode")
    args = parser.parse_args()
    
    # 尝试加载配置
    if os.path.exists(args.config):
        config = TestConfig.from_json(args.config)
        # 如果是 zh-en.json，可能没有 default values for new fields if they were omitted? 
        # But dataclass has defaults.
    else:
        print(f"Config file not found: {args.config}, using defaults.")
        config = TestConfig()
        # 仅在 config 路径包含 json 时保存默认
        if args.config.endswith(".json"):
             try:
                os.makedirs(os.path.dirname(args.config), exist_ok=True)
                with open(args.config, 'w', encoding='utf-8') as f:
                    # 使用 exclude? No need
                    json.dump(asdict(config), f, indent=2, ensure_ascii=False)
                print(f"Created default config at {args.config}")
             except Exception as e:
                print(f"Warning: Failed to save default config: {e}")
    
    if args.cli:
        # CLI 交互模式
        cli = InteractiveCLI(config)
        cli.start()
        return
    
    # 批量测试模式
    # 加载数据
    print("\n加载游戏数据...")
    ini_path = os.path.join(config.dataset_folder, config.source_file)
    extractor = GameDataExtractor(ini_path)
    extractor.print_statistics()
    
    # 生成测试句子
    print(f"\n生成 {config.num_test_sentences} 个测试句子...")
    generator = TestSentenceGenerator(extractor, config)
    test_sentences = generator.generate_batch(config.num_test_sentences)
    
    # 加载 ScWeb 数据（如果启用）
    scweb_loader = ScWebDataLoader(config)
    scweb_pairs = []
    if config.scweb_mode != "off" and scweb_loader.data:
        scweb_pairs = scweb_loader.get_test_pairs(config.scweb_test_count)
        print(f"\n从 ScWeb 数据中抽取了 {len(scweb_pairs)} 个测试对")
    
    if not test_sentences and not scweb_pairs:
        print("\n⚠️ No test sentences configured or generated. Skipping test.")
        return

    # 根据配置选择测试模式
    if config.compare_models:
        # 多模型对比模式
        print("\n" + "=" * 80)
        print("多模型对比模式")
        print("=" * 80)
        
        tester = MultiModelTester(config)
        
        # 测试普通句子
        results = []
        if test_sentences:
            print("\n" + "-" * 80)
            print("模板生成句子测试")
            print("-" * 80)
            results = tester.compare_translation(test_sentences, config.max_length)
        
        # 测试 ScWeb 数据（带参考翻译对比）
        if scweb_pairs:
            print("\n" + "=" * 80)
            print("ScWeb 数据对比测试（含参考翻译）")
            print("=" * 80)
            
            for i, (source, reference) in enumerate(scweb_pairs, 1):
                print(f"\n{'=' * 80}")
                print(f"[ScWeb {i}/{len(scweb_pairs)}] 测试句子")
                print(f"{'=' * 80}")
                print(f"源文本: {source}")
                print(f"参考翻译: {reference}")
                print(f"{'-' * 80}")
                
                translations = {}
                times = {}
                
                for model in tester.models:
                    try:
                        translation, time_ms = model.translate(source, config.max_length)
                        translations[model.name] = translation
                        times[model.name] = time_ms
                        
                        print(f"\n{model.name}:")
                        print(f"  翻译: {translation}")
                        print(f"  耗时: {time_ms:.2f} ms")
                    except Exception as e:
                        print(f"\n{model.name}: ❌ 翻译失败 - {e}")
                
                results.append({
                    "source": source,
                    "reference": reference,
                    "translations": translations,
                    "times": times,
                    "type": "scweb"
                })
        
        tester.save_comparison_results(results)
    else:
        # 单模型测试模式（向后兼容）
        print("\n" + "=" * 80)
        print("单模型测试模式")
        print("=" * 80)
        
        tester = TranslationTester(config.finetuned_model_path)
        
        all_sentences = test_sentences.copy()
        if scweb_pairs:
            all_sentences.extend([s for s, _ in scweb_pairs])
        
        results = tester.translate_batch(all_sentences, config.max_length)
        tester.save_results(results)
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()

