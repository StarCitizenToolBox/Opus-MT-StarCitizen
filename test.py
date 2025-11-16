"""
测试翻译模型 - 使用真实游戏数据生成测试句子
从 INI 文件中提取地点、载具、物品等，生成常见的游戏场景对话
支持对比原始模型、微调模型和量化模型
支持交互式 CLI 模式
"""

import os
import re
import random
import time
import sys
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
class TestConfig:
    """测试配置"""
    base_model_name = "Helsinki-NLP/opus-mt-zh-en"  # 原始基础模型
    finetuned_model_path = "./results/final_model"  # 微调后的模型路径
    quantized_model_path = "./results/final_model/onnx"  # 量化模型路径（与原始 ONNX 在同一目录）
    
    dataset_folder = "dataset"
    source_file = "chinese_(simplified).ini"
    
    # 生成配置
    num_test_sentences = 20  # 生成测试句子数量
    max_length = 128
    
    # 对比配置
    compare_models = True  # 是否进行模型对比
    load_base_model = True  # 是否加载原始模型
    load_quantized = True  # 是否加载量化模型


# ==================== 数据提取器 ====================
class GameDataExtractor:
    """从 INI 文件提取游戏数据"""
    
    # 分类正则表达式
    CATEGORIES = {
        "location": [
            r"^(Stanton|Crusader|ArcCorp|Hurston|microTech|Orison|Area18|Lorville|NewBabbage)(?!.*_Desc).*",
            r"^(Port_Olisar|GrimHex|Everus|Baijini|Seraphim).*",
            r".*(?:Station|Port|Outpost|Settlement)(?!.*_Desc).*",
        ],
        "vehicle": [
            r"^vehicle_Name.*",
        ],
        "item": [
            r"^item_Name.*",
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


# ==================== 测试句子生成器 ====================
class TestSentenceGenerator:
    """生成测试句子"""

     # 固定测试语句（真实玩家对话）
    FIXED_SENTENCES = [
        "老登 什么时候让我上你的英仙座上当炮手",
        "之前看别人说货运电梯要实装，以后不能自动装卸货了，那几千scu的货怎么装？手动？",
        "而且我把维生排气都关掉了，门也都打开了，这火也不会灭",
        "机库就一个红灯了，有无来开机库的，凑一套卡搞点组件",
    ]
    
    # 句子模板（游戏常见对话场景）
    TEMPLATES = [
        # 地点相关
        "有人去{location}打赏金吗？",
        "我在{location}，有人组队吗？",
        "{location}现在安全吗？",
        "从{location}到{location}需要多久？",
        "我要去{location}做任务",
        "{location}有好的装备商店吗？",
        
        # 载具相关
        "我开着{vehicle}",
        "{vehicle}这艘船怎么样？",
        "想买一艘{vehicle}",
        "{vehicle}能装多少货？",
        "有人会开{vehicle}吗？",
        "{vehicle}适合新手吗？",
        "我的{vehicle}在{location}",
        
        # 组合场景
        "有人去{location}打赏金吗？我开着{vehicle}",
        "开{vehicle}去{location}做任务",
        "在{location}买了{vehicle}",
        "{vehicle}停在{location}了",
        "我的{vehicle}被困在{location}了",
        
        # 物品相关
        "哪里能买到{item}？",
        "{item}多少钱？",
        "我需要{item}",
        "{location}有卖{item}吗？",
        
        # 战斗场景
        "在{location}被袭击了",
        "开着{vehicle}去{location}战斗",
        "{location}有敌人",
        
        # 贸易场景
        "在{location}卖货",
        "用{vehicle}运货去{location}",
        "{location}的价格怎么样？",
        
        # 求助场景
        "我在{location}迷路了",
        "有人在{location}附近吗？",
        "{vehicle}坏了，在{location}",
        "需要在{location}帮忙",
        
        # 闲聊场景
        "{location}真漂亮",
        "{vehicle}飞起来真爽",
        "第一次来{location}",
        "刚买的{vehicle}",
    ]
    
    def __init__(self, extractor: GameDataExtractor):
        self.extractor = extractor
    
    def generate_sentence(self, template: str = None) -> str:
        """生成一个测试句子"""
        if template is None:
            template = random.choice(self.TEMPLATES)
        
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
        sentences.extend(self.FIXED_SENTENCES)
        
        # 然后生成随机句子
        remaining = count - len(self.FIXED_SENTENCES)
        if remaining > 0:
            templates = self.TEMPLATES.copy()
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
                model = PyTorchModelWrapper("原始模型 (Base)", self.config.base_model_name)
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
                f.write(f"[{i}] 测试句子\n")
                f.write(f"{'-' * 100}\n")
                f.write(f"中文: {result['source']}\n\n")
                
                for model_name, translation in result['translations'].items():
                    time_ms = result['times'].get(model_name, 0)
                    f.write(f"{model_name}:\n")
                    f.write(f"  翻译: {translation}\n")
                    f.write(f"  耗时: {time_ms:.2f} ms\n\n")
                
                f.write("=" * 100 + "\n\n")
            
            # 统计平均耗时
            f.write("\n平均推理时间:\n")
            f.write("-" * 100 + "\n")
            
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
    # 检查命令行参数
    cli_mode = '-cli' in sys.argv or '--cli' in sys.argv
    
    config = TestConfig()
    
    if cli_mode:
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
    generator = TestSentenceGenerator(extractor)
    test_sentences = generator.generate_batch(config.num_test_sentences)
    
    # 根据配置选择测试模式
    if config.compare_models:
        # 多模型对比模式
        print("\n" + "=" * 80)
        print("多模型对比模式")
        print("=" * 80)
        
        tester = MultiModelTester(config)
        results = tester.compare_translation(test_sentences, config.max_length)
        tester.save_comparison_results(results)
    else:
        # 单模型测试模式（向后兼容）
        print("\n" + "=" * 80)
        print("单模型测试模式")
        print("=" * 80)
        
        tester = TranslationTester(config.finetuned_model_path)
        results = tester.translate_batch(test_sentences, config.max_length)
        tester.save_results(results)
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
