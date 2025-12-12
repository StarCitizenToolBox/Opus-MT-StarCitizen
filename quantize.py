"""
模型量化脚本 - 将微调后的模型量化为 ONNX 格式
使用 optimum 进行 INT4 量化，减小模型体积并提升推理速度
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer


def load_config(config_path: str) -> dict:
    """从 JSON 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='模型量化工具 - 将微调后的模型量化为 ONNX 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件
  python quantize.py --config config/zh-en.json
  
  # 指定模型路径（覆盖配置文件）
  python quantize.py --config config/zh-en.json --model-path ./my_model
  
  # 不使用配置文件，直接指定路径
  python quantize.py --model-path ./results/final_model --output-path ./results/final_model/onnx
  
  # 使用不同的量化配置
  python quantize.py --config config/en-zh.json --quantization-type arm64
  
  # 跳过某个步骤
  python quantize.py --config config/zh-en.json --skip-export
  python quantize.py --config config/zh-en.json --skip-quantize
        """
    )
    
    # 配置文件
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径 (JSON 格式，例如 config/zh-en.json)'
    )
    
    # 模型路径
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=None,
        help='微调后的模型路径 (覆盖配置文件中的 finetuned_model_path)'
    )
    
    # 输出路径
    parser.add_argument(
        '--output-path', '-o',
        type=str,
        default=None,
        help='ONNX 模型输出路径 (覆盖配置文件中的 quantized_model_path)'
    )
    
    # 量化类型
    parser.add_argument(
        '--quantization-type', '-q',
        type=str,
        choices=['avx512_vnni', 'avx2', 'arm64'],
        default='avx512_vnni',
        help='量化配置类型 (默认: avx512_vnni)'
    )
    
    # 是否使用 per_channel 量化
    parser.add_argument(
        '--per-channel',
        action='store_true',
        default=True,
        help='使用 per-channel 量化 (默认: True)'
    )
    
    parser.add_argument(
        '--no-per-channel',
        action='store_false',
        dest='per_channel',
        help='不使用 per-channel 量化'
    )
    
    # 量化文件后缀
    parser.add_argument(
        '--file-suffix',
        type=str,
        default='q4f16',
        help='量化后文件的后缀 (默认: q4f16)'
    )
    
    # 跳过步骤
    parser.add_argument(
        '--skip-export',
        action='store_true',
        default=False,
        help='跳过 ONNX 导出步骤 (仅量化已存在的 ONNX 模型)'
    )
    
    parser.add_argument(
        '--skip-quantize',
        action='store_true',
        default=False,
        help='跳过量化步骤 (仅导出 ONNX 模型)'
    )
    
    return parser.parse_args()


# ==================== 配置 ====================
class QuantizationConfig:
    """量化配置"""
    
    def __init__(
        self,
        model_path: str = "./results/final_model",
        output_path: str = "./results/final_model/onnx",
        quantization_type: str = "avx512_vnni",
        per_channel: bool = True,
        file_suffix: str = "q4f16",
        skip_export: bool = False,
        skip_quantize: bool = False
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.quantization_type = quantization_type
        self.per_channel = per_channel
        self.file_suffix = file_suffix
        self.skip_export = skip_export
        self.skip_quantize = skip_quantize
    
    @classmethod
    def from_args(cls, args) -> 'QuantizationConfig':
        """从命令行参数创建配置"""
        # 如果指定了配置文件，先加载配置
        config_data = {}
        if args.config:
            config_data = load_config(args.config)
        
        # 确定模型路径（命令行参数优先级最高）
        model_path = args.model_path or config_data.get('finetuned_model_path', './results/final_model')
        output_path = args.output_path or config_data.get('quantized_model_path', './results/final_model/onnx')
        
        return cls(
            model_path=model_path,
            output_path=output_path,
            quantization_type=args.quantization_type,
            per_channel=args.per_channel,
            file_suffix=args.file_suffix,
            skip_export=args.skip_export,
            skip_quantize=args.skip_quantize
        )


# ==================== 量化器 ====================
class ModelQuantizer:
    """模型量化器"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.model_path = Path(config.model_path)
        self.output_path = Path(config.output_path)
    
    def export_to_onnx(self):
        """步骤1: 导出模型到 ONNX 格式"""
        print("\n" + "=" * 80)
        print("步骤 1: 导出模型到 ONNX 格式")
        print("=" * 80)
        
        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        print(f"\n加载模型: {self.model_path}")
        
        try:
            # 使用 optimum 导出 ONNX
            model = ORTModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                export=True,
                provider="CPUExecutionProvider"
            )
            
            # 保存 ONNX 模型
            print(f"保存 ONNX 模型到: {self.output_path}")
            model.save_pretrained(self.output_path)
            
            # 同时保存 tokenizer
            tokenizer = MarianTokenizer.from_pretrained(self.model_path)
            tokenizer.save_pretrained(self.output_path)
            
            print("✅ ONNX 导出成功！")
            
            # 显示生成的文件
            onnx_files = list(self.output_path.glob("*.onnx"))
            print(f"\n生成的 ONNX 文件:")
            for f in onnx_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name}: {size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ ONNX 导出失败: {e}")
            raise
    
    def quantize_model(self):
        """步骤2: 量化 ONNX 模型 (INT4 + FP16)"""
        print("\n" + "=" * 80)
        print("步骤 2: 量化 ONNX 模型 (INT4 权重 + FP16 激活)")
        print("=" * 80)
        
        # 查找 encoder 和 decoder 模型
        encoder_path = self.output_path / "encoder_model.onnx"
        decoder_path = self.output_path / "decoder_model.onnx"
        decoder_with_past_path = self.output_path / "decoder_with_past_model.onnx"
        
        if not encoder_path.exists():
            print(f"❌ 找不到 encoder 模型: {encoder_path}")
            return
        
        if not decoder_path.exists() and not decoder_with_past_path.exists():
            print(f"❌ 找不到 decoder 模型")
            return
        
        # 根据配置选择量化配置
        quantization_type = self.config.quantization_type
        per_channel = self.config.per_channel
        file_suffix = self.config.file_suffix
        
        print("\n配置量化参数:")
        print("  权重精度: INT4")
        print("  激活精度: FP16")
        print(f"  量化类型: {quantization_type}")
        print(f"  Per-channel: {per_channel}")
        print(f"  文件后缀: {file_suffix}")
        print(f"  输出目录: {self.output_path}")
        
        # 创建量化配置
        if quantization_type == 'avx512_vnni':
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=per_channel)
        elif quantization_type == 'avx2':
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=per_channel)
        elif quantization_type == 'arm64':
            qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=per_channel)
        else:
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=per_channel)
        
        # 量化 encoder
        print(f"\n量化 Encoder...")
        try:
            quantizer = ORTQuantizer.from_pretrained(self.output_path, file_name="encoder_model.onnx")
            quantizer.quantize(
                save_dir=self.output_path,
                quantization_config=qconfig,
                file_suffix=file_suffix
            )
            print("✅ Encoder 量化成功")
        except Exception as e:
            print(f"⚠️  Encoder 量化失败: {e}")
        
        # 量化 decoder (优先使用 with_past 版本)
        if decoder_with_past_path.exists():
            print(f"\n量化 Decoder (decoder_with_past_model.onnx)...")
            try:
                quantizer = ORTQuantizer.from_pretrained(self.output_path, file_name="decoder_with_past_model.onnx")
                quantizer.quantize(
                    save_dir=self.output_path,
                    quantization_config=qconfig,
                    file_suffix=file_suffix
                )
                print("✅ Decoder (with_past) 量化成功")
            except Exception as e:
                print(f"⚠️  Decoder (with_past) 量化失败: {e}")
        
        # 同时量化普通 decoder（作为备用）
        if decoder_path.exists():
            print(f"\n量化 Decoder (decoder_model.onnx)...")
            try:
                quantizer = ORTQuantizer.from_pretrained(self.output_path, file_name="decoder_model.onnx")
                quantizer.quantize(
                    save_dir=self.output_path,
                    quantization_config=qconfig,
                    file_suffix=file_suffix
                )
                print("✅ Decoder 量化成功")
            except Exception as e:
                print(f"⚠️  Decoder 量化失败: {e}")
        
        # 重命名量化模型文件以匹配标准格式
        print("\n重命名量化模型文件...")
        self._rename_quantized_files()
        
        # 显示结果
        print("\n" + "=" * 80)
        print("量化完成！")
        print("=" * 80)
        
        self._show_size_comparison()
    
    def _rename_quantized_files(self):
        """重命名量化文件为标准格式"""
        file_suffix = self.config.file_suffix
        
        # 查找并重命名 encoder
        encoder_files = list(self.output_path.glob(f"encoder_model*_{file_suffix}.onnx"))
        if encoder_files:
            target_name = self.output_path / f"encoder_model_{file_suffix}.onnx"
            if encoder_files[0] != target_name:
                encoder_files[0].rename(target_name)
                print(f"  ✓ Encoder: {target_name.name}")
        
        # 查找并重命名 decoder
        decoder_files = list(self.output_path.glob(f"decoder*_{file_suffix}.onnx"))
        if decoder_files:
            target_name = self.output_path / f"decoder_model_{file_suffix}.onnx"
            if decoder_files[0] != target_name:
                decoder_files[0].rename(target_name)
                print(f"  ✓ Decoder: {target_name.name}")
    
    def _show_size_comparison(self):
        """显示量化前后的大小对比"""
        file_suffix = self.config.file_suffix
        
        print("\n模型大小对比:")
        print("-" * 80)
        
        # 原始 ONNX 模型（不含量化后缀）
        original_size = 0
        original_files = [f for f in self.output_path.glob("*.onnx") if f"_{file_suffix}" not in f.name]
        for onnx_file in original_files:
            size = onnx_file.stat().st_size
            original_size += size
            print(f"  原始: {onnx_file.name:40} {size / 1024 / 1024:8.2f} MB")
        
        print("-" * 80)
        
        # 量化后模型
        quantized_size = 0
        quantized_files = list(self.output_path.glob(f"*_{file_suffix}.onnx"))
        for onnx_file in quantized_files:
            size = onnx_file.stat().st_size
            quantized_size += size
            print(f"  量化: {onnx_file.name:40} {size / 1024 / 1024:8.2f} MB")
        
        print("-" * 80)
        
        if original_size > 0 and quantized_size > 0:
            reduction = (1 - quantized_size / original_size) * 100
            print(f"  总大小: {original_size / 1024 / 1024:.2f} MB → {quantized_size / 1024 / 1024:.2f} MB")
            print(f"  压缩率: {reduction:.1f}%")
        
        print("-" * 80)
        print(f"\n所有模型保存位置: {self.output_path}")
    
    def run(self):
        """运行完整的量化流程"""
        print("\n" + "=" * 80)
        print("模型量化工具")
        print("=" * 80)
        print(f"\n输入模型: {self.model_path}")
        print(f"输出路径: {self.output_path}")
        print(f"量化类型: {self.config.quantization_type}")
        print(f"Per-channel: {self.config.per_channel}")
        print(f"文件后缀: {self.config.file_suffix}")
        
        # 检查模型是否存在
        if not self.config.skip_export and not self.model_path.exists():
            print(f"\n❌ 错误: 找不到模型路径 {self.model_path}")
            print("请先运行 train.py 训练模型")
            return
        
        # 如果跳过导出但需要量化，检查 ONNX 目录是否存在
        if self.config.skip_export and not self.config.skip_quantize:
            if not self.output_path.exists():
                print(f"\n❌ 错误: 找不到 ONNX 模型目录 {self.output_path}")
                print("请先导出 ONNX 模型或移除 --skip-export 参数")
                return
        
        try:
            # 步骤1: 导出 ONNX
            if not self.config.skip_export:
                self.export_to_onnx()
            else:
                print("\n跳过 ONNX 导出步骤")
            
            # 步骤2: 量化
            if not self.config.skip_quantize:
                self.quantize_model()
            else:
                print("\n跳过量化步骤")
            
            print("\n" + "=" * 80)
            print("✅ 量化流程全部完成！")
            print("=" * 80)
            
            file_suffix = self.config.file_suffix
            print("\n使用方法:")
            print(f"  原始模型 (PyTorch): {self.model_path}")
            print(f"  ONNX 模型目录: {self.output_path}")
            print(f"    - 原始 ONNX: encoder_model.onnx, decoder_*.onnx")
            print(f"    - 量化 ONNX: encoder_model_{file_suffix}.onnx, decoder_model_{file_suffix}.onnx")
            print("\n运行 test.py 进行对比测试")
            
        except Exception as e:
            print(f"\n❌ 量化过程出错: {e}")
            import traceback
            traceback.print_exc()


# ==================== 主函数 ====================
def main():
    """主函数"""
    args = parse_args()
    config = QuantizationConfig.from_args(args)
    quantizer = ModelQuantizer(config)
    quantizer.run()


if __name__ == "__main__":
    main()
