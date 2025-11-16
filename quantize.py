"""
模型量化脚本 - 将微调后的模型量化为 ONNX 格式
使用 optimum 进行 INT4 量化，减小模型体积并提升推理速度
"""

import os
import shutil
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

# ==================== 配置 ====================
class QuantizationConfig:
    """量化配置"""
    model_path = "./results/final_model"  # 微调后的模型路径
    output_path = "./results/final_model/onnx"  # ONNX 输出路径（原始和量化模型都在此目录）


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
        
        # 量化配置 (INT4 权重 + FP16 激活)
        print("\n配置量化参数:")
        print("  权重精度: INT4")
        print("  激活精度: FP16")
        print(f"  输出目录: {self.output_path}")
        
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        
        # 量化 encoder
        print(f"\n量化 Encoder...")
        try:
            quantizer = ORTQuantizer.from_pretrained(self.output_path, file_name="encoder_model.onnx")
            quantizer.quantize(
                save_dir=self.output_path,
                quantization_config=qconfig,
                file_suffix="q4f16"
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
                    file_suffix="q4f16"
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
                    file_suffix="q4f16"
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
        # 查找并重命名 encoder
        encoder_files = list(self.output_path.glob("encoder_model*_q4f16.onnx"))
        if encoder_files:
            target_name = self.output_path / "encoder_model_q4f16.onnx"
            if encoder_files[0] != target_name:
                encoder_files[0].rename(target_name)
                print(f"  ✓ Encoder: {target_name.name}")
        
        # 查找并重命名 decoder
        decoder_files = list(self.output_path.glob("decoder*_q4f16.onnx"))
        if decoder_files:
            target_name = self.output_path / "decoder_model_q4f16.onnx"
            if decoder_files[0] != target_name:
                decoder_files[0].rename(target_name)
                print(f"  ✓ Decoder: {target_name.name}")
    
    def _show_size_comparison(self):
        """显示量化前后的大小对比"""
        print("\n模型大小对比:")
        print("-" * 80)
        
        # 原始 ONNX 模型（不含量化后缀）
        original_size = 0
        original_files = [f for f in self.output_path.glob("*.onnx") if "_q4f16" not in f.name]
        for onnx_file in original_files:
            size = onnx_file.stat().st_size
            original_size += size
            print(f"  原始: {onnx_file.name:40} {size / 1024 / 1024:8.2f} MB")
        
        print("-" * 80)
        
        # 量化后模型
        quantized_size = 0
        quantized_files = list(self.output_path.glob("*_q4f16.onnx"))
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
        
        # 检查模型是否存在
        if not self.model_path.exists():
            print(f"\n❌ 错误: 找不到模型路径 {self.model_path}")
            print("请先运行 train.py 训练模型")
            return
        
        try:
            # 步骤1: 导出 ONNX
            self.export_to_onnx()
            
            # 步骤2: 量化
            self.quantize_model()
            
            print("\n" + "=" * 80)
            print("✅ 量化流程全部完成！")
            print("=" * 80)
            print("\n使用方法:")
            print(f"  原始模型 (PyTorch): {self.model_path}")
            print(f"  ONNX 模型目录: {self.output_path}")
            print(f"    - 原始 ONNX: encoder_model.onnx, decoder_*.onnx")
            print(f"    - 量化 ONNX: encoder_model_q4f16.onnx, decoder_model_q4f16.onnx")
            print("\n运行 test.py 进行对比测试")
            
        except Exception as e:
            print(f"\n❌ 量化过程出错: {e}")
            import traceback
            traceback.print_exc()


# ==================== 主函数 ====================
def main():
    """主函数"""
    config = QuantizationConfig()
    quantizer = ModelQuantizer(config)
    quantizer.run()


if __name__ == "__main__":
    main()
