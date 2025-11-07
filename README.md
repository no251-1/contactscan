# PVC Contact差异识别系统

基于Passive Voltage Contrast (PVC)技术的Contact自动检测与差异分析应用。

## 功能特点

- 🔍 **自动Contact检测**：基于图像处理技术自动识别SEM图像中的Contact区域
- 📊 **亮度分析**：根据PVC原理分析每个Contact的电气状态（VDD/GND/浮空）
- ⚠️ **异常识别**：自动识别异常Contact（开路、短路、形状不规则等）
- 🔬 **双图对比**：对比两张PVC图像，找出差异Contact
- 📈 **可视化展示**：直观展示检测结果和统计信息

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行

启动Streamlit应用：
```bash
streamlit run app.py
```

应用将在浏览器中自动打开（默认地址：http://localhost:8501）

## 使用说明

### 单图分析
1. 选择"单图分析"标签
2. 上传PVC图像或使用Demo图片
3. 查看自动检测结果：
   - Contact总数统计
   - 异常Contact标记
   - 详细Contact信息表

### 双图对比
1. 选择"双图对比"标签
2. 上传参考图像和对比图像
3. 点击"开始对比分析"
4. 查看差异Contact列表和详细对比信息

### 参数调整
在侧边栏可以调整以下参数：
- 最小/最大Contact尺寸
- 高/低亮度阈值

## PVC原理

根据PVC技术原理：
- **亮区域**：高电位（VDD），二次电子易逸出
- **暗区域**：低电位（GND），二次电子难逸出
- **异常区域**：浮空节点，亮度异常，可能表示开路或短路

## 技术栈

- **Streamlit**：Web应用框架
- **OpenCV**：图像处理
- **NumPy**：数值计算
- **Matplotlib**：可视化
- **Pandas**：数据处理

## 注意事项

1. 图像质量影响检测效果，建议使用清晰的SEM图像
2. 可根据实际图像调整参数以获得最佳效果
3. 异常检测结果仅供参考，需要结合实际电路分析

