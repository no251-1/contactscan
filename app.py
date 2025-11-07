"""
PVC图片自动对比识别差异Contact应用
基于Passive Voltage Contrast技术识别集成电路中的Contact异常
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import io
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


# 页面配置
st.set_page_config(
    page_title="PVC Contact差异识别系统",
    page_icon="🔬",
    layout="wide"
)

class PVCContactAnalyzer:
    """PVC Contact分析器"""
    
    def __init__(self):
        self.min_contact_size = 5  # 最小Contact尺寸（像素）
        self.max_contact_size = 100  # 最大Contact尺寸（像素）
        self.brightness_threshold_high = 180  # 高亮度阈值（VDD区域）
        self.brightness_threshold_low = 80   # 低亮度阈值（GND区域）
        
    def preprocess_image(self, image):
        """图像预处理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 去噪
        denoised = cv2.medianBlur(gray, 5)
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced, denoised
    
    def is_circular(self, contour, min_circularity=0.7):
        """判断轮廓是否为圆形"""
        area = cv2.contourArea(contour)
        if area == 0:
            return False, 0
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False, 0
        
        # 圆形度计算
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 检查是否接近圆形
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius
        if circle_area > 0:
            extent = area / circle_area  # 轮廓面积与最小外接圆面积的比值
            # 圆形应该：圆形度高 且 填充度高
            is_circle = circularity > min_circularity and extent > 0.7
            return is_circle, circularity
        return False, circularity
    
    def is_square(self, contour, min_rectangularity=0.85):
        """判断轮廓是否为方形"""
        area = cv2.contourArea(contour)
        if area == 0:
            return False, 0
        
        # 使用轮廓近似
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 方形应该有4个顶点
        if len(approx) == 4:
            # 计算矩形度（轮廓面积与边界框面积的比值）
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            if rect_area > 0:
                rectangularity = area / rect_area
                # 检查宽高比（方形应该接近1:1）
                aspect_ratio = float(w) / h if h > 0 else 0
                is_square_shape = (rectangularity > min_rectangularity and 
                                  0.7 < aspect_ratio < 1.3)
                return is_square_shape, rectangularity
        return False, 0
    
    def detect_contacts(self, image, method='combined', min_circularity=0.65, min_rectangularity=0.80):
        """检测Contact区域 - 优化版，只识别圆形和方形"""
        contacts = []
        
        # 预处理：使用多种方法结合
        # 方法1: 自适应阈值
        if method in ['adaptive', 'combined']:
            binary1 = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, binary1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 方法2: Otsu阈值（作为补充）
        if method == 'combined':
            _, binary2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # 合并两种二值化结果
            binary = cv2.bitwise_or(binary1, binary2)
        else:
            binary = binary1
        
        # 形态学操作 - 使用更小的核，避免过度连接
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 先闭运算填充小孔
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        # 开运算去除小噪声
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_medium)
        
        # 方法1: 使用Hough圆检测识别圆形Contact
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(self.min_contact_size, 10),  # 最小圆心距离
            param1=50,  # 上阈值
            param2=30,  # 累加器阈值
            minRadius=self.min_contact_size // 2,
            maxRadius=self.max_contact_size // 2
        )
        
        detected_circles = set()
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                detected_circles.add((cx, cy, r))
        
        # 方法2: 轮廓检测识别圆形和方形
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # 过滤太小或太大的区域
            if area < self.min_contact_size**2 or area > self.max_contact_size**2:
                continue
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算中心点
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 判断是否为圆形
            is_circle, circularity = self.is_circular(contour, min_circularity=min_circularity)
            
            # 判断是否为方形
            is_square_shape, rectangularity = self.is_square(contour, min_rectangularity=min_rectangularity)
            
            # 只保留圆形或方形的Contact
            if is_circle or is_square_shape:
                shape_type = "圆形" if is_circle else "方形"
                
                # 检查是否与Hough检测的圆重复（圆形Contact优先使用Hough结果）
                is_duplicate = False
                if is_circle:
                    for (hx, hy, hr) in detected_circles:
                        dist = np.sqrt((cx - hx)**2 + (cy - hy)**2)
                        if dist < max(hr, 10):  # 距离阈值
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    contacts.append({
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'contour': contour,
                        'circularity': circularity,
                        'rectangularity': rectangularity,
                        'shape_type': shape_type
                    })
        
        # 添加Hough检测到的圆形（如果轮廓检测遗漏了）
        for (cx, cy, r) in detected_circles:
            # 检查是否已经在contacts中
            already_found = False
            for contact in contacts:
                dist = np.sqrt((cx - contact['center'][0])**2 + (cy - contact['center'][1])**2)
                if dist < r:
                    already_found = True
                    break
            
            if not already_found:
                area = np.pi * r * r
                if self.min_contact_size**2 < area < self.max_contact_size**2:
                    x, y = cx - r, cy - r
                    w, h = 2 * r, 2 * r
                    
                    # 创建轮廓（近似圆形）
                    contour = np.array([[cx + r * np.cos(angle), cy + r * np.sin(angle)]
                                       for angle in np.linspace(0, 2*np.pi, 32)], dtype=np.int32)
                    contour = contour.reshape(-1, 1, 2)
                    
                    contacts.append({
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'contour': contour,
                        'circularity': 1.0,  # Hough检测的圆，圆形度设为1.0
                        'rectangularity': 0,
                        'shape_type': "圆形"
                    })
        
        return contacts, binary
    
    def analyze_contact_brightness(self, image, contacts):
        """分析每个Contact的亮度特征"""
        analyzed_contacts = []
        
        for contact in contacts:
            cx, cy = contact['center']
            x, y, w, h = contact['bbox']
            
            # 提取Contact区域
            roi = image[max(0, y-2):min(image.shape[0], y+h+2), 
                       max(0, x-2):min(image.shape[1], x+w+2)]
            
            if roi.size > 0:
                # 计算平均亮度
                mean_brightness = np.mean(roi)
                std_brightness = np.std(roi)
                
                # 判断Contact类型
                if mean_brightness > self.brightness_threshold_high:
                    contact_type = "VDD (高电位)"
                    status = "正常-高电位"
                elif mean_brightness < self.brightness_threshold_low:
                    contact_type = "GND (接地)"
                    status = "正常-接地"
                else:
                    contact_type = "浮空/异常"
                    status = "异常"
                
                # 计算与周围区域的对比度
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contact['contour']], -1, 255, -1)
                
                # 获取周围区域
                kernel = np.ones((20, 20), np.uint8)
                dilated = cv2.dilate(mask, kernel, iterations=1)
                surrounding = cv2.bitwise_and(image, cv2.bitwise_not(mask))
                surrounding = cv2.bitwise_and(surrounding, dilated)
                
                surrounding_mean = np.mean(surrounding[surrounding > 0]) if np.any(surrounding > 0) else mean_brightness
                contrast = abs(mean_brightness - surrounding_mean)
                
                # 获取形状类型（如果存在）
                shape_type = contact.get('shape_type', '未知')
                
                contact.update({
                    'mean_brightness': mean_brightness,
                    'std_brightness': std_brightness,
                    'contact_type': contact_type,
                    'status': status,
                    'contrast': contrast,
                    'surrounding_brightness': surrounding_mean,
                    'shape_type': shape_type  # 确保shape_type被保留
                })
                
                analyzed_contacts.append(contact)
        
        return analyzed_contacts
    
    def find_abnormal_contacts(self, contacts):
        """找出异常Contact"""
        abnormal = []
        
        for contact in contacts:
            # 异常判断标准：
            # 1. 亮度异常（浮空节点）
            # 2. 对比度异常（与周围差异过大或过小）
            # 3. 圆形度异常（形状不规则）
            
            if contact['status'] == "异常":
                abnormal.append(contact)
            elif contact['contrast'] < 10:  # 对比度过小，可能是短路
                contact['status'] = "异常-可能短路"
                abnormal.append(contact)
            elif contact['contrast'] > 100:  # 对比度过大，可能是开路
                contact['status'] = "异常-可能开路"
                abnormal.append(contact)
            # 检查形状是否规则（对于已识别的圆形或方形，不需要再次检查形状）
            # 形状检测已在detect_contacts中完成，这里只检查电气异常
        
        return abnormal
    
    def compare_images(self, img1, img2, min_circularity=0.65, min_rectangularity=0.80):
        """对比两张PVC图像，找出差异Contact"""
        # 预处理
        proc1, _ = self.preprocess_image(img1)
        proc2, _ = self.preprocess_image(img2)
        
        # 检测Contact
        contacts1, _ = self.detect_contacts(proc1, min_circularity=min_circularity, 
                                           min_rectangularity=min_rectangularity)
        contacts2, _ = self.detect_contacts(proc2, min_circularity=min_circularity,
                                           min_rectangularity=min_rectangularity)
        
        # 分析亮度
        analyzed1 = self.analyze_contact_brightness(proc1, contacts1)
        analyzed2 = self.analyze_contact_brightness(proc2, contacts2)
        
        # 找出差异
        differences = []
        
        # 基于位置匹配Contact
        for c1 in analyzed1:
            min_dist = float('inf')
            matched_c2 = None
            
            for c2 in analyzed2:
                dist = np.sqrt((c1['center'][0] - c2['center'][0])**2 + 
                              (c1['center'][1] - c2['center'][1])**2)
                if dist < min_dist and dist < 50:  # 匹配阈值
                    min_dist = dist
                    matched_c2 = c2
            
            if matched_c2 is not None:
                # 比较亮度差异
                brightness_diff = abs(c1['mean_brightness'] - matched_c2['mean_brightness'])
                if brightness_diff > 30:  # 亮度差异阈值
                    differences.append({
                        'contact1': c1,
                        'contact2': matched_c2,
                        'brightness_diff': brightness_diff,
                        'position': c1['center']
                    })
        
        return differences, analyzed1, analyzed2
    
    def visualize_results(self, image, contacts, abnormal_contacts=None, title="Contact检测结果"):
        """可视化检测结果"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # 统计信息
        normal_count = 0
        abnormal_count = 0
        circle_count = 0
        square_count = 0
        
        # 绘制所有Contact
        for contact in contacts:
            center = contact['center']
            shape_type = contact.get('shape_type', '未知')
            
            # 统计形状
            if shape_type == "圆形":
                circle_count += 1
            elif shape_type == "方形":
                square_count += 1
            
            # 确定颜色和线宽
            is_abnormal = abnormal_contacts and contact in abnormal_contacts
            if is_abnormal:
                color = 'red'
                linewidth = 2.5
                alpha = 0.8
                abnormal_count += 1
            elif contact.get('status', '').startswith('正常'):
                color = 'green'
                linewidth = 1.5
                alpha = 0.6
                normal_count += 1
            else:
                color = 'yellow'
                linewidth = 1.5
                alpha = 0.6
            
            # 根据形状绘制不同的标记
            if shape_type == "圆形":
                # 圆形Contact：绘制圆形边界
                x, y, w, h = contact['bbox']
                radius = max(w, h) / 2
                circle = Circle(center, radius, linewidth=linewidth, 
                               edgecolor=color, facecolor='none', alpha=alpha)
                ax.add_patch(circle)
                # 中心点
                center_dot = Circle(center, 2, color=color, fill=True, alpha=alpha)
                ax.add_patch(center_dot)
            elif shape_type == "方形":
                # 方形Contact：绘制矩形边界
                x, y, w, h = contact['bbox']
                rect = Rectangle((x, y), w, h, linewidth=linewidth,
                               edgecolor=color, facecolor='none', alpha=alpha)
                ax.add_patch(rect)
                # 中心点
                center_dot = Circle(center, 2, color=color, fill=True, alpha=alpha)
                ax.add_patch(center_dot)
            else:
                # 未知形状：使用边界框
                x, y, w, h = contact['bbox']
                rect = Rectangle((x, y), w, h, linewidth=linewidth,
                               edgecolor=color, facecolor='none', alpha=alpha)
                ax.add_patch(rect)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='green', linewidth=2, label=f'正常Contact ({normal_count})'),
            Patch(facecolor='none', edgecolor='red', linewidth=2, label=f'异常Contact ({abnormal_count})'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 添加统计信息文本
        info_text = f'总计: {len(contacts)} | 圆形: {circle_count} | 方形: {square_count}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig


def main():
    st.title("🔬 PVC Contact差异识别系统")
    st.markdown("""
    ### Passive Voltage Contrast (PVC) Contact自动检测与差异分析
    
    本系统基于PVC技术原理，自动检测和识别SEM图像中的Contact区域，并通过亮度分析判断其电气状态：
    - **亮区域**：高电位（VDD）
    - **暗区域**：低电位（GND）
    - **异常区域**：浮空、开路或短路
    """)
    
    analyzer = PVCContactAnalyzer()
    
    # 侧边栏参数设置
    st.sidebar.header("⚙️ 参数设置")
    st.sidebar.subheader("Contact尺寸")
    analyzer.min_contact_size = st.sidebar.slider("最小Contact尺寸", 3, 20, 5)
    analyzer.max_contact_size = st.sidebar.slider("最大Contact尺寸", 50, 200, 100)
    
    st.sidebar.subheader("亮度阈值")
    analyzer.brightness_threshold_high = st.sidebar.slider("高亮度阈值", 100, 255, 180)
    analyzer.brightness_threshold_low = st.sidebar.slider("低亮度阈值", 0, 150, 80)
    
    st.sidebar.subheader("形状检测")
    min_circularity = st.sidebar.slider("最小圆形度", 0.5, 0.9, 0.65, 0.05)
    min_rectangularity = st.sidebar.slider("最小矩形度", 0.7, 0.95, 0.80, 0.05)
    
    # 将参数传递给检测函数（需要在检测时使用）
    st.sidebar.info("💡 提示：Contact只识别圆形和方形，不规则形状会被过滤")
    
    # 功能选择
    tab1, tab2, tab3 = st.tabs(["📸 单图分析", "🔍 双图对比", "📂 批量处理"])
    
    with tab1:
        st.header("单张PVC图像分析")
        
        # 图像上传或选择demo
        image_source = st.radio("选择图像来源", ["Demo图片", "上传图片"])
        
        if image_source == "Demo图片":
            demo_path = "VC_images/Service_MA_SEM_09.jpg"
            if os.path.exists(demo_path):
                image = cv2.imread(demo_path)
                st.info(f"使用Demo图片: {demo_path}")
            else:
                st.error("Demo图片不存在！")
                return
        else:
            uploaded_file = st.file_uploader("上传PVC图像", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])
            if uploaded_file is None:
                st.info("请上传一张PVC图像")
                return
            
            # 读取上传的图像
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("原图")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # 处理图像
            processed, denoised = analyzer.preprocess_image(image)
            contacts, binary = analyzer.detect_contacts(processed, 
                                                       min_circularity=min_circularity,
                                                       min_rectangularity=min_rectangularity)
            analyzed_contacts = analyzer.analyze_contact_brightness(processed, contacts)
            abnormal_contacts = analyzer.find_abnormal_contacts(analyzed_contacts)
            
            # 显示统计信息
            st.subheader("📊 检测统计")
            circle_count = sum(1 for c in analyzed_contacts if c.get('shape_type') == '圆形')
            square_count = sum(1 for c in analyzed_contacts if c.get('shape_type') == '方形')
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("检测到Contact总数", len(analyzed_contacts))
            with col2:
                st.metric("圆形Contact", circle_count)
            with col3:
                st.metric("方形Contact", square_count)
            with col4:
                st.metric("异常Contact", len(abnormal_contacts))
            with col5:
                if len(analyzed_contacts) > 0:
                    st.metric("异常率", f"{len(abnormal_contacts)/len(analyzed_contacts)*100:.1f}%")
            
            # 可视化结果
            fig = analyzer.visualize_results(processed, analyzed_contacts, abnormal_contacts)
            st.pyplot(fig)
            
            # 显示异常Contact详情
            if abnormal_contacts:
                st.subheader("⚠️ 异常Contact详情")
                for i, contact in enumerate(abnormal_contacts, 1):
                    shape_type = contact.get('shape_type', '未知')
                    with st.expander(f"异常Contact #{i} - {shape_type} - {contact['status']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**形状**: {shape_type}")
                            st.write(f"**位置**: ({contact['center'][0]}, {contact['center'][1]})")
                            st.write(f"**平均亮度**: {contact['mean_brightness']:.2f}")
                            st.write(f"**亮度标准差**: {contact['std_brightness']:.2f}")
                        with col2:
                            st.write(f"**对比度**: {contact['contrast']:.2f}")
                            if shape_type == "圆形":
                                st.write(f"**圆形度**: {contact.get('circularity', 0):.2f}")
                            elif shape_type == "方形":
                                st.write(f"**矩形度**: {contact.get('rectangularity', 0):.2f}")
                            st.write(f"**面积**: {contact['area']:.2f} 像素²")
            
            # Contact统计表
            if analyzed_contacts:
                st.subheader("📋 Contact详细信息")
                df_data = []
                for contact in analyzed_contacts:
                    shape_type = contact.get('shape_type', '未知')
                    df_data.append({
                        'ID': analyzed_contacts.index(contact) + 1,
                        '形状': shape_type,
                        '位置(X,Y)': f"({contact['center'][0]}, {contact['center'][1]})",
                        '平均亮度': f"{contact['mean_brightness']:.1f}",
                        '类型': contact['contact_type'],
                        '状态': contact['status'],
                        '对比度': f"{contact['contrast']:.1f}",
                        '圆形度': f"{contact.get('circularity', 0):.2f}",
                        '矩形度': f"{contact.get('rectangularity', 0):.2f}"
                    })
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.header("两张PVC图像对比分析")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("参考图像")
            ref_source = st.radio("参考图来源", ["Demo图片", "上传图片"], key="ref")
            if ref_source == "Demo图片":
                ref_path = "VC_images/Service_MA_SEM_09.jpg"
                if os.path.exists(ref_path):
                    ref_image = cv2.imread(ref_path)
                    st.image(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    ref_image = None
                    st.error("Demo图片不存在！")
            else:
                ref_file = st.file_uploader("上传参考图像", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], key="ref_upload")
                if ref_file:
                    file_bytes = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
                    ref_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    st.image(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    ref_image = None
        
        with col2:
            st.subheader("对比图像")
            test_source = st.radio("对比图来源", ["Demo图片", "上传图片"], key="test")
            if test_source == "Demo图片":
                test_path = "VC_images/Service_MA_SEM_09.jpg"
                if os.path.exists(test_path):
                    test_image = cv2.imread(test_path)
                    st.image(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    test_image = None
                    st.error("Demo图片不存在！")
            else:
                test_file = st.file_uploader("上传对比图像", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'], key="test_upload")
                if test_file:
                    file_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
                    test_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    st.image(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    test_image = None
        
        if ref_image is not None and test_image is not None:
            if st.button("🔍 开始对比分析", type="primary"):
                with st.spinner("正在分析图像..."):
                        differences, contacts1, contacts2 = analyzer.compare_images(
                            ref_image, test_image,
                            min_circularity=min_circularity,
                            min_rectangularity=min_rectangularity
                        )
                
                st.success(f"分析完成！发现 {len(differences)} 个差异Contact")
                
                # 显示差异统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("参考图Contact数", len(contacts1))
                with col2:
                    st.metric("对比图Contact数", len(contacts2))
                with col3:
                    st.metric("差异Contact数", len(differences))
                
                # 显示差异详情
                if differences:
                    st.subheader("🔴 差异Contact列表")
                    for i, diff in enumerate(differences, 1):
                        with st.expander(f"差异 #{i} - 位置: {diff['position']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**参考图像Contact**")
                                st.write(f"亮度: {diff['contact1']['mean_brightness']:.2f}")
                                st.write(f"类型: {diff['contact1']['contact_type']}")
                                st.write(f"状态: {diff['contact1']['status']}")
                            with col2:
                                st.write("**对比图像Contact**")
                                st.write(f"亮度: {diff['contact2']['mean_brightness']:.2f}")
                                st.write(f"类型: {diff['contact2']['contact_type']}")
                                st.write(f"状态: {diff['contact2']['status']}")
                            st.write(f"**亮度差异**: {diff['brightness_diff']:.2f}")
    
    with tab3:
        st.header("批量处理")
        st.info("批量处理功能开发中...")
        st.write("该功能将支持批量处理多张PVC图像，并生成对比报告。")


if __name__ == "__main__":
    main()

