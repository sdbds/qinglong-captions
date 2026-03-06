# 样式复用说明

本文档说明了如何从 `D:\lora-scripts\sd-scripts\gui` 复用样式系统。

## 复用的内容

### 1. 主题系统 (theme.py)

整合了来自 `sd-scripts/gui/styles.py` 的 Green/Gold 主题到现有的 Modern 主题系统中。

#### 可用的主题

1. **Modern Theme (默认)** - 深绿+金色自然主题
   - 深色背景，现代化卡片和按钮
   - 使用方式: `apply_theme()` 或 `apply_theme('modern')`

2. **Green Gold Theme (可选)** - 明亮绿金主题 (来自 sd-scripts)
   - 明亮背景，传统的绿金配色
   - 使用方式: `apply_theme('green-gold')` 或 `apply_green_gold_styles()`

#### 颜色配置

```python
from theme import COLORS, apply_theme, get_classes

# 获取颜色
primary_color = COLORS['primary']
accent_color = COLORS['accent']
```

#### CSS 类映射

| 类名 | 用途 | 来源 |
|------|------|------|
| `modern-card` | 现代化卡片 | Modern |
| `modern-btn-primary` | 主按钮 | Modern |
| `modern-btn-secondary` | 次按钮 | Modern |
| `modern-nav-btn` | 导航按钮 | Modern |
| `section-card` | 分区卡片 | sd-scripts |
| `gold-btn` | 金色主按钮 | sd-scripts |
| `green-btn` | 绿色浏览按钮 | sd-scripts |
| `red-btn` | 红色危险按钮 | sd-scripts |
| `section-title` | 分区标题 | sd-scripts |
| `toggle-container` | 切换开关 | sd-scripts |
| `editable-slider` | 可编辑滑块 | sd-scripts |

### 2. 国际化模块 (utils/i18n.py)

复用自 `sd-scripts/gui/i18n.py`，适配 Musubi Tuner GUI 的术语。

#### 支持的语言

- English (en)
- 中文 (zh)
- 日本語 (ja)
- 한국어 (ko)

#### 使用方法

```python
from utils.i18n import t, set_language, tl

# 获取翻译文本
label = t('nav_home')  # 返回 "首页" (根据当前语言)

# 创建可翻译标签
from nicegui import ui
ui.label(t('app_title'))

# 切换语言
set_language('en')  # 切换到英语
```

#### 已适配的翻译键

- `nav_home`, `nav_tagging`, `nav_cache`, `nav_train`, `nav_generate`
- `dataset_tagging`, `cache_processing`, `train_lora`, `inference`
- `start_train`, `stop_train`, `save_preset`, `load_preset`
- `train_log`, `train_started`, `train_finished`

## 使用示例

### 应用主题

```python
from nicegui import ui
from theme import apply_theme, get_classes, COLORS

# 应用 Modern 主题 (默认)
apply_theme()

# 或应用 Green Gold 主题
apply_theme('green-gold')
```

### 使用样式类

```python
from nicegui import ui
from theme import get_classes, apply_button, apply_card

# 使用 get_classes
with ui.card().classes(get_classes('card')):
    ui.label('内容')

# 使用便捷函数
btn = ui.button('点击')
apply_button(btn, variant='primary')  # 应用主按钮样式

card = ui.card()
apply_card(card, hover=True)  # 应用卡片样式，带悬停效果
```

### 创建 sd-scripts 风格的卡片

```python
from nicegui import ui
from theme import get_classes, COLORS

with ui.card().classes(get_classes('section_card')):
    ui.label('分区标题').classes('section-title')
    ui.label('内容')
```

### 创建切换开关

```python
from nicegui import ui

def create_toggle(label: str, value: bool = False):
    with ui.button().classes(f'toggle-container {"active" if value else ""}'):
        with ui.element('div').classes('toggle-switch'):
            ui.element('div').classes('toggle-knob')
        ui.label(label).classes('toggle-label')
        ui.label('ON' if value else 'OFF').classes('toggle-status')
```

### 创建可编辑滑块

```python
from nicegui import ui

def create_editable_slider(label: str, min_val: float, max_val: float, value: float):
    with ui.element('div').classes('editable-slider'):
        with ui.row().classes('slider-label-row'):
            ui.label(label).classes('slider-label')
            ui.button(f'{value:.2f}').classes('slider-value')
        with ui.element('div').classes('slider-container'):
            ui.element('div').classes('slider-track')
            ui.element('div').classes('slider-fill').style(f'width: {(value-min_val)/(max_val-min_val)*100}%')
            ui.element('div').classes('slider-thumb').style(f'left: {(value-min_val)/(max_val-min_val)*100}%')
```

## 自定义主题

### 修改颜色

```python
from theme import COLORS

# 修改主题颜色
COLORS['primary'] = '#custom_color'

# 重新应用主题
apply_theme()
```

### 添加自定义 CSS

```python
from nicegui import ui

ui.add_css('''
    .my-custom-class {
        background: linear-gradient(135deg, #custom1, #custom2);
    }
''')
```

## 文件对应关系

| 原文件 | 当前文件 | 说明 |
|--------|----------|------|
| `sd-scripts/gui/styles.py` | `theme.py` | 整合到现有主题系统 |
| `sd-scripts/gui/i18n.py` | `utils/i18n.py` | 适配后复用 |

## 注意事项

1. **主题切换**: Modern 和 Green Gold 主题使用不同的颜色配置，切换时会更新全局 `COLORS` 变量
2. **CSS 优先级**: 使用 `!important` 确保样式覆盖默认样式
3. **深色模式**: Green Gold 主题包含完整的深色模式样式支持
4. **兼容性**: 所有新增的样式类都与现有代码兼容
