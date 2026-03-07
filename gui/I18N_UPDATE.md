# i18n 本地化与主题切换更新

## 完成的更新

### 1. 完整的本地化 (i18n) 支持

所有文本元素现在支持 4 种语言：
- 🇨🇳 中文 (zh)
- 🇺🇸 English (en)
- 🇯🇵 日本語 (ja)
- 🇰🇷 한국어 (ko)

#### 更新的文件：
- `utils/i18n.py` - 添加所有翻译键
- `main.py` - 所有文本使用 `t()` 函数
- `wizard/step0_setup.py` - 添加 i18n 导入
- `wizard/step1_tagging.py` - 添加 i18n 导入
- `wizard/step2_cache.py` - 添加 i18n 导入
- `wizard/step3_train.py` - 添加 i18n 导入和部分文本翻译
- `wizard/step4_generate.py` - 添加 i18n 导入
- `launch.py` - 添加语言显示

#### 可用的翻译键：
```python
# 导航
nav_home, nav_tagging, nav_cache, nav_train, nav_generate, nav_setup

# 首页
quick_start, supported_models, features, model_architectures
app_description, step, support
model_architecture_list['flux2', 'wan', 'hunyuan', ...]
feature_list['efficient', 'multi_arch', 'workflow', 'preset', 'monitor', 'modern_ui']

# 训练页面
basic_settings, model_settings, network_settings, optimizer_settings
basic_train_params, lr_settings, timestep_sampling
memory_optimization, sampling_settings, save_precision

# 通用
browse, save, load, start, stop, clear, refresh, delete, edit
cancel, confirm, close, status_on, status_off

# 提示
tt_dim, tt_alpha, tt_lr, tt_epochs, tt_batch
```

### 2. 修复主题切换功能

主题切换按钮现在可以正常工作：
- 点击太阳/月亮图标切换深色/浅色主题
- 使用 `localStorage` 保存主题偏好
- 页面加载时自动应用保存的主题
- 通过 JavaScript 直接操作 DOM，响应更快

#### 实现方式：
```javascript
// 在 home_page() 中添加的 JavaScript
window.toggleDarkMode = function() {
    const isDark = document.body.classList.toggle('dark-mode');
    localStorage.setItem('dark_mode', isDark);
    // 更新图标
    // 保存到服务器
}
```

### 3. 增强的样式覆盖

修复了输入框和下拉框的样式：
- `.q-field__control` - 输入框控件样式
- `.q-field--outlined .q-field__control` -  outlined 输入框
- `.q-field--filled .q-field__control` - filled 输入框
- 添加了 focus 状态的边框和阴影效果
- 统一了高度 (min-height: 48px)

## 使用方法

### 切换语言
1. 点击页面右上角的语言选择下拉框
2. 选择需要的语言
3. 页面会自动刷新并应用新语言

### 切换主题
1. 点击页面右上角的太阳/月亮图标
2. 主题会在深色和浅色之间切换
3. 偏好会自动保存

## 添加新翻译

在 `utils/i18n.py` 中添加新的翻译键：

```python
TRANSLATIONS = {
    'en': {
        'new_key': 'English Text',
    },
    'zh': {
        'new_key': '中文文本',
    },
    'ja': {
        'new_key': '日本語テキスト',
    },
    'ko': {
        'new_key': '한국어 텍스트',
    },
}
```

然后在代码中使用：
```python
from utils.i18n import t

ui.label(t('new_key'))
```

## 文件修改清单

- ✅ `utils/i18n.py` - 添加完整翻译键
- ✅ `theme.py` - 添加主题切换样式和函数
- ✅ `main.py` - 完整的本地化和主题切换
- ✅ `launch.py` - 添加语言显示
- ✅ `wizard/step0_setup.py` - 添加 i18n 导入
- ✅ `wizard/step1_tagging.py` - 添加 i18n 导入
- ✅ `wizard/step2_cache.py` - 添加 i18n 导入
- ✅ `wizard/step3_train.py` - 添加 i18n 导入和部分翻译
- ✅ `wizard/step4_generate.py` - 添加 i18n 导入
