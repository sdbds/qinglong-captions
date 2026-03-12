"""
青龙字幕工具 GUI (Qinglong Captions GUI)
基于 NiceGUI 的现代化图形化界面

使用方法:
    cd gui
    python main.py

或者:
    python gui/main.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import toml
from nicegui import ui, app
from wizard.step0_setup import render_setup_step
from wizard.step1_import import render_import_step
from wizard.step2_video_split import render_video_split_step
from wizard.step3_tagger import render_tagger_step
from wizard.step4_caption import render_caption_step
from wizard.step5_export import render_export_step
from wizard.step6_tools import render_tools_step
from wizard.step7_settings import create_settings_dialog
from wizard.console_page import render_console_page
from theme import apply_theme, get_classes, COLORS
from gui.utils.i18n import t, set_language, get_i18n


def _load_app_version() -> str:
    """Read GUI version from the project metadata instead of duplicating constants."""
    try:
        pyproject = toml.load(project_root / "pyproject.toml")
        return str(pyproject.get("project", {}).get("version", "")).strip() or "0.0.0"
    except Exception:
        return "0.0.0"


# 页面标题和样式
APP_TITLE = "青龙字幕工具"
APP_TITLE_EN = "Qinglong Captions"
APP_VERSION = _load_app_version()

# 全局主题脚本
THEME_SCRIPT = """
<script>
(function() {
    // Sync Quasar Dark mode with body.dark-mode CSS class
    function syncQuasarDark(isDark) {
        if (window.Quasar && Quasar.Dark) {
            Quasar.Dark.set(isDark);
        }
        // Override Quasar --q-primary to Gold/Amber
        document.documentElement.style.setProperty('--q-primary', isDark ? '#fbbf24' : '#f59e0b', 'important');
        document.documentElement.style.setProperty('--q-color-primary', isDark ? '#fbbf24' : '#f59e0b', 'important');
    }

    // Theme toggle function
    window.toggleDarkMode = function() {
        const isDark = document.body.classList.toggle('dark-mode');
        localStorage.setItem('dark_mode', isDark);
        syncQuasarDark(isDark);
        // Update theme button icons
        document.querySelectorAll('.theme-toggle-btn .q-icon').forEach(function(icon) {
            icon.textContent = isDark ? 'dark_mode' : 'light_mode';
        });
        return isDark;
    };

    // Apply saved theme (default dark)
    var saved = localStorage.getItem('dark_mode');
    var isDark = saved === null ? true : saved === 'true';
    if (isDark) {
        document.body.classList.add('dark-mode');
    } else {
        document.body.classList.remove('dark-mode');
    }
    syncQuasarDark(isDark);

    // Update button icons
    document.querySelectorAll('.theme-toggle-btn .q-icon').forEach(function(icon) {
        icon.textContent = isDark ? 'dark_mode' : 'light_mode';
    });

    // Quasar may load late, retry sync
    setTimeout(function() { syncQuasarDark(isDark); }, 300);
})();
</script>
"""


def create_header():
    """创建现代化页面头部导航"""
    with ui.header().classes(get_classes("header")):
        with ui.row().classes("w-full items-center justify-between q-py-sm"):
            # Logo 区域
            with ui.row().classes("items-center gap-3"):
                with (
                    ui.element("div")
                    .classes("flex items-center justify-center")
                    .style(f"""
                    width: 40px;
                    height: 40px;
                    background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]});
                    border-radius: 12px;
                    font-size: 24px;
                """)
                ):
                    ui.label("🐉").classes("text-2xl")
                with ui.column().classes("gap-0"):
                    ui.label(t("app_title")).classes("text-h6 text-weight-bold header-title")
                    ui.label(f"v{APP_VERSION}").classes("text-caption header-version")

            # 导航菜单
            with ui.row().classes("gap-2"):
                nav_items = [
                    (t("nav_home"), "/", "home"),
                    (t("nav_import"), "/import", "download"),
                    (t("nav_split"), "/split", "movie"),
                    (t("nav_tagger"), "/tagger", "label"),
                    (t("nav_caption"), "/caption", "subtitles"),
                    (t("nav_export"), "/export", "upload"),
                    (t("nav_tools"), "/tools", "construction"),
                ]

                for label_text, path, icon in nav_items:
                    btn = ui.button(label_text, on_click=lambda p=path: ui.navigate.to(p), icon=icon)
                    btn.classes(get_classes("nav_btn"))
                    # 移除 Quasar 默认颜色类，防止闪烁
                    btn.classes(remove="bg-primary bg-secondary text-white q-btn--active")

            # 右侧：配置 + 主题切换 + 语言选择
            with ui.row().classes("items-center gap-3"):
                # 配置管理按钮
                settings_dlg = create_settings_dialog()
                settings_btn = ui.button(icon="settings", on_click=settings_dlg.open).props("flat round dense")
                settings_btn.style(f"color: {COLORS['accent']};")
                settings_btn.tooltip(t("nav_settings"))

                # 分隔线
                ui.element("div").style(f"width: 1px; height: 24px; background: {COLORS['accent']}; opacity: 0.5;")

                # 主题切换按钮
                theme_btn = ui.button(icon="light_mode").props("flat round dense").classes("theme-toggle-btn")
                theme_btn.on_click(lambda: ui.run_javascript("window.toggleDarkMode()"))

                # 分隔线
                ui.element("div").style(f"width: 1px; height: 24px; background: {COLORS['accent']}; opacity: 0.5;")

                # 语言选择器
                with ui.row().classes("items-center gap-2"):
                    ui.icon("language", size="20px").style(f"color: {COLORS['accent']};")
                    lang_select = (
                        ui.select(
                            {
                                "zh": "🇨🇳 中文",
                                "en": "🇺🇸 English",
                                "ja": "🇯🇵 日本語",
                                "ko": "🇰🇷 한국어",
                            },
                            label="",
                            value=get_i18n().lang,
                        )
                        .props('dense outlined use-input fill-input hide-selected input-debounce="0" dropdown-icon="expand_more"')
                        .classes("lang-selector")
                    )

                    def on_lang_change(e):
                        lang = e.value
                        if lang and lang in ["zh", "en", "ja", "ko"]:
                            set_language(lang)
                            ui.notify(t("language_changed"), type="positive")
                            # 刷新页面以应用新语言
                            ui.run_javascript("window.location.reload();")

                    lang_select.on_value_change(on_lang_change)


def page_base(content_func):
    """页面基础包装器 - 应用主题和脚本"""
    # 应用主题
    apply_theme()

    # 添加主题脚本（只执行一次）
    ui.add_body_html(THEME_SCRIPT)

    # 创建头部
    create_header()

    # 执行页面内容
    content_func()


def home_page():
    """首页/欢迎页面"""

    def content():
        with ui.column().classes(get_classes("page_container") + " gap-6"):
            # Hero Section
            with ui.element("div").classes("w-full text-center q-py-xl"):
                ui.label(t("app_title")).classes("text-h2 text-weight-bold q-mb-md").style(f"""
                    background: linear-gradient(135deg, {COLORS["primary_light"]}, {COLORS["secondary"]});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                """)
                ui.label(t("app_description")).classes("text-h6 app-desc")

            # Quick Start Cards
            with ui.card().classes(get_classes("card") + " w-full q-pa-lg"):
                with ui.row().classes("w-full items-center justify-between q-mb-lg"):
                    ui.label(t("quick_start")).classes("text-h5 text-weight-bold section-title")
                    with ui.row().classes("gap-2"):
                        ui.label(t("support")).classes("text-caption section-subtitle")
                        ui.label("6").classes(get_classes("badge") + " modern-badge-primary").style("padding: 2px 10px;")
                        ui.label(t("workflow_steps")).classes("text-caption section-subtitle")

                with ui.row().classes("w-full gap-4"):
                    steps = [
                        ("download", t("step") + " 1", t("dataset_import"), t("feature_list")["import_desc"], "/import"),
                        ("movie", t("step") + " 2", t("video_split"), t("feature_list")["split_desc"], "/split"),
                        ("label", t("step") + " 3", t("dataset_tagging"), t("feature_list")["tagger_desc"], "/tagger"),
                        ("subtitles", t("step") + " 4", t("caption_generate"), t("feature_list")["caption_desc"], "/caption"),
                        ("upload", t("step") + " 5", t("dataset_export"), t("feature_list")["export_desc"], "/export"),
                        ("construction", t("step") + " 6", t("tools"), t("feature_list")["tools_desc"], "/tools"),
                    ]

                    for icon, step_num, title, desc, path in steps:
                        with ui.card().classes("step-card flex-1").on("click", lambda p=path: ui.navigate.to(p)):
                            ui.icon(icon, size="48px").classes("q-mb-md")
                            ui.label(step_num).classes("text-caption text-uppercase q-mb-xs step-label")
                            ui.label(title).classes("text-h6 text-weight-bold q-mb-sm step-title")
                            ui.label(desc).classes("text-body2 step-desc")

            # Supported Models
            with ui.card().classes(get_classes("card") + " w-full q-pa-lg"):
                with ui.row().classes("w-full items-center gap-3 q-mb-lg"):
                    ui.icon("view_module", size="28px").style(f"color: {COLORS['primary']};")
                    ui.label(t("supported_models")).classes("text-h5 text-weight-bold section-title")

                model_list = t("model_list")
                models = [
                    ("Gemini", model_list["gemini"], "primary"),
                    ("Mistral OCR", model_list["pixtral"], "secondary"),
                    ("Step-VL", model_list["step"], "accent"),
                    ("Qwen-VL", model_list["qwen"], "primary"),
                    ("Kimi", model_list["kimi"], "secondary"),
                    ("GLM", model_list["glm"], "accent"),
                ]

                with ui.grid(columns=3).classes("w-full gap-4"):
                    for name, desc, color_key in models:
                        with ui.row().classes("model-item items-center gap-3"):
                            ui.icon("check_circle", size="20px").style(f"color: {COLORS[color_key]};")
                            with ui.column().classes("gap-0"):
                                ui.label(name).classes("text-body2 text-weight-bold model-name")
                                ui.label(desc).classes("text-caption model-desc")

            # Features Section
            with ui.card().classes(get_classes("card") + " w-full q-pa-lg"):
                with ui.row().classes("w-full items-center gap-3 q-mb-lg"):
                    ui.icon("auto_awesome", size="28px").style(f"color: {COLORS['secondary']};")
                    ui.label(t("features")).classes("text-h5 text-weight-bold section-title")

                feature_list = t("feature_list")
                features = [
                    ("🎬", feature_list["video_caption"]),
                    ("🏷️", feature_list["auto_tagging"]),
                    ("🤖", feature_list["multi_api"]),
                    ("💾", feature_list["lance_db"]),
                    ("📊", feature_list["batch_process"]),
                    ("🎨", feature_list["modern_ui"]),
                ]

                with ui.grid(columns=3).classes("w-full gap-4"):
                    for icon, desc in features:
                        with ui.row().classes("model-item items-start gap-3"):
                            ui.label(icon).classes("text-2xl")
                            with ui.column().classes("gap-0"):
                                ui.label(desc).classes("text-body2 feature-desc")

    page_base(content)


def import_page():
    """数据导入页面"""
    page_base(render_import_step)


def split_page():
    """视频分割页面"""
    page_base(render_video_split_step)


def tagger_page():
    """标签生成页面"""
    page_base(render_tagger_step)


def caption_page():
    """字幕生成页面"""
    page_base(render_caption_step)


def export_page():
    """数据导出页面"""
    page_base(render_export_step)


def tools_page():
    """工具页面"""
    page_base(render_tools_step)


def setup_page():
    """环境检查页面"""
    page_base(render_setup_step)


def console_page():
    """全屏终端页面（无导航栏）"""
    render_console_page()


def not_found_page():
    """404 页面 - 页面不存在"""

    def content():
        with ui.column().classes(get_classes("page_container") + " gap-6 items-center justify-center").style("min-height: 60vh;"):
            ui.icon("error_outline", size="80px").style(f"color: {COLORS['warning']};")
            ui.label(t("page_not_found")).classes("text-h3 text-weight-bold").style("color: var(--color-text);")
            ui.label(t("page_not_found_desc")).classes("text-body1").style("color: var(--color-text-secondary);")

            with ui.row().classes("gap-4 q-mt-lg"):
                home_btn = ui.button(t("back_to_home"), on_click=lambda: ui.navigate.to("/"), icon="home")
                home_btn.classes("modern-btn-primary")
                home_btn.classes(remove="bg-primary bg-secondary text-white")

    page_base(content)


def main():
    """主函数"""
    # 设置页面路由
    ui.page("/")(home_page)
    ui.page("/import")(import_page)
    ui.page("/split")(split_page)
    ui.page("/tagger")(tagger_page)
    ui.page("/caption")(caption_page)
    ui.page("/export")(export_page)
    ui.page("/tools")(tools_page)
    ui.page("/setup")(setup_page)
    ui.page("/console")(console_page)

    # 设置应用标题
    app.config.title = t("app_title")

    # 404 页面 - 捕获所有未定义的路由
    ui.page("/{path:path}")(not_found_page)

    # 运行应用
    ui.run(
        title=t("app_title"),
        favicon="🐉",
        dark=False,
        reload=False,
        port=8080,
        show=True,
        native=False,
    )


if __name__ == "__main__":
    main()
